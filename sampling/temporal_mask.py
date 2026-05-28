"""
Temporally-correlated k-space sampling mask.

Learns a joint Cartesian sampling pattern across all DCE phases, exploiting
the fact that phases share low-frequency anatomy but differ in the
contrast-enhanced tumour region.

Design:
  - A shared base logit map S ∈ ℝ^H encodes anatomy (1-D row selection).
  - Per-phase delta maps D_t ∈ ℝ^H capture temporal variation.
  - Mask_t rows = top-(H/R) rows by (S + D_t), with the ACS region always included.
  - Straight-through estimator lets gradients flow during training.

NOTE: Breast DCE-MRI uses Cartesian phase-encoding, so masks select *full rows*
of k-space (phase-encode lines), not individual points.  The 1-D logit therefore
has shape [H], not [H, W].  The resulting mask is broadcast to [H, W] by
expanding along the readout (W) dimension.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class TemporalKSpaceSampler(nn.Module):
    """
    Learns per-phase 1-D Cartesian k-space masks for T DCE phases jointly.

    The ACS (auto-calibration signal) region at the centre of k-space is
    always fully sampled.  The remaining `target_lines - acs_lines` rows are
    selected by the learned logit.
    """

    def __init__(
        self,
        acceleration: float = 6.0,
        n_phases: int = 3,
        shape: Tuple[int, int] = (320, 320),
        acs_fraction: float = 0.08,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.R           = acceleration
        self.T           = n_phases
        self.H, self.W   = shape
        self.temperature = temperature

        acs_lines      = int(self.H * acs_fraction)
        self.acs_lines = acs_lines

        # BUG FIX 1: target lines = H / R (row budget), not H*W/R (point budget)
        self.target_lines = max(1, int(self.H / self.R))

        # BUG FIX 2: initialise with small random noise so phases start different
        self.base_logits  = nn.Parameter(torch.randn(self.H) * 0.1)
        self.phase_deltas = nn.Parameter(torch.randn(n_phases, self.H) * 0.1)

        # Fixed ACS row mask (not learned)
        acs_mask = torch.zeros(self.H)
        centre   = self.H // 2
        half     = acs_lines // 2
        acs_mask[centre - half: centre + half] = 1.0
        self.register_buffer("acs_mask", acs_mask)    # [H]

        # Pre-compute ACS row indices for exclusion in random sampling
        self.register_buffer(
            "acs_indices",
            torch.where(acs_mask == 1.0)[0],
        )

    def _soft_logits(self) -> Tensor:
        """Return per-phase logit vectors [T, H]."""
        return self.base_logits.unsqueeze(0) + self.phase_deltas   # [T, H]

    def _binarise(self, logits: Tensor) -> Tensor:
        """
        Select exactly `target_lines` rows per phase using the logits as scores.
        ACS rows are always selected; remaining budget filled by top-ranked non-ACS rows.
        Uses straight-through estimator so gradients flow during training.

        Args:
            logits: [T, H]
        Returns:
            masks:  [T, H]  binary (float, straight-through)
        """
        T, H     = logits.shape
        acs_rows = self.acs_lines
        extra    = max(0, self.target_lines - acs_rows)   # non-ACS rows to pick

        masks = []
        for t in range(T):
            l = logits[t].clone()
            # Suppress ACS rows in scoring (they are always included separately)
            l[self.acs_indices] = -1e9

            # Pick top-`extra` non-ACS rows
            _, top_idx = l.topk(extra, sorted=False)
            binary = self.acs_mask.clone()           # start with ACS rows set
            binary[top_idx] = 1.0

            if self.training:
                # Straight-through: forward=binary, backward through soft logits
                soft   = torch.sigmoid(logits[t] / self.temperature)
                binary = binary - soft.detach() + soft
            masks.append(binary)

        return torch.stack(masks, dim=0)             # [T, H]

    def forward(self) -> Tensor:
        """
        Returns:
            masks: [T, H, W]  Cartesian masks (each row either fully sampled or zero)
        """
        logits = self._soft_logits()                 # [T, H]
        masks  = self._binarise(logits)              # [T, H]
        # .expand() returns a non-contiguous view with stride 0 along the W
        # dimension.  Downstream ops that call torch.view_as_complex (which
        # requires contiguous input) would crash.  Call .contiguous() to
        # materialise the full tensor in memory.
        return masks.unsqueeze(-1).expand(-1, -1, self.W).contiguous()   # [T, H, W]

    def apply_masks(self, kspace: Tensor) -> Tensor:
        """
        Args:
            kspace: [B, T, C, H, W, 2]
        Returns:
            undersampled kspace: [B, T, C, H, W, 2]
        """
        masks = self.forward()                        # [T, H, W]
        # Reshape for broadcasting: [1, T, 1, H, W, 1]
        m = masks.unsqueeze(0).unsqueeze(2).unsqueeze(-1)
        return kspace * m

    def effective_acceleration(self) -> float:
        """Report the actual acceleration factor of the current mask."""
        with torch.no_grad():
            masks = self.forward()                    # [T, H, W]
            sampled_per_phase = masks[0, :, 0].sum().item()   # unique rows
            return self.H / max(sampled_per_phase, 1)


class RandomEquispacedMask(nn.Module):
    """
    Baseline: random equispaced Cartesian mask (fastMRI-style).
    Samples each phase independently.
    """

    def __init__(self, acceleration: int = 6, acs_lines: int = 24):
        super().__init__()
        self.R   = acceleration
        self.acs = acs_lines

    def forward(self, shape: Tuple[int, int]) -> Tensor:
        """Returns [H, W] binary mask."""
        H, W   = shape
        mask   = torch.zeros(H)
        centre = H // 2
        half   = self.acs // 2
        mask[centre - half: centre + half] = 1.0

        # BUG FIX: total target lines is H/R; subtract already-set ACS lines
        target    = max(1, H // self.R)
        remaining = max(0, target - self.acs)
        avail     = [i for i in range(H) if mask[i] == 0]
        chosen    = torch.randperm(len(avail))[:remaining].tolist()
        for idx in chosen:
            mask[avail[idx]] = 1.0

        return mask.unsqueeze(-1).expand(H, W)   # [H, W]


class GoldenAngleMask(nn.Module):
    """
    Temporal golden-angle radial-to-Cartesian mask.
    Each phase samples a complementary set of radial spokes.
    """

    GOLDEN_ANGLE = 111.246  # degrees

    def __init__(self, n_phases: int = 3, spokes_per_phase: int = 30):
        super().__init__()
        self.T        = n_phases
        self.n_spokes = spokes_per_phase

    def forward(self, shape: Tuple[int, int]) -> Tensor:
        """Returns [T, H, W] binary masks."""
        H, W   = shape
        masks  = []
        cx, cy = W // 2, H // 2
        for t in range(self.T):
            mask = torch.zeros(H, W)
            for s in range(self.n_spokes):
                angle     = (t * self.n_spokes + s) * self.GOLDEN_ANGLE
                angle_rad = angle * torch.pi / 180.0
                cos_a     = torch.cos(torch.tensor(angle_rad)).item()
                sin_a     = torch.sin(torch.tensor(angle_rad)).item()
                for r in range(-max(H, W) // 2, max(H, W) // 2):
                    x = int(cx + r * cos_a)
                    y = int(cy + r * sin_a)
                    if 0 <= x < W and 0 <= y < H:
                        mask[y, x] = 1.0
            masks.append(mask)
        return torch.stack(masks, dim=0)   # [T, H, W]
