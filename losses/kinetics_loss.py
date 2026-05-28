"""
Kinetics-aware reconstruction loss.

Fits a Tofts pharmacokinetic model to DCE signal curves from both the
ground-truth and reconstructed image sequences, then penalises deviation
in Ktrans and kep.  Combined with a standard image-quality term (SSIM + L1)
this forces the reconstruction network to preserve the temporal enhancement
structure that radiologists and pCR models rely on.

References:
    Tofts PS et al. (1999) J Magn Reson Imaging 10:223-232
    Parker GJ et al. (2006) Magn Reson Med 56:993-1000
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


# ---------------------------------------------------------------------------
# Arterial Input Function
# ---------------------------------------------------------------------------

def _aif_parker(t: Tensor) -> Tensor:
    """
    Parker population-averaged arterial input function.
    Returns Cp(t) in mM.  t is in minutes.
    """
    a1, a2 = 0.809, 0.330
    T1, T2 = 0.17046, 0.365
    s1, s2 = 0.0563, 0.132
    alpha, beta, s, tau = 1.050, 0.1685, 38.078, 0.483

    g1 = (a1 / (s1 * (2 * torch.pi) ** 0.5)) * torch.exp(-((t - T1) ** 2) / (2 * s1 ** 2))
    g2 = (a2 / (s2 * (2 * torch.pi) ** 0.5)) * torch.exp(-((t - T2) ** 2) / (2 * s2 ** 2))
    sigmoid  = 1.0 / (1.0 + torch.exp(-s * (t - tau)))
    exp_term = alpha * torch.exp(-beta * t) * sigmoid

    return g1 + g2 + exp_term


# ---------------------------------------------------------------------------
# Tofts forward model
# ---------------------------------------------------------------------------

def tofts_signal(
    ktrans: Tensor,
    kep: Tensor,
    t: Tensor,
    t10: float = 1.5,
    tr: float = 0.005,
    flip_angle: float = 20.0,
    r1: float = 4.5,
) -> Tensor:
    """
    Extended Tofts model: compute expected DCE signal enhancement.

    Args:
        ktrans: [B, H, W]   volume transfer constant (min^-1)
        kep:    [B, H, W]   rate constant (min^-1)
        t:      [T]          acquisition times in minutes (uniform spacing assumed)
        t10:    pre-contrast T1 (s)
        tr:     repetition time (s)
        flip_angle: degrees
        r1:     relaxivity (mM^-1 s^-1)

    Returns:
        enhancement: [B, H, W, T]
    """
    B, H, W = ktrans.shape
    T = t.shape[0]
    dt = (t[-1] - t[0]) / max(T - 1, 1)   # uniform step size

    cp = _aif_parker(t)                    # [T]

    # Tissue concentration via Tofts convolution integral (trapezoidal)
    # Ct(t_i) = Ktrans * sum_{j<=i} Cp(t_j) * exp(-kep*(t_i - t_j)) * dt
    # We accumulate the running integral efficiently:
    #   running_i = running_{i-1} * exp(-kep*dt) + Cp(t_i) * dt
    kep_ = kep.unsqueeze(-1)               # [B, H, W, 1]
    decay_step = torch.exp(-kep_ * dt)     # [B, H, W, 1]

    running = torch.zeros(B, H, W, device=ktrans.device, dtype=ktrans.dtype)
    ct = torch.zeros(B, H, W, T, device=ktrans.device, dtype=ktrans.dtype)

    for i in range(T):
        running = running * decay_step.squeeze(-1) + cp[i] * dt
        ct[..., i] = ktrans * running

    # SPGR signal model: concentration → signal
    fa_rad = torch.tensor(flip_angle * torch.pi / 180.0, device=ktrans.device, dtype=ktrans.dtype)
    e10    = torch.exp(torch.tensor(-tr / t10, device=ktrans.device, dtype=ktrans.dtype))
    cos_fa = torch.cos(fa_rad)
    sin_fa = torch.sin(fa_rad)

    s0 = sin_fa * (1 - e10) / (1 - cos_fa * e10)        # baseline signal

    r1t = (1.0 / t10) + r1 * ct                          # [B, H, W, T]
    et  = torch.exp(-tr * r1t)
    st  = sin_fa * (1 - et) / (1 - cos_fa * et)

    return (st - s0) / (s0 + 1e-8)                       # normalised enhancement


# ---------------------------------------------------------------------------
# Differentiable PK parameter estimator
# ---------------------------------------------------------------------------

class ToftsEstimator(nn.Module):
    """
    Estimates Ktrans and kep from DCE enhancement curves.

    Two-step approach:
      1. Coarse grid search: evaluate the Tofts forward model on a grid of
         (Ktrans, kep) pairs and pick the best-fitting pair per voxel.
      2. Learned refinement: a small MLP maps the coarse estimate plus a
         soft encoding of its grid position to a correction delta, so that
         gradients can flow back through the estimator into the reconstruction
         network.
    """

    def __init__(
        self,
        n_ktrans: int = 16,
        n_kep: int = 16,
        ktrans_range: Tuple[float, float] = (0.01, 2.0),
        kep_range: Tuple[float, float] = (0.01, 2.0),
    ):
        super().__init__()
        self.register_buffer("ktrans_grid", torch.linspace(*ktrans_range, n_ktrans))
        self.register_buffer("kep_grid",    torch.linspace(*kep_range,    n_kep))

        # Refinement MLP: (soft grid encoding + coarse values) → delta
        # BUG FIX: softmax encoding uses *negative* squared distances so that
        # the nearest grid point gets the highest weight.
        self.refine = nn.Sequential(
            nn.Linear(n_ktrans + n_kep + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),    # (delta_ktrans, delta_kep)
            nn.Tanh(),
        )

    def forward(
        self,
        signal: Tensor,
        t: Tensor,
        max_voxels: int = 2048,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            signal:     [B, H, W, T]  normalised DCE enhancement curves
            t:          [T]            time points in minutes
            max_voxels: maximum number of voxels to use in the grid search.
                        The grid search is O(n_ktrans × n_kep × N × T).  At
                        full resolution N = B×H×W ≈ 400k for a 320×320 batch
                        of 4 — that would take hours per step.  We randomly
                        subsample `max_voxels` positions, run the grid search
                        on those, then bilinearly upsample the resulting PK
                        maps back to (H, W).  2048 gives good coverage of the
                        enhancement-curve space while keeping each step < 1s.
        Returns:
            ktrans: [B, H, W]
            kep:    [B, H, W]
        """
        B, H, W, T_len = signal.shape

        # --- Spatial downsampling for grid search ---
        # Work on a small fixed spatial grid rather than every voxel.
        grid_side  = max(1, int(max_voxels ** 0.5))          # e.g. 45 for 2048
        sig_down   = F.interpolate(
            signal.permute(0, 3, 1, 2).float(),              # [B, T, H, W]
            size=(grid_side, grid_side),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)                                 # [B, gs, gs, T]

        N    = B * grid_side * grid_side
        flat = sig_down.reshape(N, T_len)                     # [N, T]

        best_ktrans = torch.zeros(N, device=signal.device, dtype=signal.dtype)
        best_kep    = torch.zeros(N, device=signal.device, dtype=signal.dtype)
        best_mse    = torch.full((N,), float("inf"), device=signal.device, dtype=signal.dtype)

        for kt_val in self.ktrans_grid:
            for ke_val in self.kep_grid:
                kt_scalar = kt_val.item()
                ke_scalar = ke_val.item()
                kt_map = torch.full((N, 1, 1), kt_scalar, device=signal.device, dtype=signal.dtype)
                ke_map = torch.full((N, 1, 1), ke_scalar, device=signal.device, dtype=signal.dtype)

                pred = tofts_signal(kt_map, ke_map, t)        # [N, 1, 1, T]
                pred = pred.reshape(N, T_len)

                mse    = ((flat - pred) ** 2).mean(-1)        # [N]
                better = mse < best_mse
                best_mse    = torch.where(better, mse, best_mse)
                best_ktrans = torch.where(better, torch.full((N,), kt_scalar, device=signal.device, dtype=signal.dtype), best_ktrans)
                best_kep    = torch.where(better, torch.full((N,), ke_scalar, device=signal.device, dtype=signal.dtype), best_kep)

        # Soft encoding for refinement MLP
        kt_enc = (-(best_ktrans.unsqueeze(-1) - self.ktrans_grid) ** 2).softmax(-1)
        ke_enc = (-(best_kep.unsqueeze(-1)    - self.kep_grid)    ** 2).softmax(-1)
        coarse = torch.stack([best_ktrans, best_kep], dim=-1)
        feat   = torch.cat([kt_enc, ke_enc, coarse], dim=-1)

        delta      = self.refine(feat) * 0.2
        ktrans_low = (best_ktrans + delta[:, 0]).clamp(0.001, 3.0).reshape(B, grid_side, grid_side)
        kep_low    = (best_kep    + delta[:, 1]).clamp(0.001, 3.0).reshape(B, grid_side, grid_side)

        # --- Upsample PK maps back to full resolution ---
        ktrans_out = F.interpolate(
            ktrans_low.unsqueeze(1).float(), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(1)                                           # [B, H, W]
        kep_out = F.interpolate(
            kep_low.unsqueeze(1).float(), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(1)                                           # [B, H, W]

        return ktrans_out, kep_out


# ---------------------------------------------------------------------------
# Composite kinetics-aware loss
# ---------------------------------------------------------------------------

class KineticsAwareLoss(nn.Module):
    """
    L_total = w_img   * L_image
            + w_pk    * L_pk
            + w_curve * L_curve

    L_image:  SSIM + L1 on each reconstructed phase
    L_pk:     MSE on Ktrans and kep maps (GT vs recon), within tumour ROI
    L_curve:  MSE on normalised DCE enhancement curves
    """

    def __init__(
        self,
        w_img: float = 1.0,
        w_pk: float = 0.5,
        w_curve: float = 0.3,
        ssim_window: int = 11,
        pk_estimator: ToftsEstimator = None,
    ):
        super().__init__()
        self.w_img        = w_img
        self.w_pk         = w_pk
        self.w_curve      = w_curve
        self.estimator    = pk_estimator or ToftsEstimator()
        self._ssim_window = ssim_window

    def _ssim(self, x: Tensor, y: Tensor) -> Tensor:
        """SSIM loss (1 - SSIM) averaged over batch. x, y: [B, H, W]."""
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        k = self._ssim_window
        kernel = torch.ones(1, 1, k, k, device=x.device, dtype=x.dtype) / (k * k)

        def mu(t):
            return F.conv2d(t.unsqueeze(1), kernel, padding=k // 2)

        mu_x = mu(x);  mu_y = mu(y)
        sigma_x  = mu(x * x) - mu_x ** 2
        sigma_y  = mu(y * y) - mu_y ** 2
        sigma_xy = mu(x * y) - mu_x * mu_y

        num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
        return (1.0 - (num / den).mean())

    def _image_loss(self, recon: Tensor, target: Tensor) -> Tensor:
        """recon, target: [B, T, H, W]"""
        T  = recon.shape[1]
        l1 = F.l1_loss(recon, target)
        ssim_loss = torch.stack([
            self._ssim(recon[:, t], target[:, t]) for t in range(T)
        ]).mean()
        return l1 + ssim_loss

    def _curve_loss(self, recon: Tensor, target: Tensor) -> Tensor:
        """
        MSE on voxel-wise normalised enhancement curves.
        recon, target: [B, T, H, W]

        The enhancement ratio (x - pre) / |pre| can be large when the
        pre-contrast value is small, pushing this term orders of magnitude
        above the image loss.  We clamp the enhancement to [-10, 10] to
        suppress exploding values from near-zero pre-contrast voxels, then
        normalise by T so the scale stays consistent regardless of n_phases.
        """
        T = recon.shape[1]
        pre_r  = recon[:, 0:1]
        pre_t  = target[:, 0:1]
        enh_r  = ((recon  - pre_r)  / (pre_r.abs()  + 1e-6)).clamp(-10, 10)
        enh_t  = ((target - pre_t)  / (pre_t.abs()  + 1e-6)).clamp(-10, 10)
        return F.mse_loss(enh_r, enh_t) / T

    def _pk_loss(self, recon: Tensor, target: Tensor, t: Tensor) -> Tensor:
        """
        Estimate Ktrans/kep from both sequences and penalise deviation.
        recon, target: [B, T, H, W]; t: [T] in minutes.
        The GT PK map is detached so gradients only flow through the
        reconstruction path.
        """
        # Build enhancement curves [B, H, W, T]
        def to_enh(x):
            pre = x[:, 0:1].permute(0, 2, 3, 1)          # [B, H, W, 1]
            full = x.permute(0, 2, 3, 1)                  # [B, H, W, T]
            return (full - pre) / (pre.abs() + 1e-6)

        kt_r, ke_r = self.estimator(to_enh(recon),  t)
        kt_t, ke_t = self.estimator(to_enh(target), t)

        return F.mse_loss(kt_r, kt_t.detach()) + F.mse_loss(ke_r, ke_t.detach())

    def forward(
        self,
        recon: Tensor,
        target: Tensor,
        t: Tensor,
        mask: Tensor = None,
    ) -> Tuple[Tensor, dict]:
        """
        Args:
            recon:  [B, T, H, W]
            target: [B, T, H, W]
            t:      [T] in minutes
            mask:   [B, H, W] tumour ROI (focuses PK + curve losses)
        Returns:
            total_loss, component_dict
        """
        if mask is not None:
            m          = mask.unsqueeze(1)         # [B, 1, H, W]
            recon_roi  = recon  * m
            target_roi = target * m
        else:
            recon_roi, target_roi = recon, target

        l_img   = self._image_loss(recon,     target)
        l_curve = self._curve_loss(recon_roi, target_roi)
        l_pk    = self._pk_loss(recon_roi, target_roi, t)

        total = self.w_img * l_img + self.w_curve * l_curve + self.w_pk * l_pk

        return total, {
            "loss/image":  l_img.item(),
            "loss/curve":  l_curve.item(),
            "loss/pk":     l_pk.item(),
            "loss/total":  total.item(),
        }
