"""
KineticsRecon model: temporal E2E-VarNet backbone + pCR prediction head.

Architecture:
  1. SensitivityModel    — estimates coil sensitivity maps from ACS lines
  2. VarNetBlock (x8)    — unrolled variational network (reconstruction)
  3. CrossPhaseAttention — fuses reconstructed features across DCE phases
  4. PKMapHead           — estimates Ktrans, kep, ve per voxel
  5. PCRHead             — predicts pCR from tumour-ROI PK statistics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel, padding=kernel // 2, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Sensitivity model
# ---------------------------------------------------------------------------

class SensitivityModel(nn.Module):
    """
    Estimates full-resolution coil sensitivity maps from the ACS region.

    Workflow:
      1. IFFT the ACS k-space patch → low-resolution coil images [B, C, H_acs, W]
      2. Zero-pad back to full height H → [B, C, H, W]
      3. Refine with a shallow CNN
      4. Normalise across coil dimension

    BUG FIX vs previous version: the old code processed the ACS patch through
    a CNN and claimed the output had full height H — it did not.  We now
    explicitly zero-pad to full resolution before the refinement network.
    """

    def __init__(self, n_coils: int = 8, chans: int = 16, full_height: int = 320):
        super().__init__()
        self.n_coils     = n_coils
        self.full_height = full_height

        self.refine = nn.Sequential(
            ConvBlock(n_coils * 2, chans),
            ConvBlock(chans, chans),
            nn.Conv2d(chans, n_coils * 2, 1),
        )

    def forward(self, acs_kspace: Tensor, full_height: int = None) -> Tensor:
        """
        Args:
            acs_kspace:  [B, C, H_acs, W, 2]  ACS k-space (real/imag)
            full_height: target height; if None uses self.full_height
        Returns:
            maps: [B, C, H, W, 2]  normalised sensitivity maps
        """
        B, C, H_acs, W, _ = acs_kspace.shape
        H = full_height or self.full_height

        # Zero-pad k-space to full height FIRST, then IFFT.
        # Padding in k-space (frequency domain) correctly produces a full-FOV
        # low-pass image via zero-fill reconstruction.  The previous approach
        # of IFFT-then-pad in image space was wrong: it introduced Gibbs
        # ringing at the pad boundary and placed incorrect values in the
        # sensitivity map outside the ACS region.
        #
        # Also cast to float32 before view_as_complex — AMP may have cast the
        # input to float16, which view_as_complex does not support.
        acs_kspace_f = acs_kspace.float()
        acs_c   = torch.view_as_complex(acs_kspace_f.contiguous())     # [B, C, H_acs, W]
        pad_top = (H - H_acs) // 2
        pad_bot = H - H_acs - pad_top
        acs_c_padded = F.pad(acs_c, (0, 0, pad_top, pad_bot))         # [B, C, H, W]
        acs_img = torch.fft.ifftn(acs_c_padded, dim=(-2, -1), norm="ortho")  # [B, C, H, W]

        # Real/imag → channel dim for CNN
        x = torch.cat([acs_img.real, acs_img.imag], dim=1)            # [B, 2C, H, W]
        x = self.refine(x)                                           # [B, 2C, H, W]

        # Split back to real/imag
        maps = torch.stack([x[:, :C], x[:, C:]], dim=-1)            # [B, C, H, W, 2]

        # Normalise across coil dimension (RSS normalisation)
        rss  = (maps ** 2).sum(dim=-1, keepdim=True).sum(dim=1, keepdim=True).sqrt()
        return maps / (rss + 1e-8)


# ---------------------------------------------------------------------------
# Data consistency
# ---------------------------------------------------------------------------

class DataConsistency(nn.Module):
    """Replace predicted k-space with observed k-space where mask == 1."""

    def forward(self, k_pred: Tensor, k_obs: Tensor, mask: Tensor) -> Tensor:
        # mask: [B, 1, H, W] or broadcastable; k_*: [B, C, H, W, 2]
        return mask * k_obs + (1.0 - mask) * k_pred


# ---------------------------------------------------------------------------
# VarNet block
# ---------------------------------------------------------------------------

class VarNetBlock(nn.Module):
    """
    One unrolled VarNet iteration:
        k_{n+1} = DC( k_n - eta * E( R( E^H(k_n) ) ), k_obs, mask )

    where E = sensitivity-weighted FFT, E^H = IFFT + coil combination,
    R = CNN regulariser.
    """

    def __init__(self, n_coils: int = 8, chans: int = 18):
        super().__init__()
        self.regulariser = nn.Sequential(
            ConvBlock(2, chans),
            ConvBlock(chans, chans),
            nn.Conv2d(chans, 2, 1),
        )
        self.dc  = DataConsistency()
        self.eta = nn.Parameter(torch.ones(1) * 0.5)

    # ---- MRI operators ----

    @staticmethod
    def _sens_expand(img: Tensor, maps: Tensor) -> Tensor:
        """
        Image [B, H, W, 2] × sensitivity maps [B, C, H, W, 2]
        → multi-coil k-space [B, C, H, W, 2]

        NOTE: torch.view_as_complex requires float32.  When AMP is active the
        autocast context may cast tensors to float16 before reaching here, which
        causes a hard runtime error.  We explicitly cast to float32 before any
        complex operation and cast back afterwards so AMP still benefits the
        rest of the network.
        """
        img   = img.float()
        maps  = maps.float()
        ir, ii = img[..., 0].unsqueeze(1),  img[..., 1].unsqueeze(1)
        mr, mi = maps[..., 0],               maps[..., 1]
        coil_r = ir * mr - ii * mi
        coil_i = ir * mi + ii * mr
        coil_c = torch.view_as_complex(torch.stack([coil_r, coil_i], dim=-1).contiguous())
        kspace = torch.fft.fftn(coil_c, dim=(-2, -1), norm="ortho")
        return torch.view_as_real(kspace)                               # [B,C,H,W,2]

    @staticmethod
    def _sens_reduce(kspace: Tensor, maps: Tensor) -> Tensor:
        """
        Multi-coil k-space [B, C, H, W, 2] + maps [B, C, H, W, 2]
        → combined image [B, H, W, 2]  (conj(maps) · IFFT(kspace))

        See _sens_expand for the float32 cast rationale.
        """
        kspace = kspace.float()
        maps   = maps.float()
        kc  = torch.view_as_complex(kspace.contiguous())               # [B,C,H,W]
        img = torch.fft.ifftn(kc, dim=(-2, -1), norm="ortho")         # [B,C,H,W] complex
        ir, ii = img.real, img.imag
        mr, mi = maps[..., 0], maps[..., 1]
        out_r = (ir * mr + ii * mi).sum(1)
        out_i = (ii * mr - ir * mi).sum(1)
        return torch.stack([out_r, out_i], dim=-1)                     # [B,H,W,2]

    def forward(
        self,
        current_kspace: Tensor,   # [B, C, H, W, 2]
        ref_kspace:     Tensor,   # [B, C, H, W, 2]  observed (undersampled)
        mask:           Tensor,   # [B, 1, H, W, 1]  broadcastable binary
        sens_maps:      Tensor,   # [B, C, H, W, 2]
    ) -> Tensor:
        img     = self._sens_reduce(current_kspace, sens_maps)         # [B,H,W,2]
        img_in  = img.permute(0, 3, 1, 2)                              # [B,2,H,W]
        reg     = self.regulariser(img_in).permute(0, 2, 3, 1)        # [B,H,W,2]
        img_upd = img - self.eta * reg
        k_upd   = self._sens_expand(img_upd, sens_maps)                # [B,C,H,W,2]
        return self.dc(k_upd, ref_kspace, mask)


# ---------------------------------------------------------------------------
# Cross-phase attention
# ---------------------------------------------------------------------------

class CrossPhaseAttention(nn.Module):
    """
    Fuses feature maps across DCE phases via multi-head self-attention.

    BUG FIX vs previous version: the old implementation computed the fused
    tensor but then returned the original `phase_feats` unchanged — making
    the module a complete no-op.  Fixed: the fused output is now used to
    produce a residual correction that is added back to each phase feature.
    """

    def __init__(self, in_channels: int, feat_dim: int = 128, n_heads: int = 4):
        super().__init__()
        self.proj_in  = nn.Conv2d(in_channels, feat_dim, 1)
        self.attn     = nn.MultiheadAttention(feat_dim, n_heads, batch_first=True)
        self.proj_out = nn.Conv2d(feat_dim, in_channels, 1)
        self.norm     = nn.LayerNorm(feat_dim)

    def forward(self, phase_feats: List[Tensor]) -> List[Tensor]:
        """
        Args:
            phase_feats: list of T tensors, each [B, C, H, W]
        Returns:
            fused:       list of T tensors, each [B, C, H, W]  (attention-corrected)
        """
        B, C, H, W = phase_feats[0].shape
        T = len(phase_feats)

        # Project each phase to feat_dim and global-pool to get a token per phase
        tokens = []
        projs  = []
        for feat in phase_feats:
            p = self.proj_in(feat)                        # [B, feat_dim, H, W]
            projs.append(p)
            tokens.append(p.mean(dim=(-2, -1)))           # [B, feat_dim] (global avg pool)

        seq = torch.stack(tokens, dim=1)                  # [B, T, feat_dim]
        fused_seq, _ = self.attn(seq, seq, seq)           # [B, T, feat_dim]
        fused_seq = self.norm(seq + fused_seq)            # [B, T, feat_dim]

        # Broadcast fused token back spatially and add as residual
        out = []
        for t in range(T):
            delta_token = fused_seq[:, t, :]              # [B, feat_dim]
            delta_map   = delta_token[:, :, None, None].expand_as(projs[t])  # [B, feat_dim, H, W]
            correction  = self.proj_out(delta_map)        # [B, C, H, W]
            out.append(phase_feats[t] + correction)       # residual add
        return out


# ---------------------------------------------------------------------------
# PK map head
# ---------------------------------------------------------------------------

class PKMapHead(nn.Module):
    """
    Lightweight CNN: DCE magnitude sequence [B, T, H, W] → PK maps [B, 3, H, W].
    Outputs Ktrans, kep, ve — all positive via softplus.
    """

    def __init__(self, n_phases: int = 3, chans: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(n_phases, chans, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(chans, chans, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(chans, chans * 2, 3, padding=1, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(chans * 2, chans, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(chans, 3, 1),
        )

    def forward(self, dce_mag: Tensor) -> Tensor:
        x = self.encoder(dce_mag)
        x = self.decoder(x)
        return F.softplus(x)   # [B, 3, H, W], all positive


# ---------------------------------------------------------------------------
# pCR prediction head
# ---------------------------------------------------------------------------

class PCRHead(nn.Module):
    """
    Predicts pCR probability from tumour-ROI PK statistics + image features.

    BUG FIX vs previous version: percentiles are now computed only over
    ROI voxels (mask==1), not the entire image including background zeros.
    """

    def __init__(
        self,
        n_pk_stats: int = 12,      # 4 stats × 3 PK params
        n_clinical: int = 0,
        hidden: int = 128,
    ):
        super().__init__()
        self.img_feat = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(16 * 64, 64),
            nn.ReLU(),
        )
        in_dim   = n_pk_stats + n_clinical + 64
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1),
        )

    def _pk_roi_stats(self, pk_maps: Tensor, mask: Tensor) -> Tensor:
        """
        Compute mean, std, p25, p75 of each PK parameter within the tumour ROI.

        BUG FIX: extract only the masked voxels for each sample, then compute
        statistics on that subset.  Previous version included background zeros
        in the percentile computation.

        pk_maps: [B, 3, H, W]
        mask:    [B, H, W]  binary
        Returns: [B, 12]
        """
        B = pk_maps.shape[0]
        stats_list = []
        for b in range(B):
            m        = mask[b].bool().flatten()    # [H*W]
            row_stats = []
            for p in range(3):
                vals = pk_maps[b, p].flatten()     # [H*W]
                roi  = vals[m]                     # only masked voxels
                if roi.numel() == 0:
                    roi = vals                     # fallback: whole image
                mean = roi.mean()
                std  = roi.std(unbiased=False)
                p25  = roi.kthvalue(max(1, int(roi.numel() * 0.25))).values
                p75  = roi.kthvalue(max(1, int(roi.numel() * 0.75))).values
                row_stats += [mean, std, p25, p75]
            stats_list.append(torch.stack(row_stats))   # [12]
        return torch.stack(stats_list, dim=0)            # [B, 12]

    def forward(
        self,
        pk_maps:   Tensor,
        mask:      Tensor,
        img_phase: Tensor,
        clinical:  Optional[Tensor] = None,
    ) -> Tensor:
        pk_stats  = self._pk_roi_stats(pk_maps, mask)    # [B, 12]
        img_feats = self.img_feat(img_phase)              # [B, 64]

        parts = [pk_stats, img_feats]
        if clinical is not None:
            parts.append(clinical)
        feat = torch.cat(parts, dim=-1)
        return self.mlp(feat)                            # [B, 1]


# ---------------------------------------------------------------------------
# Full KineticsRecon model
# ---------------------------------------------------------------------------

class KineticsRecon(nn.Module):
    """
    End-to-end: undersampled DCE k-space → reconstructed images → pCR logit.

    Stage 1 (pre-training on fastMRI):
        Only VarNet backbone trained with standard image reconstruction loss.
        n_phases=1 for single-slice fastMRI data.

    Stage 2 (fine-tuning on MAMAMIA):
        Full network trained with KineticsAwareLoss + pCR BCE.
        n_phases=3 for pre/early-post/late-post DCE sequence.
    """

    def __init__(
        self,
        n_coils:         int = 8,
        n_phases:        int = 3,
        n_varnet_blocks: int = 8,
        varnet_chans:    int = 18,
        pk_chans:        int = 32,
        pcr_hidden:      int = 128,
        n_clinical:      int = 0,
        full_height:     int = 320,
    ):
        super().__init__()
        self.n_phases = n_phases
        self.n_coils  = n_coils

        self.sens_model    = SensitivityModel(n_coils, chans=16, full_height=full_height)
        self.varnet_blocks = nn.ModuleList([
            VarNetBlock(n_coils, varnet_chans) for _ in range(n_varnet_blocks)
        ])
        self.cross_phase   = CrossPhaseAttention(in_channels=1, feat_dim=64)
        self.pk_head       = PKMapHead(n_phases, pk_chans)
        self.pcr_head      = PCRHead(n_pk_stats=12, n_clinical=n_clinical, hidden=pcr_hidden)

    def _reconstruct_phase(
        self,
        kspace:    Tensor,   # [B, C, H, W, 2]
        mask:      Tensor,   # [B, 1, H, W, 1]
        sens_maps: Tensor,   # [B, C, H, W, 2]
    ) -> Tensor:
        """
        Unrolled VarNet reconstruction → magnitude image [B, H, W].

        BUG FIX: previous version did mag.sum(1) (simple coil sum) instead of
        RSS (root-sum-of-squares): sqrt(sum(|coil_image|^2)).
        """
        current = kspace.clone()
        for block in self.varnet_blocks:
            current = block(current, kspace, mask, sens_maps)

        # IFFT + RSS coil combination — cast to float32 first (AMP safety)
        img_c   = torch.view_as_complex(current.float().contiguous())  # [B, C, H, W]
        img_c   = torch.fft.ifftn(img_c, dim=(-2, -1), norm="ortho")  # [B, C, H, W]
        mag     = img_c.abs()                                           # [B, C, H, W]
        rss     = (mag ** 2).sum(dim=1).sqrt()                         # [B, H, W]
        return rss

    def forward(
        self,
        kspace:      Tensor,                   # [B, T, C, H, W, 2]
        masks:       Tensor,                   # [T, H, W]
        acs_kspace:  Tensor,                   # [B, C, H_acs, W, 2]
        tumour_mask: Optional[Tensor] = None,  # [B, H, W]
        clinical:    Optional[Tensor] = None,  # [B, n_clinical]
    ) -> dict:
        B, T, C, H, W, _ = kspace.shape

        # Sensitivity maps (shared across phases)
        sens = self.sens_model(acs_kspace, full_height=H)   # [B, C, H, W, 2]

        # Reconstruct each phase independently
        recon_phases = []
        for t in range(T):
            k_t    = kspace[:, t]                              # [B, C, H, W, 2]
            mask_t = masks[t]                                  # [H, W]
            # Expand to [B, 1, H, W, 1] for broadcasting in DataConsistency
            mask_bc = mask_t.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, H, W, 1]
            mask_bc = mask_bc.expand(B, 1, H, W, 1)
            phase_img = self._reconstruct_phase(k_t, mask_bc, sens)   # [B, H, W]
            recon_phases.append(phase_img)

        # Cross-phase attention — add channel dim for CNN compatibility
        feat_in  = [p.unsqueeze(1) for p in recon_phases]    # list of [B, 1, H, W]
        feat_out = self.cross_phase(feat_in)                  # list of [B, 1, H, W]
        recon_stack = torch.stack(
            [f.squeeze(1) for f in feat_out], dim=1
        )                                                     # [B, T, H, W]

        # PK maps
        pk_maps = self.pk_head(recon_stack)                   # [B, 3, H, W]

        # pCR prediction
        pcr_logit = None
        if tumour_mask is not None:
            early_phase = recon_stack[:, 1:2]                 # [B, 1, H, W]
            pcr_logit   = self.pcr_head(pk_maps, tumour_mask, early_phase, clinical)

        return {
            "recon_phases": recon_stack,
            "pk_maps":      pk_maps,
            "pcr_logit":    pcr_logit,
        }

    def freeze_backbone(self):
        for p in self.sens_model.parameters():
            p.requires_grad = False
        for p in self.varnet_blocks.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
