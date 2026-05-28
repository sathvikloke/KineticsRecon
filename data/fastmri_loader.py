"""
fastMRI Breast data loader for pre-training the VarNet backbone.

fastMRI Breast HDF5 files contain:
  kspace:  (2, 288, 640, 16, 83)  raw radial k-space
  temptv:  (192, 4, 320, 320)     temporal TV reconstruction
                                  — (z_slices, T_phases, H, W)

We use `temptv` as the ground-truth target and simulate 2D Cartesian
k-space from each phase slice. This avoids the need for NUFFT during
pre-training while still giving the backbone exposure to temporal DCE
structure.

Dataset: https://fastmri.org/dataset/
"""

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, List
import random


class FastMRISliceDataset(Dataset):
    """
    Yields individual (z-slice, temporal-phase) pairs from fastMRI Breast HDF5s.

    Each item:
        kspace:      [1, H, W, 2]  simulated single-coil undersampled k-space
        kspace_full: [1, H, W, 2]  fully-sampled k-space
        target:      [H, W]        ground-truth magnitude image
        mask:        [H, W]        undersampling mask
        acs_kspace:  [1, H_acs, W, 2]  ACS region
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        acceleration: int = 6,
        acs_lines: int = 24,
        max_slices: Optional[int] = None,
        val_frac: float = 0.15,
        seed: int = 42,
    ):
        self.root         = Path(root)
        self.split        = split
        self.acceleration = acceleration
        self.acs_lines    = acs_lines

        # Files sit directly in root (flat structure, no multicoil_train/ subdir)
        all_files = sorted(self.root.glob("*.h5"))
        if not all_files:
            raise FileNotFoundError(f"No .h5 files found in {self.root}")

        # Train / val split by file
        rng = random.Random(seed)
        files_shuf = list(all_files)
        rng.shuffle(files_shuf)
        n_val   = max(1, int(len(files_shuf) * val_frac))
        if split == "val":
            files = files_shuf[:n_val]
        else:
            files = files_shuf[n_val:]

        # Build index: (file_path, z_idx, phase_idx)
        self.slices: List[Tuple[Path, int, int]] = []
        for fpath in files:
            with h5py.File(fpath, "r") as f:
                if "temptv" not in f:
                    continue
                n_z, n_phases, _, _ = f["temptv"].shape   # (z, T, H, W)
                for z in range(n_z):
                    for t in range(n_phases):
                        self.slices.append((fpath, z, t))

        if max_slices is not None:
            rng2 = random.Random(seed + 1)
            self.slices = rng2.sample(self.slices, min(max_slices, len(self.slices)))

    def _make_mask(self, H: int) -> Tensor:
        """1-D Cartesian mask. Total lines = H // acceleration (ACS included)."""
        mask = torch.zeros(H)
        centre = H // 2
        half   = self.acs_lines // 2
        mask[centre - half: centre + half] = 1.0
        target_total = max(1, H // self.acceleration)
        extra  = max(0, target_total - self.acs_lines)
        avail  = [i for i in range(H) if mask[i] == 0]
        chosen = random.sample(avail, min(extra, len(avail)))
        for l in chosen:
            mask[l] = 1.0
        return mask

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> dict:
        fpath, z_idx, t_idx = self.slices[idx]

        with h5py.File(fpath, "r") as f:
            # temptv: (z, T, H, W) float64 — pre-reconstructed temporal TV image
            img_np = f["temptv"][z_idx, t_idx].astype(np.float32)   # [H, W]

        target = torch.from_numpy(img_np)   # [H, W]
        H, W   = target.shape

        # Simulate single-coil Cartesian k-space
        kspace_c = torch.fft.fftn(target.cfloat(), dim=(-2, -1), norm="ortho")
        kspace   = torch.view_as_real(kspace_c)     # [H, W, 2]
        norm     = kspace.abs().max() + 1e-8
        kspace   = kspace / norm

        # Undersampling mask
        mask_1d = self._make_mask(H)
        mask_2d = mask_1d.unsqueeze(1).expand(H, W).contiguous()   # [H, W]

        us_kspace = kspace * mask_2d.unsqueeze(-1)   # [H, W, 2]

        # Add coil dimension → [1, H, W, 2]
        kspace    = kspace.unsqueeze(0)
        us_kspace = us_kspace.unsqueeze(0)

        # ACS region [1, H_acs, W, 2]
        centre     = H // 2
        half       = self.acs_lines // 2
        acs_kspace = kspace[:, centre - half: centre + half, :, :]

        return {
            "kspace":      us_kspace,    # [1, H, W, 2]
            "kspace_full": kspace,       # [1, H, W, 2]
            "target":      target,       # [H, W]
            "mask":        mask_2d,      # [H, W]
            "acs_kspace":  acs_kspace,   # [1, H_acs, W, 2]
            "norm":        norm,
            "filename":    str(fpath),
            "z_idx":       z_idx,
            "t_idx":       t_idx,
        }


def fastmri_collate(batch: list) -> dict:
    """
    Collate fastMRI Breast samples. All temptv slices are 320×320 so padding
    is rarely needed, but included for safety.
    """
    max_H = max(b["kspace"].shape[-3] for b in batch)   # [1, H, W, 2]
    max_W = max(b["kspace"].shape[-2] for b in batch)

    def pad4(t: torch.Tensor) -> torch.Tensor:
        """Pad [1, H, W, 2] tensor to (max_H, max_W)."""
        cH, cW = t.shape[-3], t.shape[-2]
        return F.pad(t, (0, 0, 0, max_W - cW, 0, max_H - cH))

    def pad2(t: torch.Tensor) -> torch.Tensor:
        """Pad [H, W] tensor."""
        cH, cW = t.shape
        return F.pad(t, (0, max_W - cW, 0, max_H - cH))

    out = {
        "kspace":      torch.stack([pad4(b["kspace"])      for b in batch]),
        "kspace_full": torch.stack([pad4(b["kspace_full"]) for b in batch]),
        "target":      torch.stack([pad2(b["target"])      for b in batch]),
        "acs_kspace":  torch.stack([b["acs_kspace"]        for b in batch]),
        "norm":        torch.tensor([b["norm"]             for b in batch]),
    }
    # Shared mask — all batch items use the same pattern for correct DC
    padded_masks = torch.stack([pad2(b["mask"]) for b in batch])
    out["mask"] = padded_masks[0]   # [H, W]
    return out


def build_fastmri_loader(
    root: str,
    split: str = "train",
    acceleration: int = 6,
    batch_size: int = 4,
    num_workers: int = 4,
    max_slices: Optional[int] = None,
) -> DataLoader:
    ds = FastMRISliceDataset(
        root=root,
        split=split,
        acceleration=acceleration,
        max_slices=max_slices,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=fastmri_collate,
        pin_memory=True,
        drop_last=(split == "train"),
    )
