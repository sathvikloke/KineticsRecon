"""
MAMAMIA dataset loader.

MAMAMIA (Garrucho et al., Sci Data 2025) provides:
  - Pre-treatment T1-weighted DCE-MRI (pre, early-post, late-post phases)
  - Expert tumour segmentations (primary tumour + NME)
  - Clinical metadata including pCR status

Dataset: https://github.com/LidiaGarrucho/MAMA-MIA  /  TCIA

Expected layout:
    mamamia_root/
      images/
        case_001/
          0000.nii.gz   (pre-contrast)
          0001.nii.gz   (early post-contrast)
          0002.nii.gz   (late post-contrast)
      segmentations/
        case_001_seg.nii.gz
      clinical.csv      (columns: case_id, pCR, age, subtype, centre, ...)
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    print("Warning: nibabel not installed.  Run: pip install nibabel")


class MAMAMIADataset(Dataset):

    PHASES = ["0000", "0001", "0002"]   # pre, early post, late post

    def __init__(
        self,
        root: str,
        clinical_csv: str,
        split: str = "train",
        acceleration: int = 6,
        acs_lines: int = 24,
        n_phases: int = 3,
        slice_mode: str = "tumour_centre",   # "tumour_centre" | "all"
        augment: bool = False,
        patch_size: Tuple[int, int] = (320, 320),
        clinical_feats: Optional[List[str]] = None,
        seed: int = 42,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
    ):
        if not HAS_NIBABEL:
            raise RuntimeError("nibabel required: pip install nibabel")

        self.root           = Path(root)
        self.acceleration   = acceleration
        self.acs_lines      = acs_lines
        self.n_phases       = n_phases
        self.slice_mode     = slice_mode
        self.augment        = augment and (split == "train")
        self.patch_size     = patch_size
        self.clinical_feats = clinical_feats or []
        self.split          = split

        self.clinical = self._load_clinical(clinical_csv)

        # Discover cases
        img_dir   = self.root / "images"
        all_cases = sorted(d.name for d in img_dir.iterdir() if d.is_dir())
        all_cases = [c for c in all_cases if c in self.clinical]

        rng = random.Random(seed)
        rng.shuffle(all_cases)
        n       = len(all_cases)
        n_train = int(n * train_frac)
        n_val   = int(n * val_frac)

        if split == "train":
            cases = all_cases[:n_train]
        elif split == "val":
            cases = all_cases[n_train: n_train + n_val]
        else:
            cases = all_cases[n_train + n_val:]

        self.cases = cases
        print(f"[MAMAMIA] {split}: {len(cases)} cases")
        self.items = self._build_index()

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _load_clinical(self, csv_path: str) -> Dict[str, dict]:
        """
        Load clinical metadata. Accepts .xlsx (clinical_and_imaging_info.xlsx)
        or .csv files. Requires openpyxl for Excel: pip install openpyxl
        """
        import pandas as pd
        p = Path(csv_path)
        if p.suffix in (".xlsx", ".xls"):
            df = pd.read_excel(csv_path, dtype=str)
        else:
            df = pd.read_csv(csv_path, dtype=str)

        data = {}
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            cid = (row_dict.get("patient_id")
                   or row_dict.get("case_id")
                   or row_dict.get("PatientID"))
            if cid:
                data[str(cid).strip()] = row_dict
        return data

    def _tumour_centre_slice(self, seg: np.ndarray, axis: int) -> int:
        """Return the index along `axis` of the tumour centroid."""
        coords = np.argwhere(seg > 0)
        if len(coords) == 0:
            return seg.shape[axis] // 2
        return int(coords[:, axis].mean())

    def _build_index(self) -> List[dict]:
        items = []
        for case_id in self.cases:
            # Try all possible segmentation locations in order
            seg_path = None
            for candidate in [
                self.root / "segmentations" / "expert"    / f"{case_id}.nii.gz",
                self.root / "segmentations" / "automatic" / f"{case_id}.nii.gz",
                self.root / "segmentations"               / f"{case_id}.nii.gz",
            ]:
                if candidate.exists():
                    seg_path = candidate
                    break
            if seg_path is None:
                continue

            seg_nib  = nib.load(str(seg_path))
            seg_data = seg_nib.get_fdata(dtype=np.float32)
            # Canonical orientation: [H, W, D] (axial slices along last axis)
            seg_data = self._to_hwz(seg_data, seg_nib)

            if self.slice_mode == "tumour_centre":
                z_indices = [self._tumour_centre_slice(seg_data, axis=2)]
            else:
                occ = np.unique(np.argwhere(seg_data > 0)[:, 2])
                z_indices = occ.tolist() if len(occ) else [seg_data.shape[2] // 2]

            for z in z_indices:
                items.append({"case_id": case_id, "z_idx": int(z), "seg_path": str(seg_path)})
        return items

    # ------------------------------------------------------------------
    # Volume helpers
    # ------------------------------------------------------------------

    def _to_hwz(self, vol: np.ndarray, img_nib) -> np.ndarray:
        """
        Reorient volume to [H, W, D] (RAS+ axial).

        BUG FIX vs previous version: the old code guessed orientation from
        array shape, which is ambiguous (e.g. a 320×320×40 volume looks like
        both [H,W,D] and an unlikely [D,H,W]).  We now use nibabel's
        ornt_transform to canonicalise to RAS+ axial order reliably.
        """
        try:
            from nibabel.orientations import axcodes2ornt, ornt_transform, apply_orientation
            current_ornt = nib.io_orientation(img_nib.affine)
            target_ornt  = axcodes2ornt(("R", "A", "S"))
            transform    = ornt_transform(current_ornt, target_ornt)
            return apply_orientation(vol, transform).astype(np.float32)
        except Exception:
            # Fallback: assume [H, W, D]
            return vol.astype(np.float32)

    def _load_volume(self, case_id: str, phase: str) -> np.ndarray:
        """
        Load one DCE phase, reoriented to [H, W, D].
        MAMAMIA files are named {case_id}_{phase}.nii.gz
        e.g. DUKE_001_0000.nii.gz, DUKE_001_0001.nii.gz ...
        """
        p       = self.root / "images" / case_id / f"{case_id}_{phase}.nii.gz"
        img_nib = nib.load(str(p))
        vol     = img_nib.get_fdata(dtype=np.float32)
        return self._to_hwz(vol, img_nib)

    def _extract_patch(self, vol: np.ndarray, z: int) -> np.ndarray:
        """
        Extract a 2-D centre-crop from axial slice z of [H, W, D] volume.
        """
        slc    = vol[:, :, z]         # [H_vol, W_vol]
        H, W   = self.patch_size
        sh, sw = slc.shape

        y0  = max((sh - H) // 2, 0)
        x0  = max((sw - W) // 2, 0)
        slc = slc[y0: y0 + H, x0: x0 + W]

        # Pad if volume is smaller than patch_size
        ph = H - slc.shape[0]
        pw = W - slc.shape[1]
        if ph > 0 or pw > 0:
            slc = np.pad(slc, ((0, ph), (0, pw)))
        return slc.astype(np.float32)

    # ------------------------------------------------------------------
    # k-space simulation
    # ------------------------------------------------------------------

    def _simulate_kspace(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate single-coil k-space from a 2-D image and apply a random
        Cartesian undersampling mask at the requested acceleration.

        BUG FIX: total sampled lines = H / R.  The ACS lines are included in
        that budget; the random lines fill the rest.  The old code added
        H//R random lines *on top of* the ACS lines, over-sampling.

        Returns:
            us_kspace: [H, W, 2]  undersampled k-space (real/imag)
            mask:      [H, W]     binary sampling mask
        """
        img_t  = torch.from_numpy(img)
        kspace = torch.fft.fftn(img_t.cfloat(), dim=(-2, -1), norm="ortho")
        kspace = torch.view_as_real(kspace)                  # [H, W, 2]

        norm   = kspace.abs().max() + 1e-8
        kspace = kspace / norm

        H, W   = img.shape
        mask   = torch.zeros(H)
        centre = H // 2
        half   = self.acs_lines // 2
        mask[centre - half: centre + half] = 1.0

        # BUG FIX: target = H // R total lines; subtract ACS already set
        target_total = max(1, H // self.acceleration)
        extra_lines  = max(0, target_total - self.acs_lines)
        avail        = [i for i in range(H) if mask[i] == 0]
        chosen       = random.sample(avail, min(extra_lines, len(avail)))
        for l in chosen:
            mask[l] = 1.0

        mask_2d   = mask.unsqueeze(-1).expand(H, W)          # [H, W]
        us_kspace = kspace * mask_2d.unsqueeze(-1)            # [H, W, 2]
        return us_kspace.numpy(), mask_2d.numpy()

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    def _augment(self, phases: List[np.ndarray], seg: np.ndarray):
        """Random flips applied consistently across all phases and the mask."""
        if random.random() > 0.5:
            phases = [np.fliplr(p).copy() for p in phases]
            seg    = np.fliplr(seg).copy()
        if random.random() > 0.5:
            phases = [np.flipud(p).copy() for p in phases]
            seg    = np.flipud(seg).copy()
        return phases, seg

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def _parse_clinical(self, case_id: str) -> Optional[Tensor]:
        if not self.clinical_feats:
            return None
        row = self.clinical[case_id]
        vals = []
        for feat in self.clinical_feats:
            try:
                vals.append(float(row.get(feat, 0.0) or 0.0))
            except (ValueError, TypeError):
                vals.append(0.0)
        return torch.tensor(vals, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item    = self.items[idx]
        case_id = item["case_id"]
        z_idx   = item["z_idx"]

        # Load DCE phases and normalise to [0, 1] using 99th-percentile of
        # the pre-contrast volume so all phases share the same scale.
        # Without this, raw MRI intensities (~0-4000) create a scale mismatch
        # with the normalised k-space (~0-1), causing L1 loss ≈ thousands and
        # gradient explosion → NaN.
        phase_imgs_raw = [
            self._extract_patch(self._load_volume(case_id, ph), z_idx)
            for ph in self.PHASES[: self.n_phases]
        ]
        p99 = float(np.percentile(phase_imgs_raw[0], 99)) + 1e-6
        phase_imgs = [np.clip(p / p99, 0.0, 1.0).astype(np.float32)
                      for p in phase_imgs_raw]

        # Load segmentation
        seg_nib  = nib.load(item["seg_path"])
        seg_vol  = self._to_hwz(seg_nib.get_fdata(dtype=np.float32), seg_nib)
        seg_slc  = self._extract_patch(seg_vol, z_idx)
        seg_bin  = (seg_slc > 0).astype(np.float32)

        if self.augment:
            phase_imgs, seg_bin = self._augment(phase_imgs, seg_bin)

        # pCR label
        row     = self.clinical[case_id]
        pcr_val = row.get("pCR") or row.get("pcr") or row.get("PathologicResponse", "0")
        try:
            pcr = float(pcr_val)
        except (ValueError, TypeError):
            pcr = 0.0
        pcr_label = torch.tensor(pcr, dtype=torch.float32)

        # k-space simulation per phase
        us_kspaces, masks = [], []
        for img in phase_imgs:
            ks, mk = self._simulate_kspace(img)
            us_kspaces.append(torch.from_numpy(ks))
            masks.append(torch.from_numpy(mk))

        kspace      = torch.stack(us_kspaces, dim=0)              # [T, H, W, 2]
        masks_t     = torch.stack(masks, dim=0)                    # [T, H, W]
        target      = torch.stack(
            [torch.from_numpy(p) for p in phase_imgs], dim=0
        )                                                          # [T, H, W]
        tumour_mask = torch.from_numpy(seg_bin)                    # [H, W]

        # Add coil dimension (single-coil simulation)
        kspace = kspace.unsqueeze(1)                               # [T, 1, H, W, 2]

        # ACS region from full (unmasked) pre-contrast k-space
        pre_full = torch.from_numpy(phase_imgs[0])
        pre_ks   = torch.view_as_real(
            torch.fft.fftn(pre_full.cfloat(), dim=(-2, -1), norm="ortho")
        )                                                          # [H, W, 2]
        H_full    = pre_full.shape[0]
        centre    = H_full // 2
        half      = self.acs_lines // 2
        acs       = pre_ks[centre - half: centre + half, :, :]    # [H_acs, W, 2]
        acs       = acs.unsqueeze(0)                               # [1, H_acs, W, 2]

        # Approximate acquisition times in minutes
        t = torch.linspace(0.0, 5.0, self.n_phases)

        return {
            "kspace":      kspace,          # [T, 1, H, W, 2]
            "masks":       masks_t,         # [T, H, W]
            "target":      target,          # [T, H, W]
            "tumour_mask": tumour_mask,     # [H, W]
            "pcr_label":   pcr_label,       # scalar
            "acs_kspace":  acs,             # [1, H_acs, W, 2]
            "t":           t,               # [T] minutes
            "case_id":     case_id,
            "z_idx":       z_idx,
            "clinical":    self._parse_clinical(case_id),
        }


def mamamia_collate(batch: list) -> dict:
    out = {}
    for k in ["kspace", "masks", "target", "tumour_mask", "pcr_label", "t"]:
        out[k] = torch.stack([b[k] for b in batch])
    out["acs_kspace"] = torch.stack([b["acs_kspace"] for b in batch])
    out["case_ids"]   = [b["case_id"] for b in batch]
    clin = [b["clinical"] for b in batch]
    out["clinical"] = torch.stack(clin) if all(c is not None for c in clin) else None
    return out


def build_mamamia_loader(
    root: str,
    clinical_csv: str,
    split: str = "train",
    acceleration: int = 6,
    batch_size: int = 4,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    ds = MAMAMIADataset(root=root, clinical_csv=clinical_csv, split=split,
                        acceleration=acceleration, **kwargs)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=(split == "train"),
        num_workers=num_workers, collate_fn=mamamia_collate,
        pin_memory=True, drop_last=(split == "train"),
    )
