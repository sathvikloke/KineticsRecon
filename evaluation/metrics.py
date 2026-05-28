"""
Evaluation metrics for KineticsRecon.

Computes three axes:
  1. Image quality   — SSIM, PSNR per reconstructed phase
  2. Kinetic fidelity — Ktrans / kep / ve MAE, enhancement curve RMSE
  3. Clinical utility  — pCR AUC-ROC, sensitivity, specificity

Usage:
    python -m evaluation.metrics \
        --checkpoint runs/finetune/best.pt \
        --mamamia_root /data/mamamia \
        --mamamia_csv  /data/mamamia/clinical.csv \
        --acceleration 6 \
        --out_csv      results/eval_R6.csv
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Image quality metrics
# ---------------------------------------------------------------------------

def ssim(x: Tensor, y: Tensor, window_size: int = 11) -> float:
    """
    Structural Similarity Index between two [H, W] or [B, H, W] tensors.
    Returns mean SSIM over batch.
    """
    if x.dim() == 2:
        x, y = x.unsqueeze(0), y.unsqueeze(0)
    B, H, W = x.shape
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    k = window_size
    kernel = torch.ones(1, 1, k, k, device=x.device) / (k * k)

    def mu(t):
        return F.conv2d(t.unsqueeze(1), kernel, padding=k // 2).squeeze(1)

    mx, my   = mu(x), mu(y)
    mxx, myy = mu(x * x), mu(y * y)
    mxy      = mu(x * y)

    vx  = mxx - mx ** 2
    vy  = myy - my ** 2
    vxy = mxy - mx * my

    num = (2 * mx * my + C1) * (2 * vxy + C2)
    den = (mx ** 2 + my ** 2 + C1) * (vx + vy + C2)
    return (num / den).mean().item()


def psnr(x: Tensor, y: Tensor, data_range: float = None) -> float:
    """Peak Signal-to-Noise Ratio in dB."""
    if data_range is None:
        data_range = max(y.max().item() - y.min().item(), 1e-8)
    mse = F.mse_loss(x, y).item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(data_range ** 2 / mse)


def nmse(x: Tensor, y: Tensor) -> float:
    """Normalised Mean Squared Error."""
    return (F.mse_loss(x, y) / (y ** 2).mean()).item()


def compute_metrics(recon: Tensor, target: Tensor) -> Dict[str, float]:
    """Compute all image quality metrics for a [B, H, W] pair."""
    recon  = recon.detach().cpu()
    target = target.detach().cpu()
    return {
        "ssim": ssim(recon, target),
        "psnr": psnr(recon, target),
        "nmse": nmse(recon, target),
    }


# ---------------------------------------------------------------------------
# Kinetic fidelity metrics
# ---------------------------------------------------------------------------

def enhancement_curve_rmse(recon: Tensor, target: Tensor) -> float:
    """
    RMSE of voxel-wise normalised enhancement curves.
    recon, target: [B, T, H, W]

    Clamp enhancement to [-10, 10] to suppress blow-up from near-zero
    pre-contrast voxels (same guard used in the training curve loss).
    """
    pre_r = recon[:, 0:1]
    pre_t = target[:, 0:1]
    enh_r = ((recon  - pre_r)  / (pre_r.abs()  + 1e-6)).clamp(-10, 10)
    enh_t = ((target - pre_t)  / (pre_t.abs()  + 1e-6)).clamp(-10, 10)
    return F.mse_loss(enh_r, enh_t).sqrt().item()


def pk_map_mae(pk_recon: Tensor, pk_target: Tensor) -> Dict[str, float]:
    """
    Mean absolute error on pharmacokinetic parameter maps.
    pk_*: [B, 3, H, W]  (Ktrans, kep, ve)
    """
    names = ["Ktrans", "kep", "ve"]
    out = {}
    for i, name in enumerate(names):
        out[f"MAE_{name}"] = F.l1_loss(pk_recon[:, i], pk_target[:, i]).item()
    return out


# ---------------------------------------------------------------------------
# Classification metrics (pCR)
# ---------------------------------------------------------------------------

def compute_roc(labels: List[int], probs: List[float]) -> Tuple[List, List, float]:
    """Compute ROC curve and AUC (trapezoidal)."""
    pairs   = sorted(zip(probs, labels), reverse=True)
    n_pos   = sum(labels)
    n_neg   = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return [0, 1], [0, 1], 0.5

    tprs, fprs = [0.0], [0.0]
    tp = fp = 0
    auc = 0.0
    prev_fpr = prev_tpr = 0.0

    for _, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        tprs.append(tpr)
        fprs.append(fpr)
        prev_fpr, prev_tpr = fpr, tpr

    tprs.append(1.0)
    fprs.append(1.0)
    return fprs, tprs, auc


def sensitivity_specificity(
    labels: List[int],
    probs: List[float],
    threshold: float = 0.5,
) -> Tuple[float, float]:
    preds = [1 if p >= threshold else 0 for p in probs]
    tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
    tn = sum(p == 0 and l == 0 for p, l in zip(preds, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return sens, spec


# ---------------------------------------------------------------------------
# Full evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    model,
    sampler,
    loader,
    criterion,
    device,
    acceleration: int,
    n_phases: int = 3,
) -> Dict:
    """
    Run evaluation on a DataLoader and return aggregated metrics.
    """
    model.eval()
    sampler.eval()

    all_ssim  = []
    all_psnr  = []
    all_nmse  = []
    all_curve = []
    all_probs = []
    all_labels = []
    case_results = []

    with torch.no_grad():
        for batch in loader:
            kspace    = batch["kspace"].to(device)
            target    = batch["target"].to(device)
            acs       = batch["acs_kspace"].to(device)
            tumour_mk = batch["tumour_mask"].to(device)
            pcr_label = batch["pcr_label"].to(device)
            t_vec     = batch["t"][0].to(device)
            clinical  = batch.get("clinical")
            if clinical is not None:
                clinical = clinical.to(device)
            case_ids  = batch["case_ids"]

            masks_v = sampler()
            masks_b = masks_v.unsqueeze(0).unsqueeze(2).unsqueeze(-1)
            us_kspace = kspace * masks_b

            out    = model(us_kspace, masks_v, acs,
                           tumour_mask=tumour_mk, clinical=clinical)
            recon  = out["recon_phases"]
            logit  = out["pcr_logit"]

            # Image quality per phase
            for t in range(n_phases):
                m = compute_metrics(recon[:, t], target[:, t])
                all_ssim.append(m["ssim"])
                all_psnr.append(m["psnr"])
                all_nmse.append(m["nmse"])

            # Enhancement curve fidelity
            all_curve.append(enhancement_curve_rmse(
                recon.cpu(), target.cpu()
            ))

            # pCR
            probs = torch.sigmoid(logit.squeeze(1)).cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(pcr_label.cpu().int().tolist())

            for i, cid in enumerate(case_ids):
                case_results.append({
                    "case_id": cid,
                    "pcr_prob": probs[i],
                    "pcr_label": int(pcr_label[i].item()),
                })

    _, _, auc = compute_roc(all_labels, all_probs)
    sens, spec = sensitivity_specificity(all_labels, all_probs)

    return {
        "SSIM":          sum(all_ssim) / len(all_ssim),
        "PSNR":          sum(all_psnr) / len(all_psnr),
        "NMSE":          sum(all_nmse) / len(all_nmse),
        "curve_RMSE":    sum(all_curve) / len(all_curve),
        "pCR_AUC":       auc,
        "pCR_sens":      sens,
        "pCR_spec":      spec,
        "n_cases":       len(case_results),
        "case_results":  case_results,
    }


def print_results(results: Dict, label: str = ""):
    print(f"\n{'='*50}")
    if label:
        print(f"  {label}")
    print(f"  Image quality:  SSIM={results['SSIM']:.4f}  PSNR={results['PSNR']:.2f}  NMSE={results['NMSE']:.4f}")
    print(f"  Kinetic fidelity: curve RMSE={results['curve_RMSE']:.4f}")
    print(f"  pCR prediction: AUC={results['pCR_AUC']:.4f}  Sens={results['pCR_sens']:.3f}  Spec={results['pCR_spec']:.3f}")
    print(f"  N cases: {results['n_cases']}")
    print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser("KineticsRecon evaluation")
    p.add_argument("--checkpoint",    required=True)
    p.add_argument("--mamamia_root",  required=True)
    p.add_argument("--mamamia_csv",   required=True)
    p.add_argument("--acceleration",  type=int, default=6)
    p.add_argument("--n_phases",      type=int, default=3)
    p.add_argument("--batch_size",    type=int, default=4)
    p.add_argument("--out_csv",       default="results.csv")
    args = p.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from models.recon_net import KineticsRecon
    from sampling.temporal_mask import TemporalKSpaceSampler
    from losses.kinetics_loss import KineticsAwareLoss, ToftsEstimator
    from data.mamamia_loader import build_mamamia_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = KineticsRecon(n_coils=1, n_phases=args.n_phases).to(device)
    sampler = TemporalKSpaceSampler(
        acceleration=args.acceleration, n_phases=args.n_phases
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    sampler.load_state_dict(ckpt["sampler"])

    test_loader = build_mamamia_loader(
        root=args.mamamia_root,
        clinical_csv=args.mamamia_csv,
        split="test",
        acceleration=args.acceleration,
        batch_size=args.batch_size,
        n_phases=args.n_phases,
    )

    results = evaluate(
        model, sampler, test_loader,
        criterion=None, device=device,
        acceleration=args.acceleration,
        n_phases=args.n_phases,
    )

    print_results(results, label=f"Test set  R={args.acceleration}×")

    # Write case-level CSV
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "pcr_prob", "pcr_label"])
        writer.writeheader()
        writer.writerows(results["case_results"])

    # Write summary JSON
    summary = {k: v for k, v in results.items() if k != "case_results"}
    with open(out.with_suffix(".json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results written to {out} and {out.with_suffix('.json')}")
