"""
Two-stage training for KineticsRecon.

Stage 1 — Pre-training (fastMRI):
    VarNet backbone trained for single-phase reconstruction (standard SSIM+L1).

Stage 2 — Fine-tuning (MAMAMIA):
    Full model trained with kinetics-aware loss + pCR BCE.
    Backbone optionally frozen for the first N epochs.

Usage:
    python train.py --stage pretrain  --fastmri_root /data/fastmri --output_dir runs/pretrain
    python train.py --stage finetune  --mamamia_root /data/mamamia --mamamia_csv /data/mamamia/clinical.csv \
                                      --checkpoint runs/pretrain/best.pt --output_dir runs/finetune
"""

import argparse
import json
import time
from pathlib import Path
from typing import List

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

from data.fastmri_loader import build_fastmri_loader
from data.mamamia_loader import build_mamamia_loader
from evaluation.metrics import compute_metrics
from losses.kinetics_loss import KineticsAwareLoss, ToftsEstimator
from models.recon_net import KineticsRecon
from sampling.temporal_mask import TemporalKSpaceSampler


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def log(msg: str, log_file=None):
    print(msg)
    if log_file:
        with open(log_file, "a") as f:
            f.write(msg + "\n")


def save_checkpoint(state: dict, path: str):
    torch.save(state, path)
    print(f"  saved → {path}")


def load_checkpoint(model: nn.Module, path: str, strict: bool = True):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    saved_sd = ckpt["model"]

    if not strict:
        # Filter out keys whose shape doesn't match the current model.
        # This handles the pretrain→finetune transfer where pk_head and
        # cross_phase have different n_phases-dependent shapes.
        model_sd   = model.state_dict()
        filtered   = {}
        skipped    = []
        for k, v in saved_sd.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                filtered[k] = v
            else:
                skipped.append(k)
        if skipped:
            print(f"  skipped {len(skipped)} keys (shape mismatch): {skipped[:3]}{'...' if len(skipped)>3 else ''}")
        missing, unexpected = model.load_state_dict(filtered, strict=False)
    else:
        missing, unexpected = model.load_state_dict(saved_sd, strict=True)

    if missing:
        print(f"  missing keys: {missing[:3]}{'...' if len(missing)>3 else ''}")
    if unexpected:
        print(f"  unexpected keys: {unexpected[:3]}{'...' if len(unexpected)>3 else ''}")
    return ckpt.get("epoch", 0), ckpt.get("best_metric", None)


def _ssim_scalar(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Global-statistics SSIM approximation for use in the pretrain loop."""
    mu_x, mu_y = x.mean(), y.mean()
    s_x  = ((x - mu_x) ** 2).mean().clamp(min=0).sqrt()
    s_y  = ((y - mu_y) ** 2).mean().clamp(min=0).sqrt()
    s_xy = ((x - mu_x) * (y - mu_y)).mean()
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu_x * mu_y + C1) * (2 * s_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (s_x ** 2 + s_y ** 2 + C2)
    return num / (den + 1e-8)


def _compute_auc(labels: List[int], probs: List[float]) -> float:
    """Trapezoidal AUC-ROC without sklearn."""
    n_pos = sum(labels);  n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    pairs = sorted(zip(probs, labels), reverse=True)
    auc = prev_fpr = prev_tpr = 0.0
    tp = fp = 0
    for _, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos;  fpr = fp / n_neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_fpr, prev_tpr = fpr, tpr
    return auc


# ---------------------------------------------------------------------------
# Stage 1: Pre-training on fastMRI
# ---------------------------------------------------------------------------

def pretrain(args):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "log.txt"
    log(f"[Pretrain] device={device}  output={out_dir}", log_file)

    train_loader = build_fastmri_loader(
        root=args.fastmri_root, split="train",
        acceleration=args.acceleration,
        batch_size=args.batch_size, num_workers=args.num_workers,
        max_slices=args.max_slices,
    )
    val_loader = build_fastmri_loader(
        root=args.fastmri_root, split="val",
        acceleration=args.acceleration,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    # Single-phase model for fastMRI
    model = KineticsRecon(
        n_coils=args.n_coils, n_phases=1,
        n_varnet_blocks=args.n_blocks,
        full_height=args.patch_size,
    ).to(device)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    optimiser = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=1e-6)
    scaler    = GradScaler(device_str, enabled=args.amp)
    best_ssim = -1.0

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Pretraining", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        t0 = time.time()
        train_loss = 0.0

        batch_bar = tqdm(train_loader, desc=f"  E{epoch:03d} train", leave=False,
                         unit="batch", dynamic_ncols=True)
        for batch in batch_bar:
            kspace = batch["kspace"].to(device)
            target = batch["target"].to(device)
            acs    = batch["acs_kspace"].to(device)
            mask   = batch["mask"].to(device)
            kspace_t = kspace.unsqueeze(1)
            mask_t   = mask.unsqueeze(0)

            with autocast(device_str, enabled=args.amp):
                out   = model(kspace_t, mask_t, acs)
                recon = out["recon_phases"][:, 0]
                l1    = nn.functional.l1_loss(recon, target)
                loss  = l1 + (1 - _ssim_scalar(recon, target))

            optimiser.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimiser)
            scaler.update()
            train_loss += loss.item()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        # Validation
        model.eval()
        all_ssim, all_psnr = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"  E{epoch:03d} val  ", leave=False,
                              unit="batch", dynamic_ncols=True):
                kspace = batch["kspace"].unsqueeze(1).to(device)
                target = batch["target"].to(device)
                acs    = batch["acs_kspace"].to(device)
                mask   = batch["mask"].to(device)
                mask_t = mask.unsqueeze(0)
                out    = model(kspace, mask_t, acs)
                recon  = out["recon_phases"][:, 0]
                m      = compute_metrics(recon, target)
                all_ssim.append(m["ssim"]);  all_psnr.append(m["psnr"])

        avg_ssim = sum(all_ssim) / len(all_ssim)
        avg_psnr = sum(all_psnr) / len(all_psnr)
        avg_loss = train_loss / len(train_loader)
        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}", SSIM=f"{avg_ssim:.4f}", PSNR=f"{avg_psnr:.1f}")
        log(f"Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  "
            f"SSIM={avg_ssim:.4f}  PSNR={avg_psnr:.2f}  ({time.time()-t0:.0f}s)", log_file)

        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            save_checkpoint({"model": model.state_dict(), "epoch": epoch, "best_metric": best_ssim},
                            str(out_dir / "best.pt"))

    save_checkpoint({"model": model.state_dict(), "epoch": args.epochs}, str(out_dir / "final.pt"))
    log(f"Pre-training complete. Best SSIM={best_ssim:.4f}", log_file)


# ---------------------------------------------------------------------------
# Stage 2: Fine-tuning on MAMAMIA
# ---------------------------------------------------------------------------

def finetune(args):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "log.txt"
    log(f"[Finetune] device={device}  output={out_dir}", log_file)

    train_loader = build_mamamia_loader(
        root=args.mamamia_root, clinical_csv=args.mamamia_csv,
        split="train", acceleration=args.acceleration,
        batch_size=args.batch_size, num_workers=args.num_workers,
        n_phases=args.n_phases, patch_size=(args.patch_size, args.patch_size),
        augment=True,
    )
    val_loader = build_mamamia_loader(
        root=args.mamamia_root, clinical_csv=args.mamamia_csv,
        split="val", acceleration=args.acceleration,
        batch_size=args.batch_size, num_workers=args.num_workers,
        n_phases=args.n_phases, patch_size=(args.patch_size, args.patch_size),
    )

    model = KineticsRecon(
        n_coils=1, n_phases=args.n_phases,
        n_varnet_blocks=args.n_blocks,
        full_height=args.patch_size,
    ).to(device)

    if args.checkpoint:
        start_epoch, _ = load_checkpoint(model, args.checkpoint, strict=False)
        log(f"  Loaded pretrained checkpoint: {args.checkpoint}", log_file)
    else:
        start_epoch = 0

    sampler   = TemporalKSpaceSampler(
        acceleration=args.acceleration, n_phases=args.n_phases,
        shape=(args.patch_size, args.patch_size),
    ).to(device)
    estimator = ToftsEstimator().to(device)
    criterion = KineticsAwareLoss(
        w_img=args.w_img, w_pk=args.w_pk, w_curve=args.w_curve,
        pk_estimator=estimator,
    )
    bce_loss  = nn.BCEWithLogitsLoss()

    backbone_params = (list(model.sens_model.parameters())
                       + list(model.varnet_blocks.parameters()))
    head_params     = (list(model.pk_head.parameters())
                       + list(model.pcr_head.parameters())
                       + list(model.cross_phase.parameters())
                       + list(sampler.parameters())
                       + list(estimator.parameters()))

    optimiser = optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params,     "lr": args.lr},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=1e-6)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    scaler    = GradScaler(device_str, enabled=args.amp)

    if args.freeze_epochs > 0:
        model.freeze_backbone()
        log(f"  Backbone frozen for {args.freeze_epochs} epochs", log_file)

    best_auc = -1.0
    history  = []

    epoch_bar = tqdm(range(start_epoch + 1, start_epoch + args.epochs + 1),
                     desc="Finetuning", unit="epoch")
    for epoch in epoch_bar:

        if args.freeze_epochs > 0 and epoch == start_epoch + args.freeze_epochs + 1:
            model.unfreeze_all()
            log("  Backbone unfrozen", log_file)

        # ---- Train ----
        model.train();  sampler.train()
        t0          = time.time()
        ep_metrics  = {k: 0.0 for k in ["recon", "pk", "pcr", "total"]}
        n_steps     = 0

        for batch in tqdm(train_loader, desc=f"  E{epoch:03d} train", leave=False,
                          unit="batch", dynamic_ncols=True):
            kspace    = batch["kspace"].to(device)        # [B, T, 1, H, W, 2]
            target    = batch["target"].to(device)        # [B, T, H, W]
            t_vec     = batch["t"][0].to(device)          # [T]
            tumour_mk = batch["tumour_mask"].to(device)   # [B, H, W]
            pcr_label = batch["pcr_label"].to(device)     # [B]
            acs       = batch["acs_kspace"].to(device)    # [B, 1, H_acs, W, 2]
            clinical  = batch.get("clinical")
            if clinical is not None:
                clinical = clinical.to(device)

            # Compute masks once and reuse for both undersampling the input
            # and passing into the model's DataConsistency step.  Calling
            # sampler() twice would run two separate straight-through forward
            # passes; while currently identical (no randomness), it's fragile
            # and wastes compute.
            masks_t = sampler()                                    # [T, H, W]
            m_bc    = masks_t.unsqueeze(0).unsqueeze(2).unsqueeze(-1)  # [1,T,1,H,W,1]
            us_ksp  = kspace * m_bc                               # [B, T, 1, H, W, 2]

            with autocast(device_str, enabled=args.amp):
                out      = model(us_ksp, masks_t, acs, tumour_mask=tumour_mk, clinical=clinical)
                recon    = out["recon_phases"]            # [B, T, H, W]
                logit    = out["pcr_logit"]               # [B, 1]

                kin_loss, comps = criterion(recon, target, t_vec, mask=tumour_mk)
                pcr_loss        = bce_loss(logit.squeeze(1), pcr_label)
                total_loss      = kin_loss + args.w_pcr * pcr_loss

            # Skip batch if any loss is NaN (guards against bad slices)
            if not torch.isfinite(total_loss):
                optimiser.zero_grad()
                continue

            optimiser.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(sampler.parameters()), 1.0
            )
            scaler.step(optimiser);  scaler.update()

            ep_metrics["recon"] += comps["loss/image"]
            ep_metrics["pk"]    += comps["loss/pk"]
            ep_metrics["pcr"]   += pcr_loss.item()
            ep_metrics["total"] += total_loss.item()
            n_steps += 1

        scheduler.step()
        for k in ep_metrics:
            ep_metrics[k] /= max(n_steps, 1)

        # ---- Validate ----
        model.eval();  sampler.eval()
        val_ssim, val_psnr, val_probs, val_labels = [], [], [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"  E{epoch:03d} val  ", leave=False,
                              unit="batch", dynamic_ncols=True):
                kspace    = batch["kspace"].to(device)
                target    = batch["target"].to(device)
                acs       = batch["acs_kspace"].to(device)
                tumour_mk = batch["tumour_mask"].to(device)
                pcr_label = batch["pcr_label"].to(device)
                clinical  = batch.get("clinical")
                if clinical is not None:
                    clinical = clinical.to(device)

                masks_t = sampler()
                m_bc    = masks_t.unsqueeze(0).unsqueeze(2).unsqueeze(-1)
                us_ksp  = kspace * m_bc
                out      = model(us_ksp, masks_t, acs, tumour_mask=tumour_mk, clinical=clinical)
                recon    = out["recon_phases"]
                logit    = out["pcr_logit"]

                for t in range(recon.shape[1]):
                    m = compute_metrics(recon[:, t], target[:, t])
                    val_ssim.append(m["ssim"]);  val_psnr.append(m["psnr"])

                val_probs.extend(torch.sigmoid(logit.squeeze(1)).cpu().tolist())
                val_labels.extend(pcr_label.cpu().int().tolist())

        avg_ssim = sum(val_ssim) / len(val_ssim)
        avg_psnr = sum(val_psnr) / len(val_psnr)
        auc      = _compute_auc(val_labels, val_probs)

        msg = (f"Epoch {epoch:3d}  recon={ep_metrics['recon']:.4f}  "
               f"pk={ep_metrics['pk']:.4f}  pcr={ep_metrics['pcr']:.4f}  |  "
               f"val SSIM={avg_ssim:.4f}  PSNR={avg_psnr:.1f}  AUC={auc:.4f}  "
               f"({time.time()-t0:.0f}s)")
        epoch_bar.set_postfix(SSIM=f"{avg_ssim:.4f}", AUC=f"{auc:.4f}",
                              recon=f"{ep_metrics['recon']:.4f}")
        log(msg, log_file)
        history.append({"epoch": epoch, "ssim": avg_ssim, "psnr": avg_psnr, "auc": auc})

        if auc > best_auc:
            best_auc = auc
            save_checkpoint(
                {"model": model.state_dict(), "sampler": sampler.state_dict(),
                 "estimator": estimator.state_dict(), "epoch": epoch, "best_metric": best_auc},
                str(out_dir / "best.pt"),
            )

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    log(f"Fine-tuning complete. Best pCR AUC={best_auc:.4f}", log_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("KineticsRecon training")
    p.add_argument("--stage", choices=["pretrain", "finetune"], required=True)
    p.add_argument("--output_dir",   default="runs/default")
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--acceleration", type=int,   default=6)
    p.add_argument("--n_coils",      type=int,   default=8)
    p.add_argument("--n_blocks",     type=int,   default=8)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--patch_size",   type=int,   default=320)
    p.add_argument("--amp",          action="store_true")
    # Pretrain
    p.add_argument("--fastmri_root",  default=None)
    p.add_argument("--max_slices",    type=int, default=None,
                   help="Cap total training samples (speeds up CPU runs)")
    # Finetune
    p.add_argument("--mamamia_root", default=None)
    p.add_argument("--mamamia_csv",  default=None)
    p.add_argument("--checkpoint",   default=None)
    p.add_argument("--n_phases",     type=int,   default=3)
    p.add_argument("--freeze_epochs",type=int,   default=5)
    p.add_argument("--w_img",        type=float, default=1.0)
    p.add_argument("--w_pk",         type=float, default=0.5)
    p.add_argument("--w_curve",      type=float, default=0.3)
    p.add_argument("--w_pcr",        type=float, default=1.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.stage == "pretrain":
        pretrain(args)
    else:
        finetune(args)
