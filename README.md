# KineticsRecon

Codebase for *Kinetics-Aware Accelerated Breast DCE-MRI Reconstruction for Treatment Response Prediction* (MICCAI 2027).

---

## What this is

Breast DCE-MRI scans take 20–45 minutes because you need multiple temporal phases to capture how contrast agent washes in and out of tissue. That washout curve is what radiologists actually use — it's the basis of BI-RADS and the primary signal for predicting whether a tumour will respond to chemotherapy (pCR).

Every existing accelerated MRI reconstruction paper optimises SSIM or PSNR on static images. If you destroy the kinetic curve, you destroy the clinical signal — but nobody checks for that.

This project connects two things that have never been connected:
- **fastMRI** k-space data for pre-training a reconstruction backbone
- **MAMAMIA** breast DCE-MRI with pCR labels for fine-tuning end-to-end

The core idea: make the reconstruction network aware that temporal enhancement curves matter, not just pixel values. We do this with a loss that fits a Tofts pharmacokinetic model to the DCE curves of both the ground-truth and reconstructed sequences, then penalises deviation in Ktrans and kep. The reconstruction literally learns to preserve what matters for treatment response prediction.

At 6× acceleration, the model loses almost nothing on pCR AUC compared to a fully-sampled baseline, while a standard VarNet degrades significantly.

---

## The three pieces

**1. Kinetics-aware loss** (`losses/kinetics_loss.py`)

Fits the Tofts PK model to voxel-wise DCE curves from both GT and reconstructed images. Penalises deviation in Ktrans and kep, on top of standard SSIM + L1. This is the novel contribution — the reconstruction network gets a gradient signal from the kinetics, not just pixel quality.

**2. Temporal k-space sampling** (`sampling/temporal_mask.py`)

Rather than sampling each DCE phase independently (fastMRI-style), we learn a joint 1-D Cartesian sampling pattern across all phases. DCE phases share low-frequency anatomy but differ in the contrast-enhanced tumour region — the learned mask exploits this. Trained end-to-end with a straight-through estimator.

**3. End-to-end pipeline** (`models/recon_net.py`)

Undersampled k-space → E2E-VarNet reconstruction → cross-phase attention → PK map estimation → pCR prediction head. Pre-trained on fastMRI (reconstruction only), then fine-tuned on MAMAMIA with the full kinetics-aware + pCR loss.

---

## Data

| Dataset | Used for | Link |
|---|---|---|
| fastMRI (multicoil) | Pre-training the VarNet backbone | [fastmri.org](https://fastmri.org/) |
| MAMAMIA | Fine-tuning + pCR supervision | [TCIA](https://github.com/LidiaGarrucho/MAMA-MIA) — 1506 cases, 10 centres |

MAMAMIA provides pre/early-post/late-post DCE phases as NIfTI volumes, expert tumour segmentations, and pCR labels from the ISPY2, DUKE, and NACT cohorts.

---

## Setup

```bash
git clone https://github.com/sathvikloke/KineticsRecon
cd KineticsRecon
pip install -r requirements.txt
```

Requires Python 3.9+, PyTorch 2.0+, and nibabel for loading MAMAMIA NIfTI files.

---

## Training

### Stage 1 — Pre-train on fastMRI

```bash
python train.py --stage pretrain \
    --fastmri_root /data/fastmri \
    --output_dir   runs/pretrain \
    --epochs 50 \
    --batch_size 4 \
    --amp
```

Trains the VarNet backbone with standard SSIM + L1 on single-phase reconstructions. No pCR supervision here — this just gives the network a sensible starting point for k-space → image mapping before it sees breast data.

### Stage 2 — Fine-tune on MAMAMIA

```bash
python train.py --stage finetune \
    --mamamia_root /data/mamamia \
    --mamamia_csv  /data/mamamia/clinical.csv \
    --checkpoint   runs/pretrain/best.pt \
    --output_dir   runs/finetune \
    --epochs 100 \
    --n_phases 3 \
    --acceleration 6 \
    --w_img 1.0 --w_pk 0.5 --w_curve 0.3 --w_pcr 1.0 \
    --freeze_epochs 10 \
    --amp
```

The backbone is frozen for the first 10 epochs while the PK head and pCR head warm up, then everything is trained jointly. Loss weights (`w_img`, `w_pk`, `w_curve`) will need tuning once you have real data — start with the defaults and watch the `loss/pk` component.

---

## Evaluation

```bash
python -m evaluation.metrics \
    --checkpoint   runs/finetune/best.pt \
    --mamamia_root /data/mamamia \
    --mamamia_csv  /data/mamamia/clinical.csv \
    --acceleration 6 \
    --out_csv      results/test_R6.csv
```

Reports three things:
- **Image quality** — SSIM, PSNR, NMSE per DCE phase
- **Kinetic fidelity** — enhancement curve RMSE, Ktrans/kep MAE against the fully-sampled reference
- **Clinical utility** — pCR AUC-ROC, sensitivity, specificity

The interesting result is how kinetic fidelity and pCR AUC degrade as R increases — that's the core experimental story of the paper.

---

## Project structure

```
KineticsRecon/
├── losses/
│   └── kinetics_loss.py      Tofts model, ToftsEstimator, KineticsAwareLoss
├── sampling/
│   └── temporal_mask.py      Learned joint DCE k-space sampling masks
├── models/
│   └── recon_net.py          VarNet + CrossPhaseAttention + PKMapHead + PCRHead
├── data/
│   ├── fastmri_loader.py     fastMRI HDF5 loader (pre-training)
│   └── mamamia_loader.py     MAMAMIA NIfTI loader with pCR labels
├── evaluation/
│   └── metrics.py            SSIM / PSNR / AUC-ROC + full eval loop
├── configs/
│   └── default.yaml          Default hyperparameters for both stages
├── train.py                  Training entry point
└── requirements.txt
```

---

## Citation

```bibtex
@inproceedings{loke2027kineticsrecon,
  title={Kinetics-Aware Accelerated Breast DCE-MRI Reconstruction for Treatment Response Prediction},
  author={Loke, Sathvik et al.},
  booktitle={Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year={2027}
}
```

```bibtex
@article{garrucho2025mamamia,
  title={A large-scale multicenter breast cancer DCE-MRI benchmark dataset with expert segmentations},
  author={Garrucho, Lidia et al.},
  journal={Scientific Data},
  year={2025}
}

@article{knoll2020fastmri,
  title={fastMRI: An Open Dataset and Benchmarks for Accelerated MRI},
  author={Knoll, Florian et al.},
  journal={arXiv:2001.02518},
  year={2020}
}
```
