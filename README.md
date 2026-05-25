# VRDL Final Project — SIDD sRGB Image Denoising

**Author:** Chin Lu (Luluboy)  
**Course:** Visual Recognition with Deep Learning (VRDL) — Final Project  
**Competition:** [Kaggle SIDD Benchmark sRGB PSNR](https://www.kaggle.com/competitions/sidd-benchmark-srgb-psnr)  
**Best Score:** PSNR **40.4852** (NAFNet-TTA + Restormer-FT-TTA8, α=0.745, late submission) — Gold medal threshold ≥40.0019

---

## Overview

This project tackles image denoising on the SIDD (Smartphone Image Denoising Dataset) benchmark. Given 1280 noisy 256×256 sRGB blocks, the goal is to produce denoised outputs that maximize PSNR against the hidden ground truth.

## Methods & Results

| Method | PSNR (dB) | Kaggle Status |
|--------|-----------|--------------|
| NAFNet baseline | 40.3675 | ✅ Gold |
| Restormer baseline | 40.0902 | ✅ Gold |
| Restormer+TTA (FP32, 8-way) | 40.1351 | ✅ Gold |
| NAFNet+TTA | 39.5428 | ❌ Degraded |
| Ensemble v1 (α=0.5) | 39.9547 | ❌ Degraded |
| **NAFNet-TTA + Restormer-FT-TTA8 (α=0.745)** | **40.4852** | ✅ **Best (Gold)** |

## Project Structure

```
vrdl_final/
├── code/
│   ├── NAFNet/           # NAFNet repository (megvii-research/NAFNet)
│   ├── Restormer/        # Restormer repository (swz30/Restormer)
│   ├── ensemble.py        # Weighted ensemble pipeline
│   ├── make_submission.py # Baseline inference → CSV
│   ├── infer_restormer*.py # Restormer inference scripts
│   ├── tta_infer*.py     # Test-Time Augmentation scripts
│   ├── validate_*.py     # Local CSV format validation
│   └── prepare_training_patches.py
├── data/                 # SIDD dataset (gitignored, ~12GB)
├── docs/
│   └── plans/            # 7-phase execution plans
│       ├── 00-setup.md, 01-baseline.md, 02-tta.md
│       ├── 03-ensemble.md, 04-finetune.md
│       ├── 05-third-model.md, 06-inference-tricks.md, 07-stretch.md
│   └── presentation.html  # Slides (TBD)
├── weights/              # Pre-trained model checkpoints (gitignored)
│   ├── NAFNet-SIDD-width64.pth
│   └── RealDenoising_Restormer.pth
├── checkpoints/          # Fine-tuned model checkpoints (gitignored)
│   └── restormer_ft_epoch*.pth
├── logs/                 # Training/inference logs (gitignored)
├── submissions/          # Kaggle submission CSVs (gitignored, ~335MB each)
├── report/
│   ├── report_draft_v1.md  # Full project report draft
│   └── results_report.md   # PSNR results summary
├── train_restormer_finetune.py
├── requirements.txt
└── README.md
```

## Setup

```bash
# Environment: miniconda3 + Python 3.13 + PyTorch 2.5.1 + CUDA 12.1
conda create -n vrdl python=3.13
conda activate vrdl
pip install torch torchvision einops timm scipy h5py opencv-python tqdm pandas numpy

# GPU: NVIDIA RTX 3080 Ti (12GB)
```

## Inference

```bash
# NAFNet baseline
python code/make_submission.py

# Restormer + TTA
python code/tta_infer_restormer_float32.py

# Weighted ensemble v2 (0.7 NAFNet + 0.3 Restormer+TTA)
python code/ensemble.py
```

## Kaggle Submission

```bash
kaggle competitions submit \
  -c sidd-benchmark-srgb-psnr \
  -f submissions/SubmitSrgb_ensemble_v2.csv \
  -m "NAFNet-TTA + Restormer-FT-TTA8: α=0.745, PSNR 40.4852"
```

## Key Findings

- **TTA on NAFNet caused score degradation** due to a channel-order bug in the geometric transforms (flips + transpose). Only TTA on Restormer was used successfully.
- **Weighted ensemble outperformed simple averaging**: α=0.7 for NAFNet was optimal (tested α ∈ {0.3, 0.4, 0.5, 0.6, 0.7}).
- **Fine-tuning was not beneficial** for this dataset: fine-tuned Restormer achieved only 40.0899 PSNR (vs. 40.1351 with pre-trained + TTA).

---

## Team Contribution

| Member | Contributions |
|--------|--------------|
| Chin Lu (Luluboy) | All tasks: baseline setup, TTA, ensemble, fine-tuning, submission, report |

## References

- [NAFNet](https://github.com/megvii-research/NAFNet) — Simple Baselines for Image Restoration
- [Restormer](https://github.com/swz30/Restormer) — Efficient Transformer for Image Restoration
- [SIDD Dataset](https://abdokamel.github.io/sidd/) — Smartphone Image Denoising Dataset
- [Kaggle Competition](https://www.kaggle.com/competitions/sidd-benchmark-srgb-psnr) — SIDD Benchmark sRGB PSNR