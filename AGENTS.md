# AGENTS.md — VRDL Final Project: SIDD Denoising

## Project Overview
- **Competition**: SIDD Benchmark - sRGB - PSNR (https://www.kaggle.com/competitions/sidd-benchmark-srgb-psnr)
- **Task**: Image denoising on real smartphone images (SIDD dataset)
- **Metric**: PSNR on sRGB benchmark blocks (1280 blocks of 256×256)
- **Goal**: Maximize Kaggle leaderboard score

## Grading Reference (50% of total grade is performance)
- PSNR ≥ 40.00 → Gold medal (top 10 of 98) → 25 pts
- Top-3 rank → 40 pts
- PSNR ≥ Best × 1.02 (≈ 41.56) → 50 pts (full performance marks)

## Hardware
- Main training GPU: NVIDIA RTX 3080 Ti (12 GB VRAM)
- Must use AMP (`torch.cuda.amp`) to fit fine-tuning at batch 4-6
- Free Kaggle / Colab GPUs can run parallel ensemble experiments

## High-Level Strategy
1. Start with NAFNet-width64 pretrained → expect PSNR ≈ 40.30 → gold medal
2. Add ×8 self-ensemble (TTA) → +0.05-0.10 dB
3. Add Restormer ensemble → +0.10-0.15 dB
4. Fine-tune NAFNet on SIDD-Medium → +0.10-0.20 dB
5. Add 3rd model (MIRNet-v2 / DRCT / MambaIR) → +0.05-0.10 dB
6. Inference tricks (patch overlap, multi-scale TTA) → +0.02-0.05 dB
7. Stretch goal: NAFNet-width96 fine-tune / self-distillation → push past 41.5

## Plans Directory
Detailed phase-by-phase todo lists are in `docs/plans/`:
- `00-setup.md` — Environment & resource preparation
- `01-baseline.md` — First submission (NAFNet direct inference)
- `02-tta.md` — ×8 self-ensemble TTA
- `03-ensemble.md` — Multi-model ensemble (NAFNet + Restormer)
- `04-finetune.md` — Fine-tune NAFNet on SIDD-Medium
- `05-third-model.md` — Add MIRNet-v2 / DRCT / MambaIR
- `06-inference-tricks.md` — Patch overlap, multi-scale, noise estimation
- `07-stretch.md` — Stretch goals for 50 pts tier

## Key Repos / References
- NAFNet: https://github.com/megvii-research/NAFNet
- Restormer: https://github.com/swz30/Restormer
- MIRNet-v2: https://github.com/swz30/MIRNetv2
- Uformer: https://github.com/ZhendongWang6/Uformer
- DRCT: https://github.com/ming053l/DRCT
- MambaIR: https://github.com/csguoh/MambaIR
- SIDD Kaggle submission script reference: https://github.com/AbdoKamel/sidd_kaggle_submit

## Agent Working Rules
- Before submitting to Kaggle, always sanity-check with single-model run to avoid wasting daily quota (5 submissions/day).
- Maintain `submissions/log.csv` with: commit hash, config, expected PSNR, actual PSNR, rank.
- VRAM OOM handling: reduce batch → enable AMP → gradient checkpointing (NEVER reduce patch size below 256).
- Always preserve original pretrained weights as fallback baseline.
- Never attempt full from-scratch training (NAFNet requires 8× A100 × 2 days).
- Submission CSV format errors are the most common pitfall. Always do a round-trip with raw noisy data first to verify block ordering before submitting model output.

## Directory Structure
```
vrdl_final/
├── AGENTS.md
├── docs/
│   └── plans/
│       ├── 00-setup.md
│       ├── 01-baseline.md
│       ├── 02-tta.md
│       ├── 03-ensemble.md
│       ├── 04-finetune.md
│       ├── 05-third-model.md
│       ├── 06-inference-tricks.md
│       └── 07-stretch.md
├── code/
├── data/
│   └── BenchmarkNoisyBlocksSrgb.mat
├── weights/
│   ├── NAFNet-SIDD-width64.pth
│   └── ...
├── submissions/
│   └── log.csv
└── logs/
```
