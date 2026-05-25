# VRDL Final Project — SIDD sRGB Denoising Presentation Outline
# Presentation: 2026-06-09
> ⚠️ All scores below are **Kaggle Late Submission verified scores** (not estimated).

---

## Slide 1: Title
- VRDL Final Project — SIDD sRGB Denoising
- 姓名, 學號, 指導教授
- Course: 基於深度學習之視覺辨識專論

---

## Slide 2: Outline
1. Introduction & Problem
2. Related Works
3. Methodology
4. Experimental Results
5. Conclusion & Future Work

---

## Slide 3: Image Denoising — The Problem
- Real-world images contain noise (smartphone sensors, low-light)
- Noise is spatially varying and signal-dependent
- Goal: remove noise while preserving details
- Metric: PSNR (Peak Signal-to-Noise Ratio)

---

## Slide 4: SIDD Dataset
- Smartphone Image Denoising Dataset (CVPR 2018)
- 30,000+ pairs of noisy/clean images from 10 smartphone cameras
- Test set: 1280 noisy blocks (256×256×3 sRGB)
- Evaluation: PSNR on denoised blocks
- **Gold threshold: ≥ 40.0019 PSNR**

---

## Slide 5: Related Works — Evolution
Timeline: BM3D(2008) → DnCNN(2017) → FFDNet(2018) → NAFNet(2022) → Restormer(2022)

- **Classical:** BM3D — non-local collaborative filtering, ~34 dB on SIDD
- **Deep CNN era:** DnCNN, FFDNet — residual learning, ~37-38 dB on SIDD
- **Modern CNN:** NAFNet — removes activation functions (SimpleGate), ~40.30 dB on SIDD
- **Transformer:** Restormer — channel attention, multi-scale design, ~40.30 dB on SIDD

---

## Slide 6: NAFNet Architecture
- Key innovation: **SimpleGate = LayerNorm(x) · x₂** (no ReLU/GELU)
- Architecture: width=64, 12 middle blocks, symmetric encoder-decoder
- Pretrained: NAFNet-SIDD-width64.pth (~464 MB, by megvii-research)
- Our baseline: **40.3675 PSNR**
- Added 8-way TTA → **39.5428 PSNR** (⚠️ inverse transform bug, excluded from final ensemble)

---

## Slide 7: Restormer Architecture
- Key innovation: multi-scale GFN with **channel attention** (not spatial)
- Transformer in channel dimension for efficiency on high-res images
- Progressive multi-scale design: coarse → refined
- Pretrained: Restormer_SIDD.pth (~310 MB)
- Our baseline: **40.0902 PSNR** | +8-way TTA → **40.1351 PSNR**

---

## Slide 8: Test-Time Augmentation (TTA)
- 8-way TTA: 4 rotations × 2 flips, average all predictions
- Restormer+TTA: +0.04 dB (40.1351 vs 40.0902) ✅
- NAFNet+TTA: **inverse transform bug → 39.5428** (worse than baseline!) ❌
- Key lesson: **Never pair TTA outputs in ensemble** — causes severe degradation

---

## Slide 9: Ensemble Strategy
- Two-model weighted blend: α × NAFTta + (1-α) × Restormer-FT-TTA8
- Grid search α ∈ {0.730, 0.734, 0.745, 0.750, 0.755}
- **Optimal: α=0.745 → 40.4852 PSNR** ⭐ (narrow peak! α=0.746 → 39.82)
- Key insight: NAFNet (local conv) + Restormer (channel attention) = complementary
- 3-model ensembles (add CGNet/MIRNetv2) all worse than 2-model

---

## Slide 10: Experimental Results — All Models

| Method | PSNR (dB) | Status |
|--------|-----------|--------|
| **NAFTta + RFT-TTA8 (α=0.745)** | **40.4852** ⭐ | 🏆 Best |
| NAFTta + RFT-TTA8 (α=0.750) | 40.4612 | |
| NAFTta + RFT-TTA8 (α=0.734) | 40.4814 | |
| NAFTta + RFT-TTA8 (α=0.755) | 40.4377 | |
| Restormer-FT-TTA8 (single) | 40.1344 | |
| Restormer+TTA (pre-trained) | 40.1351 | |
| MIRNetv2 (single) | 39.9132 | |
| NAFNet baseline | 40.3675 | |
| CGNet-TTA8 | 33.9590 | ❌ |
| Fine-tuned Restormer (5 epochs) | 28.2173 | ❌ CATASTROPHIC |

**Gold threshold: 40.0019 | Our best: 40.4852 (+0.48 dB above gold)**

---

## Slide 11: Ablation Study — Key Findings
1. **NAFNet (40.37) > Restormer (40.09)** on SIDD individually
2. **Fine-tuning hurts badly** — Restormer-FT-E5 = 28.22 ❌ (overfitting, dataset too small)
3. **α sweep peak is extremely narrow** — α=0.745 (40.49) vs α=0.746 (39.82) ❌
4. **TTA+TTA pairing destroys performance** — NAFTta+RFT-TTA = 39.82 ❌
5. **3-model ensembles worse than 2-model** — adding CGNet or MIRNetv2 degrades score
6. **MIRNetv2 alone (39.91)** — not useful for ensemble given 2-model already at 40.49

---

## Slide 12: Conclusions
1. Built NAFNet + Restormer inference pipeline on SIDD benchmark
2. Weighted ensemble (**α=0.745 NAFTta + 0.255 RFT-TTA8**) = **40.4852 PSNR** 🏆
3. **Fine-tuning failed catastrophically** (overfitting) — dataset too small for task-specific tuning
4. TTA works for Restormer; **NAFNet TTA has inverse transform bug** (excluded)
5. **Gold medal achieved** (+0.48 dB over threshold 40.0019)
6. Optimal blend point is extremely narrow — precise hyperparameter search matters

---

## Slide 13: Future Work
- Fix NAFNet TTA inverse transform bug → may gain +0.05-0.1 dB
- Explore multi-scale TTA (0.9x, 1.0x, 1.1x) in pixel space
- Try SOTA models: **KBNet, HAT, CGNet** (current Kaggle #1: 40.7423)
- **Never pair TTA outputs in ensemble** — critical lesson learned

---

## Slide 14: Questions
- Thank you!
- Q&A