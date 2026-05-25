# 階段 4：Fine-tune NAFNet 於 SIDD-Medium，預期 +0.10~0.20 dB

- [ ] 下載 SIDD-Medium sRGB 資料集（~12 GB）：https://www.eecs.yorku.ca/~kamel/sidd/dataset.php
- [ ] 設定 BasicSR 框架的 `train.py` + NAFNet SIDD config，**改為 fine-tune 模式**：
    - [ ] 從 `NAFNet-SIDD-width64.pth` 初始化（config 中 `pretrain_network_g`）
    - [ ] Learning rate `1e-5`（小 lr 防 catastrophic forgetting）
    - [ ] Iterations 20K-30K
    - [ ] Batch size **4 或 6**（3080 Ti 開 AMP）
    - [ ] 啟用 AMP：`use_amp: true`
    - [ ] Optimizer Adam，cosine annealing scheduler
    - [ ] Patch size 維持 256×256
- [ ] Dry-run 100 iters 確認 loss 下降才正式開跑
- [ ] 訓練完用新 weight 取代原 NAFNet 進 ensemble
- [ ] **保留原預訓練 weight 做 baseline**，若 fine-tuned 變差就 fallback
- [ ] 產出 `submissions/SubmitSrgb_ensemble_v2_finetuned.csv`
- [ ] 上傳 Kaggle，預期 **PSNR ≈ 40.50-40.65**
- [ ] 紀錄到 `submissions/log.csv`
