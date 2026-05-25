# 階段 3：加入 Restormer Ensemble，預期 +0.10~0.15 dB

- [ ] `git clone https://github.com/swz30/Restormer code/Restormer`
- [ ] 下載 Restormer SIDD denoising 預訓練（連結在 https://github.com/swz30/Restormer/blob/main/Denoising/README.md ）
- [ ] 寫 Restormer 推論 script `code/infer_restormer.py`，獨立產出 submission，預期 **PSNR ≈ 40.02**
- [ ] 對 Restormer 套用 ×8 TTA
- [ ] 實作 weighted average ensemble：`final = α × NAFNet_TTA + (1-α) × Restormer_TTA`
- [ ] Grid search α ∈ {0.3, 0.4, 0.5, 0.6, 0.7}
- [ ] **格式注意**：兩個 model 輸出在 ensemble 前 clamp 到 [0, 255] uint8
- [ ] 產出 `submissions/SubmitSrgb_ensemble_v1.csv`
- [ ] 用最佳 α 上傳，預期 **PSNR ≈ 40.40-40.55（top-3 邊緣）**
- [ ] 紀錄到 `submissions/log.csv`
