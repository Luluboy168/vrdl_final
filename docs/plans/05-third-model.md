# 階段 5：加入第三模型，預期 +0.05~0.10 dB

- [ ] 從以下挑 1-2 個試（按推薦順序）：
    - [ ] **MIRNet-v2**：https://github.com/swz30/MIRNetv2 （SIDD 預訓練，PSNR ~39.84）
    - [ ] **Uformer**：https://github.com/ZhendongWang6/Uformer （SIDD 預訓練）
    - [ ] **DRCT**（NTIRE 2024）：https://github.com/ming053l/DRCT
    - [ ] **MambaIR**：https://github.com/csguoh/MambaIR
- [ ] 每個 model 獨立跑 inference + TTA，比較單模型 PSNR
- [ ] 把表現最好的加入既有 NAFNet+Restormer ensemble
- [ ] 重新 grid search 三 model 權重（起手：`α=0.5, β=0.3, γ=0.2`）
- [ ] 產出 `submissions/SubmitSrgb_ensemble_v3.csv`
- [ ] 上傳 Kaggle，預期 **PSNR ≈ 40.55-40.70**
- [ ] 紀錄到 `submissions/log.csv`
