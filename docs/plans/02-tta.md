# 階段 2：×8 Self-Ensemble (TTA)，預期 +0.05~0.10 dB

- [ ] 實作 8-way TTA：4 個 rotation (0°/90°/180°/270°) × 2 個 flip (none / horizontal)
- [ ] 每個變體分別過 NAFNet 推論，輸出 inverse-transform 回原方向
- [ ] 8 個輸出平均作為最終結果
- [ ] **驗證 transform 對稱性**：用全黑 + 白色十字測試圖，做完 8 次 transform/inverse 後應與原圖完全相同（hash 比對）
- [ ] 產出 `submissions/SubmitSrgb_nafnet_tta8.csv`
- [ ] 上傳 Kaggle，預期 **PSNR ≈ 40.35-40.45**
- [ ] 紀錄分數到 `submissions/log.csv`
