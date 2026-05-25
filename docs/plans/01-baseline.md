# 階段 1：第一個 Submission（保底金牌 PSNR ≈ 40.30）

- [ ] `git clone https://github.com/megvii-research/NAFNet code/NAFNet`
- [ ] 下載 `NAFNet-SIDD-width64.pth` 預訓練權重（連結在 https://github.com/megvii-research/NAFNet/blob/main/docs/SIDD.md ）到 `weights/`
- [ ] 跑通官方 `basicsr/test.py` 對單張 SIDD 範例圖推論，驗證模型載入成功
- [ ] 參考 https://github.com/AbdoKamel/sidd_kaggle_submit/blob/main/prepare_submission_srgb.py 撰寫推論 + submission script，存於 `code/make_submission.py`
- [ ] 對 `BenchmarkNoisyBlocksSrgb.mat`(shape ~40×32×256×256×3, uint8) 的 1280 個 blocks 逐一推論
- [ ] 將每個 block 結果 base64 編碼，產出 `submissions/SubmitSrgb_baseline.csv`(columns: `ID, BLOCK`)
- [ ] **Sanity check**：先用直接 copy noisy → CSV 上傳，分數應 ≈ 36-38 PSNR，確認 block 順序正確後再丟 model output
- [ ] 上傳 Kaggle，預期 PSNR ≈ 40.30 → 金牌 ✓
- [ ] 紀錄 baseline PSNR 與 leaderboard 排名於 `submissions/log.csv`
