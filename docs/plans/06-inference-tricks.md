# 階段 6：Inference Tricks，預期 +0.02~0.05 dB

- [ ] **Patch Overlap Inference**：block 邊緣 overlap 推論後加權平均（解決 patch 邊界 artifact）
    - [ ] 推論時對 256×256 block 上下左右 padding 32px，推論後裁回中央
    - [ ] 或對相鄰 block 做 50% overlap 平均
- [ ] **Multi-scale TTA**：input ±10% 縮放後推論，resize 回原尺寸平均
- [ ] **Noise level estimation**（可選）：估計 block 雜訊強度（用 wavelet 或 patch variance），依雜訊大小動態調 model 權重
- [ ] 每個 trick 獨立 ablation，**只有 PSNR 真進步才保留**
- [ ] 產出 `submissions/SubmitSrgb_ensemble_v4_tricks.csv`
- [ ] 紀錄每個 trick 的 PSNR delta 到 `submissions/log.csv`
