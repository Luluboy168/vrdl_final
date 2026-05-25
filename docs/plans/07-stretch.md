# 階段 7：衝刺 50 pts（PSNR ≥ 41.56，stretch goal）

> 此階段非必要，達標即滿分 (50 pts)。

- [ ] 嘗試 **NAFNet-width96** 變體 fine-tune
    - [ ] 3080 Ti 必須啟用 gradient checkpointing（`torch.utils.checkpoint`）
    - [ ] Batch size 降到 2
- [ ] **Self-distillation**：用最佳 ensemble 結果當 pseudo-GT，re-train 單模型逼近
    - [ ] 把 ensemble output 存成「pseudo clean image」
    - [ ] 用 (noisy, pseudo_clean) 配對再 fine-tune NAFNet
- [ ] **Test-time finetuning**：對每張測試圖用 self-supervised loss（Noise2Noise / blind-spot network）微調幾步
- [ ] **架構融合**：嘗試 NAFNet + Restormer 結構在中間層 feature concat / cross-attend
- [ ] **NAFNet-baseline-width64**（與 NAFNet 平行的 baseline 變體）：嘗試取代或加入 ensemble
- [ ] 產出 `submissions/SubmitSrgb_stretch.csv`
- [ ] 紀錄到 `submissions/log.csv`
