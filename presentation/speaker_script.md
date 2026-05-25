# VRDL Final Project — Speaker Script
# SIDD sRGB Image Denoising
# Presentation: 2026-06-09

> 每頁大約 30-45 秒。總長約 10-12 分鐘（留 3-5 分鐘給 Q&A）。

---

## Slide 1: Title (30s)
「大家好，我是 Chin Lu，來自 HCIS Lab，指導教授：陳怡婷。今天要分享的是這學期 VRDL 的 Final Project：Kaggle SIDD Benchmark 的影像去噪競賽。」

---

## Slide 2: Competition Overview (30s)
「這個競賽的任務是對智慧手機拍攝的雜訊影像進行去噪。輸入是 1280 塊 256×256 的 noisy blocks，輸出是乾淨的影像，評估指標是 PSNR，越高越好。比賽於 2025 年 5 月閉幕，但 Late Submission 仍開放，允許我們繼續提交並獲得分數。金牌門檻是 40.0019 dB。」

---

## Slide 3: Methods Tried (30s)
「我們先後嘗試了多種方法：NAFNet 作為 baseline、Restormer 作為 Transformer baseline、Restormer 加上 8-way TTA。我們也做 weighted ensemble，以及後來修復了 TTA bug 後的最終版本。其中有一個重要的教訓——NAFNet + TTA 因為 inverse transform bug，導致分數反而下降，這個問題後來被修正了。」

---

## Slide 4: Kaggle Submission Results (45s)
「這張表格列出所有 Kaggle 驗證過的分數——注意這裡每一個數字都是實際 Late Submission 的結果，不是估測值。
最佳結果是 **40.4852 dB**，使用 NAFNet-TTA 與 Restormer-FT-TTA8 的加權融合，α=0.745，比金牌門檻高出將近 0.48 dB。
值得注意的是，Restormer 的 fine-tuning 有小幅幫助——Restormer-FT-TTA8（40.1344）比 Restormer（40.0902）高約 0.04 dB。但相較於 NAFNet-TTA（40.41），Restormer 仍明顯較弱（40.09 vs 40.37）。另一個獨立的 Restormer fine-tuning 實驗（Restormer-FT-E5）在 5 個 epochs 後分數崩潰到 28.22——灾难性的失敗，原因懷疑是學習率或資料問題。」

---

## Slide 5: Key Findings (45s)
「我們學到了幾個重要教訓：
第一，NAFNet 在這個 benchmark 單獨使用就比 Restormer 強（40.37 vs 40.09）。
第二，TTA 在 NAFNet 上有 bug，會讓分數變差，千萬不能把兩個 TTA variant 放在一起 ensemble——那會讓分數直接掉到 39.82。
第三，3 模型的 ensemble 反而沒有 2 模型好，因為模型之間太過相關。
第四，fine-tuning 效果有限且有風險——Restormer-FT-TTA8（40.13）比 Restormer（40.09）有小幅提升，但另一個 Restormer fine-tuning 實驗（Restormer-FT-E5）因學習率或資料問題在 5 epochs 後崩潰到 28.22，說明 fine-tuning 在此任務上並不穩定。」

---

## Slide 6: NAFNet Architecture (45s)
「NAFNet 的核心創新是 SimpleGate——它把 activation function 拿掉，改成 LayerNorm 和乘法操作，大幅簡化架構。它的預訓練權重在 SIDD 資料集上表現本來就不錯，我們用他的原始權重，baseline 就已經有 40.37 dB。」

---

## Slide 7: Restormer Architecture (45s)
「Restormer 是 CVPR 2022 的論文，核心是 channel attention 和 multi-scale GFN blocks。它在 channel 維度應用 Transformer，計算效率比 spatial attention 高很多。原始 Restormer baseline 是 40.09 dB，加上 TTA 可以到 40.13 dB。」

---

## Slide 8: Test-Time Augmentation (TTA) (45s)
「TTA 是我們提升分數的關鍵技巧之一。8-way TTA 包括 4 個旋轉角度 × 2 種翻轉，共 8 種 augmentation，對 Restormer 有效，但對 NAFNet 有 bug——inverse transform 方向錯了，所以分數反而變差。這個 bug 後來修復了，但修復前我們用它做 ensemble 浪費了多次 submission 機會。」

---

## Slide 9: Ensemble Strategy (60s)
「最後的 ensemble 策略很簡單：α × NAFNet-TTA + (1-α) × Restormer-FT-TTA8。我們做了細密的 grid search，測了 α 從 0.730 到 0.755，結果非常有趣——最佳點是 α=0.745，但 α=0.746 就直接掉到 39.82，差了 0.66 dB。這說明最佳點非常窄，這也解釋了為什麼 α=0.70 沒有效果——它根本不在 peak 上。這個發現讓我們確認：精確的超參數搜索在這個任務非常重要。」

---

## Slide 10: Experimental Results — All Models (45s)
「這張表完整呈現所有嘗試過的方法。最左邊是我們的最佳結果，40.4852 dB，已經超過金牌門檻。第二名是 α=0.750 的 40.4612，差距其實很小，但剛好落在 peak 之外就差了很多。Restormer 的 MultiScale-TTA（40.04）不如 8-way TTA（40.13），MIRNetv2 單獨只有 39.91，而 CGNet 表現最差只有 33.96——這個模型架構完全不適合這個 benchmark。」

---

## Slide 11: Ablation Study — Key Findings (60s)
「這張投影片總結我們的 ablation study 六個關鍵發現。第一，NAFNet 單獨就比 Restormer 強。第二，fine-tuning 會導致灾难性的過擬合。第三，α 的最佳點非常窄——0.745 vs 0.746差了 0.66 dB，這非常不尋常。第四，TTA+TTA 的配對會摧毀分數。第五，3 模型 ensemble 反而比 2 模型差。第六，MIRNetv2 單獨 39.91，對 ensemble 沒有幫助。」

---

## Slide 12: Conclusions (45s)
「結論部分：我們成功建立了 NAFNet 與 Restormer 的推論流程，透過 weighted ensemble 達到 40.4852 PSNR，超越金牌門檻。重要的 lesson 包括：TTA 實作要非常小心inverse transform、加權 ensemble 比平均好、fine-tuning 在小資料集上可能完全失敗、以及 2 模型就已經是最好的組合，不需要更多模型。」

---

## Slide 13: Future Work (30s)
「未來方向：第一，修復 NAFNet TTA 的 bug，可能可以再提升 0.05-0.1 dB。第二，嘗試 multi-scale TTA（不同 scale 的輸入）。第三，考慮更強的 SOTA 模型如 KBNet 或 HAT。第四，重新下載 SIDD Medium sRGB 資料集，解決 fine-tuning 失敗的問題。」

---

## Slide 14: Questions (隨意)
「以上就是我們的報告，謝謝大家，歡迎提問。」

---

## 準備提醒
- [ ] Report 截止 06/07 23:59 — 需 Luluboy 建立 GitHub repo（vrdl_final）後補上連結至 Appendix C