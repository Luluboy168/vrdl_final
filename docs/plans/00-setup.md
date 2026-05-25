# 階段 0：環境準備

- [ ] 從 https://www.kaggle.com/competitions/sidd-benchmark-srgb-psnr/data 下載測試集 `BenchmarkNoisyBlocksSrgb.mat`(~400 MB)，放到 `data/`
- [ ] 建立 conda / venv 環境：Python 3.10+、PyTorch 2.x、CUDA 對應版本
- [ ] 安裝套件：`pip install torch torchvision basicsr einops timm scipy h5py numpy opencv-python tqdm pandas`
- [ ] 驗證 GPU：`python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
- [ ] 驗證 AMP 可用：`python -c "from torch.cuda.amp import autocast, GradScaler; print('OK')"`
- [ ] 建立 working directory 結構：`mkdir -p data weights code submissions logs`
- [ ] 初始化 `submissions/log.csv`，欄位：`timestamp, commit, config, expected_psnr, actual_psnr, rank, note`
