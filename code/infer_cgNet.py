#!/usr/bin/env python3
"""
infer_cgNet.py
CGNet (CascadedGaze) inference script for VRDL Final Project
SIDD sRGB Denoising Kaggle submission
"""
import os
import sys
import wget
import torch
import scipy.io
import numpy as np
import pandas as pd
import base64
import cv2
from tqdm import tqdm

# ===== 路徑設定 =====
PROJECT_DIR = '/home/luluboy/projects/vrdl_final'
CODE_DIR    = os.path.join(PROJECT_DIR, 'code')
CGNET_DIR   = os.path.join(PROJECT_DIR, 'CGNet_repo')
DATA_DIR    = os.path.join(PROJECT_DIR, 'data')
SUBS_DIR    = os.path.join(PROJECT_DIR, 'submissions')

MAT_URL   = 'http://130.63.97.225/share/SIDD_Blocks/BenchmarkNoisyBlocksSrgb.mat'
MAT_FILE  = os.path.join(DATA_DIR, 'BenchmarkNoisyBlocksSrgb.mat')
WEIGHT_FILE = os.path.join(CGNET_DIR, 'CGNet_SIDD.pth')
OUTPUT_CSV  = os.path.join(SUBS_DIR, 'SubmitSrgb_cgnet_tta.csv')

# ===== Helper functions =====
def img2tensor(img, bgr2rgb=False, float32=True):
    """HWC [0,255] uint8 → CHW float32 [0,1]"""
    img = img.astype(np.float32) / 255.0
    if bgr2rgb:
        img = img[:, :, [2, 1, 0]]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img

def tensor2img(tensors, rgb2bgr=False):
    """CHW float32 [0,1] → HWC uint8 [0,255]"""
    out = tensors[0].detach().cpu()
    out = out.clamp(0, 1).mul(255).round().byte()
    out = out.permute(1, 2, 0).numpy()
    if rgb2bgr:
        out = out[:, :, ::-1]
    return out

def array_to_base64(x):
    """numpy array (H,W,C) uint8 → base64 PNG string (Kaggle format)"""
    _, enc = cv2.imencode('.png', x)
    return base64.b64encode(enc.tobytes()).decode('utf-8')

def download_mat():
    if os.path.exists(MAT_FILE):
        size_mb = os.path.getsize(MAT_FILE) / 1e6
        print(f'✅ {MAT_FILE} 已存在 ({size_mb:.1f} MB)，略過下載。')
        return True
    print(f'📥 從 {MAT_URL} 下載 BenchmarkNoisyBlocksSrgb.mat ...')
    try:
        wget.download(MAT_URL, MAT_FILE)
        print('\n✅ 下載完成。')
        return True
    except Exception as e:
        print(f'❌ 下載失敗: {e}')
        print('請手動下載後放入 data/BenchmarkNoisyBlocksSrgb.mat')
        return False

def build_model():
    # Add CGNet repo to path so it can find basicsr package
    sys.path.insert(0, CGNET_DIR)

    from basicsr.models.archs.CGNet_arch import CascadedGaze

    model = CascadedGaze(
        img_channel=3,
        width=60,
        enc_blk_nums=[2, 2, 4, 6],
        middle_blk_num=10,
        dec_blk_nums=[2, 2, 2, 2],
        GCE_CONVS_nums=[3, 3, 2, 2]
    )

    print(f'📦 載入權重: {WEIGHT_FILE}')
    state_dict = torch.load(WEIGHT_FILE, map_location='cpu', weights_only=True)

    if isinstance(state_dict, dict) and 'params' in state_dict:
        state_dict = state_dict['params']

    # Remove 'module.' prefix if present (DDP training痕迹)
    new_sd = {}
    for k, v in state_dict.items():
        new_sd[k[7:] if k.startswith('module.') else k] = v

    # Load with strict=False to handle any minor structural differences
    model.load_state_dict(new_sd, strict=False)
    print('✅ CGNet 模型權重載入成功 (strict=False)')

    # Move to GPU and set eval mode
    model = model.cuda()
    model.eval()
    return model

@torch.no_grad()
def denoise_block(model, block):
    """推論單個 256×256×3 uint8 block"""
    # CGNet uses RGB (same as input), no BGR conversion needed
    img_t = img2tensor(block, bgr2rgb=False).unsqueeze(0)   # (1,3,256,256)
    img_t = img_t.cuda()
    out_t = model(img_t)                                     # (1,3,256,256)
    denoised = tensor2img(out_t, rgb2bgr=False)              # (256,256,3) uint8 RGB
    return denoised

def main():
    print('='*60)
    print('VRDL Final – CGNet (CascadedGaze) Submission')
    print('='*60)

    # 1. 下載資料
    if not download_mat():
        sys.exit(1)

    # 2. 建立模型
    model = build_model()

    # 3. 讀取 .mat
    print(f'\n📂 讀取 {MAT_FILE} ...')
    mat = scipy.io.loadmat(MAT_FILE)
    inputs = mat['BenchmarkNoisyBlocksSrgb']
    print(f'   shape: {inputs.shape}, dtype: {inputs.dtype}')
    n_i, n_j = inputs.shape[0], inputs.shape[1]
    n_blocks = n_i * n_j
    print(f'   共 {n_i}×{n_j}={n_blocks} blocks (預期 40×32=1280)')

    # 4. 推論迴圈
    print(f'\n🔁 開始推論 {n_blocks} 個 blocks ...')
    results = []
    for i in tqdm(range(n_i), desc='Row'):
        for j in range(n_j):
            block = inputs[i, j, :, :, :]          # (256,256,3) uint8
            denoised = denoise_block(model, block) # (256,256,3) uint8 RGB
            b64 = array_to_base64(denoised)
            block_id = i * n_j + j
            results.append({'ID': block_id, 'BLOCK': b64})

    # 5. 寫入 CSV
    df = pd.DataFrame(results)
    os.makedirs(SUBS_DIR, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f'\n✅ 已儲存: {OUTPUT_CSV}')
    print(f'   列數: {len(df)}, 檔案大小: {os.path.getsize(OUTPUT_CSV)/1e6:.1f} MB')

    # 6. 驗證格式
    print(f'\n📋 格式驗證:')
    print(f'   ID 範圍: {df["ID"].min()} ~ {df["ID"].max()}')
    print(f'   總列數: {len(df)}')
    sample_b64 = df.iloc[0]['BLOCK']
    print(f'   首個 BLOCK base64 長度: {len(sample_b64)}')

    print('\n📋 下一步：')
    print(f'  1. 前往 https://www.kaggle.com/competitions/sidd-benchmark-srgb-psnr/submit')
    print(f'  2. 上傳 {OUTPUT_CSV}')

if __name__ == '__main__':
    main()