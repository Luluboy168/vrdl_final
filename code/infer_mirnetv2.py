#!/usr/bin/env python3
"""
infer_mirnetv2.py
用 MIRNetv2 (real_denoising.pth) 對 1280 個 blocks 推論，產出 SubmitSrgb_mirnetv2.csv

流程：
  1. 讀取 BenchmarkNoisyBlocksSrgb.mat  (40,32,256,256,3) uint8
  2. 以 MIRNetv2 逐一推論（torch.no_grad + eval mode）
  3. 產出 base64 CSV（與 baseline CSV 格式一致）
"""
import os
import sys
import torch
import scipy.io
import numpy as np
import pandas as pd
import base64
import cv2

PROJECT_DIR  = '/home/luluboy/projects/vrdl_final'
CODE_DIR     = os.path.join(PROJECT_DIR, 'code', 'MIRNetv2')
DATA_DIR     = os.path.join(PROJECT_DIR, 'data')
WEIGHTS_DIR  = os.path.join(PROJECT_DIR, 'weights')
SUBS_DIR     = os.path.join(PROJECT_DIR, 'submissions')

MAT_FILE    = os.path.join(DATA_DIR, 'BenchmarkNoisyBlocksSrgb.mat')
WEIGHT_FILE = os.path.join(WEIGHTS_DIR, 'real_denoising.pth')
OUTPUT_CSV  = os.path.join(SUBS_DIR, 'SubmitSrgb_mirnetv2.csv')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'🚀 使用設備: {DEVICE}')

# ===== Helper functions =====
def img2tensor(img):
    """HWC [0,255] uint8 → CHW float32 [0,1]"""
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img

def tensor2img(tensor):
    """CHW float32 [0,1] → HWC uint8 [0,255]"""
    out = tensor.detach().cpu()
    out = out.clamp(0, 1).mul(255).round().byte()
    out = out.permute(1, 2, 0).numpy()
    return out

def array_to_base64(x):
    """numpy array (H,W,C) uint8 → base64 raw-bytes string（官方格式）"""
    return base64.b64encode(x.tobytes()).decode('utf-8')

# ===== Build MIRNetv2 model =====
def build_model():
    sys.path.insert(0, os.path.join(CODE_DIR, 'basicsr', 'models', 'archs'))
    from mirnet_v2_arch import MIRNet_v2

    model = MIRNet_v2(
        inp_channels=3,
        out_channels=3,
        n_feat=80,
        chan_factor=1.5,
        n_RRG=4,
        n_MRB=2,
        height=3,
        width=2,
        scale=1,
        bias=False,
        task='real_denoising',   # 讓模型執行 residual: out += inp
    )

    print(f'📦 載入權重: {WEIGHT_FILE}')
    ckpt = torch.load(WEIGHT_FILE, map_location='cpu')
    state_dict = ckpt['params'] if 'params' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    print('✅ MIRNetv2 權重載入成功（strict=True）')

    model.to(DEVICE)
    model.eval()
    return model

def main():
    # 1. Load .mat
    print(f'📂 讀取 {MAT_FILE} ...')
    mat = scipy.io.loadmat(MAT_FILE)
    keys = [k for k in mat.keys() if not k.startswith('__')]
    print(f'   .mat keys: {keys}')
    noisy_blocks = mat[keys[0]]  # (40, 32, 256, 256, 3), uint8
    print(f'   shape: {noisy_blocks.shape}, dtype: {noisy_blocks.dtype}')

    rows, cols = noisy_blocks.shape[0], noisy_blocks.shape[1]
    total = rows * cols
    print(f'   共 {rows}×{cols}={total} blocks')

    # 2. Build model
    model = build_model()

    # 3. Inference
    records = []
    print(f'🔄 開始 MIRNetv2 推論...')

    for i in range(rows):
        for j in range(cols):
            block_id = i * cols + j  # 0-based index

            # HWC [0,255] uint8
            img = noisy_blocks[i, j]  # (256, 256, 3)

            # to tensor [0,1], add batch dim
            img_t = img2tensor(img).unsqueeze(0).to(DEVICE)  # (1,3,256,256)

            with torch.no_grad():
                out_t = model(img_t)  # (1,3,256,256), float32 [0,1]

            # to uint8 [0,255]
            out_img = tensor2img(out_t[0])  # (256,256,3)
            out_img = np.clip(out_img, 0, 255).astype(np.uint8)

            b64 = array_to_base64(out_img)
            records.append({'ID': block_id, 'BLOCK': b64})

            if (block_id + 1) % 200 == 0 or (block_id + 1) == total:
                print(f'   [{block_id+1}/{total}] 完成  (last b64 len: {len(b64)})')

    # 4. Save CSV
    print(f'💾 寫入 {OUTPUT_CSV} ...')
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    size_mb = os.path.getsize(OUTPUT_CSV) / 1e6
    print(f'✅ 完成！CSV ({size_mb:.1f} MB), {len(records)} rows')

    return OUTPUT_CSV

if __name__ == '__main__':
    main()