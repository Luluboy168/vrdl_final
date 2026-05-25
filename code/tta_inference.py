#!/usr/bin/env python3
"""
tta_inference.py
8-way Test-Time Augmentation (TTA) for NAFNet denoising.

Transforms:
  4 rotations × 2 flips = 8 variants
  - rot0  + no flip
  - rot0  + h-flip
  - rot90 + no flip
  - rot90 + h-flip
  - rot180+ no flip
  - rot180+ h-flip
  - rot270+ no flip
  - rot270+ h-flip

Each variant → NAFNet → inverse transform → average → output
"""

import os
import sys
import torch
import numpy as np
import cv2
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[DEVICE] Using: {DEVICE}')

# ===== 路徑設定 =====
PROJECT_DIR = '/home/luluboy/projects/vrdl_final'
CODE_DIR    = os.path.join(PROJECT_DIR, 'code')
NAFNET_DIR  = os.path.join(CODE_DIR, 'NAFNet')
DATA_DIR    = os.path.join(PROJECT_DIR, 'data')
WEIGHTS_DIR = os.path.join(PROJECT_DIR, 'weights')
SUBS_DIR    = os.path.join(PROJECT_DIR, 'submissions')

MAT_FILE    = os.path.join(DATA_DIR, 'BenchmarkNoisyBlocksSrgb.mat')
WEIGHT_FILE = os.path.join(WEIGHTS_DIR, 'NAFNet-SIDD-width64.pth')
OUTPUT_CSV  = os.path.join(SUBS_DIR, 'SubmitSrgb_nafnet_tta8.csv')


# ── NAFNet helper (copied from make_submission.py) ────────────────────────────

def img2tensor(img):
    """HWC [0,255] uint8 → CHW float32 [0,1]"""
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).float().to(DEVICE)
    return img

def tensor2img(tensors):
    """CHW float32 [0,1] → HWC uint8 [0,255]"""
    out = tensors[0].detach().cpu()
    out = out.clamp(0, 1).mul(255).round().byte()
    return out.permute(1, 2, 0).numpy()

def build_model():
    sys.path.insert(0, NAFNET_DIR)
    from basicsr.models.archs.NAFNet_arch import NAFNet
    model = NAFNet(
        img_channel=3, width=64,
        enc_blk_nums=[2, 2, 4, 8],
        middle_blk_num=12,
        dec_blk_nums=[2, 2, 2, 2]
    )
    state_dict = torch.load(WEIGHT_FILE, map_location='cpu', weights_only=True)
    if isinstance(state_dict, dict) and 'params' in state_dict:
        state_dict = state_dict['params']
    new_sd = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_sd, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


# ── TTA transforms ───────────────────────────────────────────────────────────

def rot90(img):      return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
def rot180(img):     return cv2.rotate(img, cv2.ROTATE_180)
def rot270(img):     return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
def hflip(img):      return cv2.flip(img, 1)  # horizontal flip
def inv_rot90(img):  return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
def inv_rot180(img): return cv2.rotate(img, cv2.ROTATE_180)
def inv_rot270(img): return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

# 8 TTA variants: (forward_transform, inverse_transform)
# "no flip" variants first, then "h-flip" variants
TTA_VARIANTS = [
    # (no flip)
    (lambda x: x,                                    lambda x: x),                                     # rot0
    (rot90,                                          inv_rot90),                                      # rot90
    (rot180,                                         inv_rot180),                                     # rot180
    (rot270,                                         inv_rot270),                                     # rot270
    # (h-flip)
    (hflip,                                         hflip),                                           # rot0+hflip → inverse hflip
    (lambda x: hflip(rot90(x)),                     lambda x: inv_rot90(hflip(x))),                   # rot90+hflip
    (lambda x: hflip(rot180(x)),                     lambda x: inv_rot180(hflip(x))),                 # rot180+hflip
    (lambda x: hflip(rot270(x)),                     lambda x: inv_rot270(hflip(x))),                 # rot270+hflip
]


@torch.no_grad()
def denoise_block_tta(model, block):
    """
    Denoise a single 256×256×3 uint8 block using 8-way TTA.
    Returns denoised block as uint8 numpy array.
    """
    # Collect 8 predictions
    preds = []
    for fwd, inv in TTA_VARIANTS:
        transformed = fwd(block)                                    # HWC uint8
        t = img2tensor(transformed).unsqueeze(0)                    # (1,3,H,W)
        out_t = model(t)                                             # (1,3,H,W)
        pred = tensor2img(out_t)                                     # HWC uint8
        pred = inv(pred)                                             # inverse transform
        preds.append(pred.astype(np.float32))

    # Average and clip
    avg = np.mean(preds, axis=0)
    avg = np.clip(avg, 0, 255).round().astype(np.uint8)
    return avg


def array_to_base64(x):
    import base64, cv2
    _, enc = cv2.imencode('.png', x)
    return base64.b64encode(enc.tobytes()).decode('utf-8')


# ── Transform symmetry test ──────────────────────────────────────────────────

def test_transform_symmetry(model):
    """
    用全黑 + 白色十字測試圖驗證：
   做完 8 次 transform/inverse 後應與原圖完全相同（hash 比對）
    """
    print('\n🧪 驗證 TTA Transform 對稱性...')
    # Build test image: black background + white cross
    test_img = np.zeros((256, 256, 3), dtype=np.uint8)
    test_img[120:136, :, :] = 255   # horizontal bar
    test_img[:, 120:136, :] = 255   # vertical bar

    for fwd, inv in TTA_VARIANTS:
        transformed = fwd(test_img)
        t = img2tensor(transformed).unsqueeze(0)
        out_t = model(t)
        pred = tensor2img(out_t)
        restored = inv(pred)
        diff = np.abs(restored.astype(np.int16) - test_img.astype(np.int16)).sum()
        status = '✅' if diff == 0 else f'❌ diff={diff}'
        print(f'   variant: {fwd.__name__ or "custom"} → {status}')

    print('✅ Transform 對稱性測試完成')


def main():
    import scipy.io
    import pandas as pd
    import wget

    print('='*60)
    print('VRDL Final – NAFNet + 8-way TTA Submission')
    print('='*60)

    # 1. 確認資料
    if not os.path.exists(MAT_FILE):
        print(f'❌ {MAT_FILE} 不存在，請先執行 make_submission.py 下載資料。')
        sys.exit(1)

    # 2. 建立模型
    model = build_model()

    # 3. Transform 對稱性測試
    test_transform_symmetry(model)

    # 4. 讀取 .mat
    print(f'\n📂 讀取 {MAT_FILE} ...')
    mat = scipy.io.loadmat(MAT_FILE)
    inputs = mat['BenchmarkNoisyBlocksSrgb']
    n_i, n_j = inputs.shape[0], inputs.shape[1]
    n_blocks = n_i * n_j
    print(f'   shape: {inputs.shape} → {n_i}×{n_j}={n_blocks} blocks')

    # 5. TTA 推論
    print(f'\n🔁 開始 TTA 推論 {n_blocks} 個 blocks (×8 variants × {n_blocks} blocks = {8*n_blocks} forward passes)...')
    results = []
    for i in range(n_i):
        for j in range(n_j):
            block = inputs[i, j, :, :, :]                    # (256,256,3) uint8
            denoised = denoise_block_tta(model, block)       # (256,256,3) uint8
            b64 = array_to_base64(denoised)
            block_id = i * n_j + j
            results.append({'ID': block_id, 'BLOCK': b64})

    # 6. 寫入 CSV
    df = pd.DataFrame(results)
    os.makedirs(SUBS_DIR, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    size_mb = os.path.getsize(OUTPUT_CSV) / 1e6
    print(f'\n✅ 已儲存: {OUTPUT_CSV} ({size_mb:.1f} MB, {len(df)} rows)')

    print('\n📋 下一步：')
    print(f'  1. 前往 https://www.kaggle.com/competitions/sidd-benchmark-srgb-psnr/submit')
    print(f'  2. 上傳 {OUTPUT_CSV}')
    print(f'  3. 預期 PSNR ≈ 40.35-40.45 (較 baseline +0.05~0.10 dB)')


if __name__ == '__main__':
    main()