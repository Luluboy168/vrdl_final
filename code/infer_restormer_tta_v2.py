#!/usr/bin/env python3
"""
Restormer + 8-way TTA inference - based on the WORKING baseline script.
Uses raw bytes base64 (not PNG) with correct column names ID/BLOCK.
"""
import os, sys, gc
import torch
import scipy.io
import numpy as np
import pandas as pd
import cv2
import base64

PROJECT_DIR = '/home/luluboy/projects/vrdl_final'
CODE_DIR     = os.path.join(PROJECT_DIR, 'code')
RESTORMER_DIR = os.path.join(CODE_DIR, 'Restormer')
DATA_DIR     = os.path.join(PROJECT_DIR, 'data')
WEIGHTS_DIR  = os.path.join(PROJECT_DIR, 'weights')
SUBS_DIR     = os.path.join(PROJECT_DIR, 'submissions')

MAT_FILE    = os.path.join(DATA_DIR, 'BenchmarkNoisyBlocksSrgb.mat')
WEIGHT_FILE = os.path.join(WEIGHTS_DIR, 'RealDenoising_Restormer.pth')
OUTPUT_CSV  = os.path.join(SUBS_DIR, 'SubmitSrgb_restormer_tta8.csv')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'🚀 使用設備: {DEVICE}')


# === TTA transforms ===
def rot90(img): return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
def rot180(img): return cv2.rotate(img, cv2.ROTATE_180)
def rot270(img): return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
def hflip(img): return cv2.flip(img, 1)
def inv_rot90(img): return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
def inv_rot180(img): return cv2.rotate(img, cv2.ROTATE_180)
def inv_rot270(img): return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

TTA_VARIANTS = [
    (lambda x: x,                              lambda x: x),
    (rot90,                                    inv_rot90),
    (rot180,                                   inv_rot180),
    (rot270,                                   inv_rot270),
    (hflip,                                    hflip),
    (lambda x: hflip(rot90(x)),                lambda x: inv_rot90(hflip(x))),
    (lambda x: hflip(rot180(x)),               lambda x: inv_rot180(hflip(x))),
    (lambda x: hflip(rot270(x)),               lambda x: inv_rot270(hflip(x))),
]


# === Helper functions (same as baseline) ===
def img2tensor(img, bgr2rgb=False, float32=True):
    """HWC [0,255] uint8 → CHW float32 [0,1]"""
    img = img.astype(np.float32) / 255.0
    if bgr2rgb:
        img = img[:, :, [2, 1, 0]]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img


def tensor2img(tensor):
    """CHW float32 [0,1] → HWC uint8 [0,255]"""
    out = tensor.detach().cpu()
    out = out.clamp(0, 1).mul(255).round().byte()
    out = out.permute(1, 2, 0).numpy()
    return out


def array_to_base64_raw(x):
    """numpy array (H,W,C) uint8 → base64 raw bytes string (Kaggle format)"""
    return base64.b64encode(x.tobytes()).decode('utf-8')


# === Build Restormer model (same as baseline) ===
def build_model():
    sys.path.insert(0, os.path.join(RESTORMER_DIR, 'basicsr/models/archs'))
    from restormer_arch import Restormer

    model = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='BiasFree',
        dual_pixel_task=False,
    )

    print(f'📦 載入權重: {WEIGHT_FILE}')
    ckpt = torch.load(WEIGHT_FILE, map_location='cpu')

    if 'params' in ckpt:
        state_dict = ckpt['params']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    print('✅ Restormer 權重載入成功（strict=True）')

    model.to(DEVICE)
    model.eval()
    return model


def main():
    # 1. Load .mat
    print(f'📂 讀取 {MAT_FILE} ...')
    mat = scipy.io.loadmat(MAT_FILE)
    keys = [k for k in mat.keys() if not k.startswith('__')]
    print(f'   .mat keys: {keys}')
    noisy_blocks = mat[keys[0]]
    print(f'   shape: {noisy_blocks.shape}, dtype: {noisy_blocks.dtype}')

    rows, cols = noisy_blocks.shape[0], noisy_blocks.shape[1]
    total = rows * cols
    print(f'   共 {rows}×{cols}={total} blocks')

    # 2. Build model
    model = build_model()
    torch.cuda.reset_peak_memory_stats()

    # 3. TTA inference
    records = []
    print(f'🔄 開始 Restormer + TTA 推論...')

    for i in range(rows):
        for j in range(cols):
            block_id = i * cols + j + 1  # 1-based

            # HWC [0,255] uint8
            block = noisy_blocks[i, j]  # (256, 256, 3)

            # 8-way TTA
            accum = np.zeros((256, 256, 3), dtype=np.float64)
            for fwd, inv in TTA_VARIANTS:
                xformed = fwd(block)
                img_t = img2tensor(xformed).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    out_t = model(img_t)

                out_np = tensor2img(out_t[0])  # (256,256,3), uint8
                restored = inv(out_np.astype(np.float32))
                accum += restored

                del img_t, out_t, out_np, xformed, restored

            avg = accum / len(TTA_VARIANTS)
            out_img = np.clip(avg, 0, 255).round().astype(np.uint8)

            b64 = array_to_base64_raw(out_img)
            records.append({'ID': block_id, 'BLOCK': b64})

            if block_id % 100 == 0 or block_id == total:
                print(f'   [{block_id}/{total}] 完成. Peak: {torch.cuda.max_memory_allocated()/1e9:.3f}GB')
                torch.cuda.empty_cache()

    # 4. Save CSV
    print(f'💾 寫入 {OUTPUT_CSV} ...')
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    size_mb = os.path.getsize(OUTPUT_CSV) / 1e6
    print(f'✅ 完成！CSV ({size_mb:.1f} MB), {len(records)} rows')
    print(f'   Peak GPU: {torch.cuda.max_memory_allocated()/1e9:.3f} GB')


if __name__ == '__main__':
    main()