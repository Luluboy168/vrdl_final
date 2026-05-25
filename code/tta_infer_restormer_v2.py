#!/usr/bin/env python3
"""
tta_infer_restormer_v2.py - Restormer + 8-way TTA with model-reload per variant.
Strategy: For each block, run 8 TTA variants by reloading model between variants.
This avoids any memory accumulation from holding the model + 8 intermediate tensors.
"""

import os, sys, gc
import torch
import scipy.io
import numpy as np
import pandas as pd
import cv2
import base64

torch.backends.cudnn.benchmark = True

PROJECT_DIR   = '/home/luluboy/projects/vrdl_final'
CODE_DIR       = os.path.join(PROJECT_DIR, 'code')
RESTORMER_DIR  = os.path.join(CODE_DIR, 'Restormer')
DATA_DIR       = os.path.join(PROJECT_DIR, 'data')
WEIGHTS_DIR    = os.path.join(PROJECT_DIR, 'weights')
SUBS_DIR       = os.path.join(PROJECT_DIR, 'submissions')

MAT_FILE      = os.path.join(DATA_DIR, 'BenchmarkNoisyBlocksSrgb.mat')
WEIGHT_FILE   = os.path.join(WEIGHTS_DIR, 'RealDenoising_Restormer.pth')
OUTPUT_CSV    = os.path.join(SUBS_DIR, 'SubmitSrgb_restormer_tta8.csv')


# TTA transforms
def rot90(img):      return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
def rot180(img):     return cv2.rotate(img, cv2.ROTATE_180)
def rot270(img):     return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
def hflip(img):      return cv2.flip(img, 1)
def inv_rot90(img):  return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
def inv_rot180(img): return cv2.rotate(img, cv2.ROTATE_180)
def inv_rot270(img): return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

TTA_VARIANTS = [
    (lambda x: x,                                   lambda x: x),
    (rot90,                                         inv_rot90),
    (rot180,                                        inv_rot180),
    (rot270,                                        inv_rot270),
    (hflip,                                         hflip),
    (lambda x: hflip(rot90(x)),                     lambda x: inv_rot90(hflip(x))),
    (lambda x: hflip(rot180(x)),                    lambda x: inv_rot180(hflip(x))),
    (lambda x: hflip(rot270(x)),                    lambda x: inv_rot270(hflip(x))),
]


def img2tensor(img):
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img


def build_model():
    """Load model, move to GPU, return model in inference mode."""
    sys.path.insert(0, os.path.join(RESTORMER_DIR, 'basicsr/models/archs'))
    from restormer_arch import Restormer

    model = Restormer(
        inp_channels=3, out_channels=3, dim=48,
        num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type='BiasFree', dual_pixel_task=False,
    )

    ckpt = torch.load(WEIGHT_FILE, map_location='cpu')
    if 'params' in ckpt:
        state_dict = ckpt['params']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    model.to('cuda')
    model.eval()
    return model


@torch.no_grad()
def run_single_variant(model, block, fwd, inv):
    """Run one TTA variant: transform → model → inverse transform."""
    transformed = fwd(block)
    t = img2tensor(transformed).unsqueeze(0).cuda()
    out = model(t)[0]
    out = out.permute(1, 2, 0).cpu().numpy()  # (256,256,3) float32 [0,1]
    out = out * 255.0
    restored = inv(out.astype(np.float32))
    del t, out, transformed
    return restored


def denoise_block_tta(block):
    """Denoise using 8-way TTA, reloading model for each variant to avoid OOM."""
    accumulator = np.zeros((256, 256, 3), dtype=np.float64)
    
    for fwd, inv in TTA_VARIANTS:
        # Reload model each time to ensure clean memory state
        model = build_model()
        pred = run_single_variant(model, block, fwd, inv)
        accumulator += pred
        # Aggressive cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    avg = accumulator / len(TTA_VARIANTS)
    avg = np.clip(avg, 0, 255).round().astype(np.uint8)
    return avg


def test_transform_symmetry():
    """Quick sanity check with one model load."""
    model = build_model()
    test_img = np.zeros((256, 256, 3), dtype=np.uint8)
    test_img[120:136, :, :] = 255
    test_img[:, 120:136, :] = 255

    print('  Testing TTA transform symmetry (variant 0 only)...')
    fwd, inv = TTA_VARIANTS[0]
    transformed = fwd(test_img)
    t = img2tensor(transformed).unsqueeze(0).cuda()
    out = model(t)[0]
    out = out.permute(1, 2, 0).cpu().numpy()
    out = out * 255.0
    restored = inv(out.astype(np.float32))
    diff = np.abs(restored.astype(np.int16) - test_img.astype(np.int16)).sum()
    status = '✅' if diff == 0 else f'❌ diff={diff}'
    print(f'    variant 0: {status}')
    del model, t, out
    gc.collect()
    torch.cuda.empty_cache()
    print('  ✅ Symmetry test done')


def main():
    print('='*60)
    print('VRDL Final – Restormer + 8-way TTA (model-reload v2)')
    print('='*60)

    if not os.path.exists(MAT_FILE):
        print(f'❌ {MAT_FILE} 不存在')
        sys.exit(1)

    # Quick sanity test
    test_transform_symmetry()

    # Load .mat
    print(f'\n📂 讀取 {MAT_FILE} ...')
    mat = scipy.io.loadmat(MAT_FILE)
    keys = [k for k in mat.keys() if not k.startswith('__')]
    inputs = mat[keys[0]]
    n_i, n_j = inputs.shape[0], inputs.shape[1]
    n_blocks = n_i * n_j
    print(f'   shape: {inputs.shape} → {n_i}×{n_j}={n_blocks} blocks')

    # TTA inference
    print(f'\n🔁 開始 Restormer TTA 推論 ({n_blocks} blocks × 8 variants)...')
    results = []
    for i in range(n_i):
        for j in range(n_j):
            block = inputs[i, j, :, :, :]
            denoised = denoise_block_tta(block)
            b64 = base64.b64encode(denoised.tobytes()).decode('utf-8')
            block_id = i * n_j + j
            results.append({'ID': block_id, 'BLOCK': b64})

            if (block_id + 1) % 50 == 0 or (block_id + 1) == n_blocks:
                print(f'   [{block_id+1}/{n_blocks}] blocks done')

    # Save CSV
    os.makedirs(SUBS_DIR, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    size_mb = os.path.getsize(OUTPUT_CSV) / 1e6
    print(f'\n✅ 已儲存: {OUTPUT_CSV} ({size_mb:.1f} MB, {len(df)} rows)')


if __name__ == '__main__':
    main()
