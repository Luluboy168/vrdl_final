#!/usr/bin/env python3
"""Restormer + 8-way TTA inference — float32 version (no half/autocast)."""
import os, sys, gc
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
import torch
import scipy.io
import numpy as np
import pandas as pd
import cv2
import base64

DEVICE = torch.device('cuda')
print(f'[DEVICE] Using: {DEVICE}')

PROJECT_DIR   = '/home/luluboy/projects/vrdl_final'
CODE_DIR      = os.path.join(PROJECT_DIR, 'code')
RESTORMER_DIR = os.path.join(CODE_DIR, 'Restormer')
DATA_DIR      = os.path.join(PROJECT_DIR, 'data')
WEIGHTS_DIR   = os.path.join(PROJECT_DIR, 'weights')
SUBS_DIR      = os.path.join(PROJECT_DIR, 'submissions')

MAT_FILE    = os.path.join(DATA_DIR, 'BenchmarkNoisyBlocksSrgb.mat')
WEIGHT_FILE = os.path.join(WEIGHTS_DIR, 'RealDenoising_Restormer.pth')
OUTPUT_CSV  = os.path.join(SUBS_DIR, 'SubmitSrgb_restormer_tta8.csv')


def img2tensor(img):
    return torch.from_numpy(img.astype(np.float32)/255.0).permute(2,0,1).float()


def build_model():
    sys.path.insert(0, os.path.join(RESTORMER_DIR, 'basicsr/models/archs'))
    from restormer_arch import Restormer

    model = Restormer(
        inp_channels=3, out_channels=3, dim=48,
        num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type='BiasFree', dual_pixel_task=False,
    )

    print(f'Loading Restormer weights from: {WEIGHT_FILE}')
    ckpt = torch.load(WEIGHT_FILE, map_location='cpu')
    sd = ckpt.get('params', ckpt.get('state_dict', ckpt))
    model.load_state_dict(sd, strict=True)
    model.to(DEVICE)
    model.eval()
    print('Model loaded successfully (float32, no half)')
    return model


# TTA transforms
def rot90(img): return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
def rot180(img): return cv2.rotate(img, cv2.ROTATE_180)
def rot270(img): return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
def hflip(img): return cv2.flip(img, 1)
def inv_rot90(img): return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
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


@torch.no_grad()
def run_tta_block(model, block):
    accum = np.zeros((256, 256, 3), dtype=np.float64)
    for fwd, inv in TTA_VARIANTS:
        xformed = fwd(block)
        t = img2tensor(xformed).unsqueeze(0).to(DEVICE)
        out = model(t)  # plain float32, no autocast
        out_np = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_np = out_np * 255.0
        restored = inv(out_np.astype(np.float32))
        accum += restored
        del t, out, out_np, xformed
        torch.cuda.empty_cache()
    avg = accum / len(TTA_VARIANTS)
    return np.clip(avg, 0, 255).round().astype(np.uint8)


def main():
    print('='*60)
    print('Restormer + 8-way TTA Inference (float32)')
    print('='*60)

    if not os.path.exists(MAT_FILE):
        print(f'ERROR: {MAT_FILE} not found')
        sys.exit(1)

    model = build_model()

    # Sanity check
    print('Running sanity check...')
    test_block = np.random.randint(0, 255, (256,256,3), dtype=np.uint8)
    xformed = rot90(test_block)
    t = img2tensor(xformed).unsqueeze(0).to(DEVICE)
    out = model(t)
    print(f'  Sanity OK. Peak: {torch.cuda.max_memory_allocated()/1e9:.3f}GB')
    del t, out
    torch.cuda.empty_cache()

    # Load data
    print(f'Loading data from {MAT_FILE}...')
    mat = scipy.io.loadmat(MAT_FILE)
    keys = [k for k in mat.keys() if not k.startswith('__')]
    inputs = mat[keys[0]]
    n_i, n_j = inputs.shape[0], inputs.shape[1]
    n_blocks = n_i * n_j
    print(f'  Shape: {inputs.shape} -> {n_blocks} blocks')

    print(f'\nRunning TTA inference on {n_blocks} blocks...')

    results = []
    torch.cuda.reset_peak_memory_stats()
    for i in range(n_i):
        for j in range(n_j):
            block = inputs[i, j, :, :, :]
            denoised = run_tta_block(model, block)
            b64 = base64.b64encode(denoised.tobytes()).decode('utf-8')
            block_id = i * n_j + j
            results.append({'ID': block_id, 'BLOCK': b64})
            if (block_id + 1) % 200 == 0 or block_id + 1 == n_blocks:
                print(f'  [{block_id+1}/{n_blocks}] done. Peak: {torch.cuda.max_memory_allocated()/1e9:.3f}GB')

    os.makedirs(SUBS_DIR, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    size_mb = os.path.getsize(OUTPUT_CSV) / 1e6
    print(f'\nDone! Saved {OUTPUT_CSV} ({size_mb:.1f} MB, {len(df)} rows)')
    print(f'Peak GPU: {torch.cuda.max_memory_allocated()/1e9:.3f} GB')


if __name__ == '__main__':
    main()