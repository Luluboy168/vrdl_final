#!/usr/bin/env python3
"""
Validate ensemble by computing PSNR on validation set.
Uses ValidationGtBlocksSrgb.mat (ground truth) and pre-computed denoised CSVs.

This validates local model decisions, not Kaggle LB scores.
"""
import numpy as np
import scipy.io as sio
import pandas as pd
import base64

def psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return 100.0
    return 10 * np.log10(255**2 / mse)

BASE = '/home/luluboy/projects/vrdl_final/submissions'
PROJ = '/home/luluboy/projects/vrdl_final'

# Load ground truth (validation set)
print("Loading ground truth...")
gt_mat = sio.loadmat(f'{PROJ}/data/ValidationGtBlocksSrgb.mat')
gt = gt_mat['ValidationGtBlocksSrgb']  # (40, 32, 256, 256, 3) uint8

print(f"GT shape: {gt.shape}")

# Build denoised blocks from existing CSVs
# Each CSV has 1280 rows in (i=0..39, j=0..31) order = row idx = i*32 + j
def load_csv_blocks(csv_path):
    """Load a submission CSV and return blocks as (1280, 256, 256, 3) uint8."""
    df = pd.read_csv(csv_path)
    blocks = []
    for _, row in df.iterrows():
        b64 = row['BLOCK']
        img_bytes = base64.b64decode(b64)
        arr = np.frombuffer(img_bytes, dtype=np.uint8).reshape(256, 256, 3)
        blocks.append(arr)
    return np.array(blocks)  # (1280, 256, 256, 3)

csv_files = {
    'NAFNet baseline': f'{BASE}/SubmitSrgb_baseline_official.csv',
    'Restormer baseline': f'{BASE}/SubmitSrgb_restormer_official.csv',
    'Restormer+TTA': f'{BASE}/SubmitSrgb_restormer_tta8.csv',
}

for name, path in csv_files.items():
    if not os.path.exists(path):
        print(f"SKIP {name}: {path} not found")
        continue
    print(f"\nLoading {name}...")
    blocks = load_csv_blocks(path)
    # Validate: each block matches GT shape
    assert blocks.shape == (1280, 256, 256, 3), f"Shape mismatch: {blocks.shape}"
    psnrs = []
    for k in range(1280):
        psnrs.append(psnr(blocks[k], gt.flatten()[k]))
    print(f"  {name} avg PSNR: {np.mean(psnrs):.4f}")