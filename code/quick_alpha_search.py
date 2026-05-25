#!/usr/bin/env python3
"""Quick alpha search using existing CSVs - no model inference needed."""
import numpy as np
import scipy.io as sio
import base64
import pandas as pd

def psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return 100.0
    return 10 * np.log10(255**2 / mse)

BASE = '/home/luluboy/projects/vrdl_final/submissions'
PROJ = '/home/luluboy/projects/vrdl_final'

print("Loading ground truth...")
gt_mat = sio.loadmat(f'{PROJ}/data/ValidationGtBlocksSrgb.mat')
gt = gt_mat['ValidationGtBlocksSrgb']  # (40, 32, 256, 256, 3)
print(f"GT shape: {gt.shape}")

def load_csv_blocks(csv_path):
    df = pd.read_csv(csv_path)
    blocks = []
    for _, row in df.iterrows():
        b64 = row['BLOCK']
        img_bytes = base64.b64decode(b64)
        arr = np.frombuffer(img_bytes, dtype=np.uint8).reshape(256, 256, 3)
        blocks.append(arr)
    return np.array(blocks)

# Load the two main models
print("Loading NAFNet baseline...")
naf = load_csv_blocks(f'{BASE}/SubmitSrgb_baseline_official.csv')
print(f"NAFNet shape: {naf.shape}")

print("Loading Restormer+TTA...")
rest = load_csv_blocks(f'{BASE}/SubmitSrgb_restormer_tta8.csv')
print(f"Restormer shape: {rest.shape}")

# Compute per-block PSNR for each model
# gt ordering: gt[i, j] = block i,j (i=0..39 row, j=0..31 col)
# CSV ordering: row k = i*32 + j
print("\nComputing per-block PSNR for each model...")
psnr_naf = []
psnr_rest = []
for k in range(1280):
    i = k // 32
    j = k % 32
    gt_block = gt[i, j]  # (256, 256, 3)
    psnr_naf.append(psnr(naf[k], gt_block))
    psnr_rest.append(psnr(rest[k], gt_block))
psnr_naf = np.array(psnr_naf)
psnr_rest = np.array(psnr_rest)

print(f"NAFNet avg PSNR: {psnr_naf.mean():.4f}")
print(f"Restormer avg PSNR: {psnr_rest.mean():.4f}")

# Ensemble at pixel level
# Build per-block pixel arrays for fast computation
naf_f = naf.astype(np.float64)
rest_f = rest.astype(np.float64)

print("\n=== Alpha Grid Search (fine-grained) ===")
best_alpha = 0.7
best_psnr = -1

for alpha in [0.50, 0.55, 0.60, 0.62, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69,
              0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.78, 0.80]:
    # Ensemble at pixel level
    ens = (alpha * naf_f + (1 - alpha) * rest_f).astype(np.uint8)
    psnr_ens = []
    for k in range(1280):
        i = k // 32
        j = k % 32
        psnr_ens.append(psnr(ens[k], gt[i, j]))
    psnr_ens_mean = np.mean(psnr_ens)
    marker = " <-- BEST" if psnr_ens_mean > best_psnr else ""
    print(f"  alpha={alpha:.2f}: PSNR={psnr_ens_mean:.4f}{marker}")
    if psnr_ens_mean > best_psnr:
        best_psnr = psnr_ens_mean
        best_alpha = alpha

print(f"\nBest alpha={best_alpha:.2f} with local PSNR={best_psnr:.4f}")
print(f"NOTE: Kaggle may differ. Current best Kaggle: 40.4874 (alpha=0.70)")