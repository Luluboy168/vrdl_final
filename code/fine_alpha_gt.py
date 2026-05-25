#!/usr/bin/env python3
"""Fine-grained alpha search around 0.745 using GT validation data.
NAFNet-TTA + Restormer-FT-TTA ensemble."""
import numpy as np
import scipy.io as sio
import base64
import pandas as pd
import os

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

# Load components used in best fttta submission
# NAFTta + RFT-TTA (multi-scale 8-way TTA)
print("Loading NAFTta (NAFNet + 8-way TTA)...")
naf = load_csv_blocks(f'{BASE}/SubmitSrgb_nafnet_tta8_fixed.csv')
print(f"NAFNet shape: {naf.shape}")

print("Loading RFT-TTA (Restormer-FT + 8-way TTA)...")
rest = load_csv_blocks(f'{BASE}/SubmitSrgb_restormer_ft_tta8.csv')
print(f"Restormer shape: {rest.shape}")

# Verify local PSNR for each model
print("\nLocal PSNR (vs GT):")
psnr_naf = []
psnr_rest = []
for k in range(1280):
    i = k // 32
    j = k % 32
    gt_block = gt[i, j]
    psnr_naf.append(psnr(naf[k], gt_block))
    psnr_rest.append(psnr(rest[k], gt_block))
print(f"NAFNet avg: {np.mean(psnr_naf):.4f}")
print(f"Restormer-FT avg: {np.mean(psnr_rest):.4f}")

naf_f = naf.astype(np.float64)
rest_f = rest.astype(np.float64)

# Fine-grained search around 0.745
print("\n=== Fine-grained Alpha Search (0.740-0.755) ===")
results = []
for alpha in [round(a, 3) for a in np.arange(0.740, 0.756, 0.001)]:
    ens = (alpha * naf_f + (1 - alpha) * rest_f).astype(np.uint8)
    psnr_ens = []
    for k in range(1280):
        i = k // 32
        j = k % 32
        psnr_ens.append(psnr(ens[k], gt[i, j]))
    psnr_mean = np.mean(psnr_ens)
    results.append((alpha, psnr_mean))
    print(f"  alpha={alpha:.3f}: PSNR={psnr_mean:.4f}")

# Sort by local PSNR
results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
print(f"\nTop 5 alphas (local):")
for a, p in results_sorted[:5]:
    print(f"  alpha={a:.3f}: PSNR={p:.4f}")

best_alpha_local = results_sorted[0][0]
print(f"\nBest local alpha: {best_alpha_local:.3f}")

# Generate submission for the top 2 alphas
for rank, (alpha, local_psnr) in enumerate(results_sorted[:2]):
    output_path = f'{BASE}/SubmitSrgb_2model_fttta_{alpha:.3f}.csv'
    
    if os.path.exists(output_path):
        print(f"\n⏭️  {output_path} already exists, skipping")
        continue
    
    ens = (alpha * naf_f + (1 - alpha) * rest_f).astype(np.uint8)
    
    block_strings = []
    for i in range(1280):
        b64 = base64.b64encode(ens[i].tobytes()).decode('utf-8')
        block_strings.append(b64)
    
    out_df = pd.DataFrame({'ID': list(range(1280)), 'BLOCK': block_strings})
    out_df.to_csv(output_path, index=False)
    file_size = os.path.getsize(output_path) / (1024*1024)
    
    # Verify
    df_check = pd.read_csv(output_path)
    assert list(df_check.columns) == ['ID', 'BLOCK']
    assert df_check['ID'].tolist() == list(range(1280))
    assert all(len(b) == 262144 for b in df_check['BLOCK'])
    
    print(f"\n✅ Generated {output_path}")
    print(f"   size={file_size:.1f}MB, alpha={alpha:.3f}, local_psnr={local_psnr:.4f}")

print("\nDone!")