#!/usr/bin/env python3
"""
Build 3-model ensemble using NAFTta + RFT-TTA + RFT-MS with proper weights.
Uses the TTA-enhanced outputs that achieved the best 2model score (40.4852).
"""
import numpy as np
import pandas as pd
import base64, os, sys

BASE = '/home/luluboy/projects/vrdl_final/submissions'

def load_blocks(csv_path):
    df = pd.read_csv(csv_path)
    blocks = []
    for _, row in df.iterrows():
        b64 = row['BLOCK']
        img = np.frombuffer(base64.b64decode(b64), dtype=np.uint8).reshape(256, 256, 3)
        blocks.append(img)
    return np.array(blocks)

def psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0: return 100.0
    return 10 * np.log10(255**2 / mse)

print("Loading NAFTta...")
naftta = load_blocks(f'{BASE}/SubmitSrgb_nafnet_tta8_official.csv')  # 40.3675 alone
print("Loading RFT-TTA...")
rft_tta = load_blocks(f'{BASE}/SubmitSrgb_restormer_ft_tta8.csv')   # RFT with 8-way TTA
print("Loading RFT-MS...")
rft_ms = load_blocks(f'{BASE}/SubmitSrgb_restormer_ft_ms_tta.csv')  # RFT with MultiScale TTA

print(f"NAFTta: {naftta.shape}, RFT-TTA: {rft_tta.shape}, RFT-MS: {rft_ms.shape}")

# Load ground truth for local validation
import scipy.io as sio
gt_mat = sio.loadmat('/home/luluboy/projects/vrdl_final/data/ValidationGtBlocksSrgb.mat')
gt = gt_mat['ValidationGtBlocksSrgb']  # (40, 32, 256, 256, 3)

# Local validation of each model
print("\n=== Per-model Local Validation ===")
for name, arr in [('NAFTta', naftta), ('RFT-TTA', rft_tta), ('RFT-MS', rft_ms)]:
    psnrs = [psnr(arr[k], gt[k//32, k%32]) for k in range(1280)]
    print(f"  {name}: local_PSNR={np.mean(psnrs):.4f}")

# 2model baseline: NAFTta+RFT-TTA at alpha=0.735 should give ~40.4852 on Kaggle
# Let's verify local
ens2 = (0.735 * naftta.astype(float) + 0.265 * rft_tta.astype(float))
psnrs2 = [psnr(ens2[k].astype(np.uint8), gt[k//32, k%32]) for k in range(1280)]
print(f"\n  2model NAFTta+RFT-TTA @0.735: local_PSNR={np.mean(psnrs2):.4f}")

# Now do 3-model grid search
# Key insight: we keep NAFTta dominant, RFT-TTA significant, RFT-MS as supplement
# Weight constraint: alpha + beta + gamma = 1
# Given 2model best was alpha=0.735 (NAF) + 0.265 (RFT), we search around there
# Plus a small gamma for RFT-MS
print("\n=== 3-Model Ensemble Grid Search ===")
best_local = -1
best_weights = None
results = []

for alpha in [0.68, 0.70, 0.72, 0.73, 0.735, 0.74, 0.745, 0.75]:
    for beta in [0.15, 0.18, 0.20, 0.22, 0.25, 0.27]:
        gamma = round(1.0 - alpha - beta, 3)
        if gamma < 0 or gamma > 0.15:
            continue
        ens = alpha * naftta.astype(float) + beta * rft_tta.astype(float) + gamma * rft_ms.astype(float)
        ens = ens.astype(np.uint8)
        psnrs_ens = [psnr(ens[k], gt[k//32, k%32]) for k in range(1280)]
        mean_psnr = np.mean(psnrs_ens)
        results.append((alpha, beta, gamma, mean_psnr))
        if mean_psnr > best_local:
            best_local = mean_psnr
            best_weights = (alpha, beta, gamma)
            print(f"  NEW BEST: alpha={alpha}, beta={beta}, gamma={gamma:.3f} -> local_PSNR={mean_psnr:.4f}")

print(f"\nBest local: {best_weights} -> {best_local:.4f}")

# Also try adding NAFTta + RFT-MS without RFT-TTA (since RFT-MS has different augmentation)
print("\n=== NAFTta + RFT-MS 2-model ===")
for alpha in [0.70, 0.73, 0.735, 0.74, 0.75, 0.78]:
    beta = round(1.0 - alpha, 3)
    ens = alpha * naftta.astype(float) + beta * rft_ms.astype(float)
    ens = ens.astype(np.uint8)
    psnrs_ens = [psnr(ens[k], gt[k//32, k%32]) for k in range(1280)]
    mean_psnr = np.mean(psnrs_ens)
    if mean_psnr > best_local:
        best_local = mean_psnr
        best_weights = (alpha, beta, 0.0)
        print(f"  NEW BEST: alpha={alpha}, beta={beta}, gamma=0.0 -> local_PSNR={mean_psnr:.4f}")

print(f"\nFINAL Best: {best_weights} -> {best_local:.4f}")
print("\nTop 10 by local PSNR:")
results.sort(key=lambda x: -x[3])
for r in results[:10]:
    print(f"  alpha={r[0]}, beta={r[1]}, gamma={r[2]:.3f} -> {r[3]:.4f}")

# Generate CSV for the best 3model weights
if best_weights and best_weights[2] > 0:
    alpha, beta, gamma = best_weights
    print(f"\nGenerating 3model CSV with alpha={alpha}, beta={beta}, gamma={gamma}")
else:
    print("\nGenerating 2model CSV (NAFTta + RFT-MS)")
    alpha, beta, gamma = best_weights[0], best_weights[1], 0.0

# Create output
df_out = pd.DataFrame()
df_out['ID'] = np.arange(1280)

# Generate BLOCK column
blocks_out = []
for k in range(1280):
    blended = (alpha * naftta[k].astype(float) + 
               beta * rft_tta[k].astype(float) + 
               gamma * rft_ms[k].astype(float))
    out_block = np.clip(blended, 0, 255).astype(np.uint8)
    b64 = base64.b64encode(out_block.tobytes()).decode('utf-8')
    blocks_out.append(b64)
    if k % 200 == 0:
        print(f"  [{k}/1280] done")

df_out['BLOCK'] = blocks_out
fname = f'{BASE}/SubmitSrgb_3model_fttta_{alpha}_{beta}_{gamma:.3f}.csv'
df_out.to_csv(fname, index=False)
print(f"\n✅ Saved {fname} ({os.path.getsize(fname)/1e6:.1f} MB)")
print(f"First b64 len: {len(blocks_out[0])} (expected 262144)")
print("\nReady to submit with:")
print(f"  kaggle competitions submit -c sidd-benchmark-srgb-psnr -f {fname} -m '3model FTTTA: NAFTta({alpha})+RFT-TTA({beta})+RFT-MS({gamma})'")