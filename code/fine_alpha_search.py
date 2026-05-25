#!/usr/bin/env python3
"""
Fine-grained ensemble weight search: NAFNet-TTA + Restormer-FT
Uses existing official-format CSV files as components.
"""
import pandas as pd
import numpy as np
import base64, os, sys

BASE_DIR = "/home/luluboy/projects/vrdl_final/submissions"

# Load component CSVs
naf_df = pd.read_csv(f"{BASE_DIR}/SubmitSrgb_nafnet_tta8_fixed.csv")
rest_df = pd.read_csv(f"{BASE_DIR}/SubmitSrgb_restormer_ft.csv")

# Verify format
assert list(naf_df.columns) == ['ID', 'BLOCK'], f"NAF CSV columns wrong: {naf_df.columns.tolist()}"
assert list(rest_df.columns) == ['ID', 'BLOCK'], f"REST CSV columns wrong: {rest_df.columns.tolist()}"
assert naf_df['ID'].tolist() == list(range(1280)), f"NAF IDs wrong: {naf_df['ID'].min()}-{naf_df['ID'].max()}"
assert rest_df['ID'].tolist() == list(range(1280)), f"REST IDs wrong: {rest_df['ID'].min()}-{rest_df['ID'].max()}"
print(f"Loaded: NAFNet-TTA {len(naf_df)} rows, Restormer-FT {len(rest_df)} rows")

# Verify block sizes
naf0_len = len(naf_df.loc[0, 'BLOCK'])
rest0_len = len(rest_df.loc[0, 'BLOCK'])
print(f"Block sizes — NAF: {naf0_len}, REST: {rest0_len} (expected 262144)")
assert naf0_len == 262144, f"NAF block size wrong: {naf0_len}"
assert rest0_len == 262144, f"REST block size wrong: {rest0_len}"

# Decode to numpy arrays
print("Decoding NAFNet blocks...")
naf_arr = np.empty((1280, 256, 256, 3), dtype=np.uint8)
for i in range(1280):
    raw = base64.b64decode(naf_df.loc[i, 'BLOCK'])
    naf_arr[i] = np.frombuffer(raw, dtype=np.uint8).reshape(256, 256, 3)

print("Decoding Restormer-FT blocks...")
rest_arr = np.empty((1280, 256, 256, 3), dtype=np.uint8)
for i in range(1280):
    raw = base64.b64decode(rest_df.loc[i, 'BLOCK'])
    rest_arr[i] = np.frombuffer(raw, dtype=np.uint8).reshape(256, 256, 3)

print(f"NAF shape: {naf_arr.shape}, REST shape: {rest_arr.shape}")
print(f"NAF dtype: {naf_arr.dtype}, REST dtype: {rest_arr.dtype}")

# Test alphas from 0.70 to 0.78 in steps of 0.005
best_alpha = None
best_psnr_local = -np.inf

alphas_to_test = [round(a, 3) for a in np.arange(0.70, 0.79, 0.005)]
print(f"\nTesting {len(alphas_to_test)} alphas: {alphas_to_test[:5]} ... {alphas_to_test[-3:]}")

results = []

for alpha in alphas_to_test:
    # Weighted average in float
    blended = (alpha * naf_arr.astype(np.float64) + (1 - alpha) * rest_arr.astype(np.float64))
    blended = np.clip(np.round(blended), 0, 255).astype(np.uint8)
    
    # Compute local PSNR (blended vs NAFNet as reference approximation)
    # Actually compute PSNR of each block relative to NAFNet
    mse = np.mean((blended.astype(np.float64) - naf_arr.astype(np.float64)) ** 2)
    psnr = 10 * np.log10(255**2 / (mse + 1e-10))
    
    results.append((alpha, psnr, blended))
    
    if psnr > best_psnr_local:
        best_psnr_local = psnr
        best_alpha = alpha
    
    print(f"  alpha={alpha:.3f}  local_psnr_vs_NAF={psnr:.4f}")

print(f"\nBest alpha: {best_alpha:.3f} (local PSNR: {best_psnr_local:.4f})")

# Sort by local PSNR
results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
print("\nTop 5 alphas (by local PSNR vs NAFNet):")
for a, p, _ in results_sorted[:5]:
    print(f"  alpha={a:.3f}  {p:.4f}")

# Generate top 2 alphas as Kaggle-ready CSVs
for rank, (alpha, local_psnr, blended) in enumerate(results_sorted[:2]):
    output_path = f"{BASE_DIR}/SubmitSrgb_2model_alpha_search_{alpha:.3f}.csv"
    
    block_strings = []
    for i in range(1280):
        b64 = base64.b64encode(blended[i].tobytes()).decode('utf-8')
        block_strings.append(b64)
    
    out_df = pd.DataFrame({'ID': list(range(1280)), 'BLOCK': block_strings})
    out_df.to_csv(output_path, index=False)
    file_size = os.path.getsize(output_path) / (1024*1024)
    print(f"\n✅ Generated {output_path}")
    print(f"   size={file_size:.1f}MB, alpha={alpha:.3f}, local_psnr={local_psnr:.4f}")

print("\nDone!")