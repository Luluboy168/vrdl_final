#!/usr/bin/env python3
"""Efficient ensemble weight search - vectorized batch processing.
Best known: alpha=0.7 -> 40.4874 PSNR
Trying: 0.72 (slightly more NAFNet weight, which outperforms Restormer individually)
"""
import pandas as pd
import numpy as np
import base64
import os
import time

BASE = "/home/luluboy/projects/vrdl_final/submissions"
NAFNET_F = f"{BASE}/SubmitSrgb_nafnet_tta8_fixed.csv"
RESTORMER_F = f"{BASE}/SubmitSrgb_restormer_tta8.csv"
OUT_DIR = f"{BASE}/weight_search"
os.makedirs(OUT_DIR, exist_ok=True)

ALPHA = 0.72
BETA = 1.0 - ALPHA
OUT_NAME = f"SubmitSrgb_ensemble_alpha72.csv"
OUT_PATH = f"{OUT_DIR}/{OUT_NAME}"

print(f"Loading NAFNet+TTA ({NAFNET_F})...")
t0 = time.time()
naf = pd.read_csv(NAFNET_F)
print(f"  Loaded in {time.time()-t0:.1f}s, shape: {naf.shape}")

print(f"Loading Restormer+TTA ({RESTORMER_F})...")
t0 = time.time()
rest = pd.read_csv(RESTORMER_F)
print(f"  Loaded in {time.time()-t0:.1f}s, shape: {rest.shape}")

# Sort by ID
naf = naf.sort_values('ID').reset_index(drop=True)
rest = rest.sort_values('ID').reset_index(drop=True)

n_blocks = len(naf)
print(f"Processing {n_blocks} blocks with alpha={ALPHA} (beta={BETA})...")

t0 = time.time()

# Pre-decode all blocks into numpy arrays (batch decode)
naf_arr = np.zeros((n_blocks, 256, 256, 3), dtype=np.float32)
rest_arr = np.zeros((n_blocks, 256, 256, 3), dtype=np.float32)

for i in range(n_blocks):
    naf_b64 = naf['BLOCK'].iloc[i]
    rest_b64 = rest['BLOCK'].iloc[i]
    
    # Decode from base64 to raw bytes, reshape to (256,256,3) uint8
    naf_img = np.frombuffer(base64.b64decode(naf_b64), dtype=np.uint8).reshape(256, 256, 3)
    rest_img = np.frombuffer(base64.b64decode(rest_b64), dtype=np.uint8).reshape(256, 256, 3)
    
    naf_arr[i] = naf_img.astype(np.float32)
    rest_arr[i] = rest_img.astype(np.float32)
    
    if (i + 1) % 200 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        remaining = (n_blocks - i - 1) / rate
        print(f"  Decoded {i+1}/{n_blocks} blocks... {elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining")

print(f"All blocks decoded in {time.time()-t0:.1f}s")

# Weighted ensemble (vectorized)
t1 = time.time()
ens_arr = ALPHA * naf_arr + BETA * rest_arr
ens_arr = np.clip(ens_arr, 0, 255).astype(np.uint8)
print(f"Ensemble computed in {time.time()-t1:.1f}s")

# Encode back to base64 (one block at a time is fast enough)
t2 = time.time()
new_blocks = []
for i in range(n_blocks):
    b64 = base64.b64encode(ens_arr[i].tobytes()).decode('utf-8')
    new_blocks.append(b64)
    if (i + 1) % 200 == 0:
        print(f"  Encoded {i+1}/{n_blocks} blocks...")

print(f"All blocks encoded in {time.time()-t2:.1f}s, total: {time.time()-t0:.1f}s")

# Verify format
sample_len = len(new_blocks[0])
print(f"Output block length: {sample_len} (expected 262144)")
assert sample_len == 262144, f"Block length wrong: {sample_len}"
assert len(new_blocks) == 1280

# Create and save output
df = pd.DataFrame()
df['ID'] = np.arange(n_blocks)
df['BLOCK'] = new_blocks

df.to_csv(OUT_PATH, index=False)
size_mb = os.path.getsize(OUT_PATH) / 1e6
print(f"Saved to {OUT_PATH} ({size_mb:.1f} MB)")

# Quick validation
print("\nValidating format...")
df2 = pd.read_csv(OUT_PATH)
assert list(df2.columns) == ['ID', 'BLOCK']
assert df2['ID'].tolist() == list(range(1280))
assert all(len(b) == 262144 for b in df2['BLOCK'])
print("✅ Format validation PASSED")
print(f"\nNext: copy to submissions dir and submit with message 'NAFNet+TTA α=0.72 + Restormer+TTA α=0.28'")
print(f"cp {OUT_PATH} {BASE}/{OUT_NAME}")