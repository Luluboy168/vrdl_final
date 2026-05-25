#!/usr/bin/env python3
"""Generate missing 2-model ensemble CSVs and verify format."""
import numpy as np
import pandas as pd
import base64
import os
import time

BASE = '/home/luluboy/projects/vrdl_final/submissions'

print("Loading NAFNet_TTA...")
df_naf = pd.read_csv(f'{BASE}/SubmitSrgb_nafnet_tta8_fixed.csv')
print(f"  NAFNet: {len(df_naf)} rows, cols={list(df_naf.columns)}, shape check={df_naf['BLOCK'].str.len().iloc[0]}")

print("Loading Restormer_FT_TTA...")
df_res = pd.read_csv(f'{BASE}/SubmitSrgb_restormer_ft_tta8.csv')
print(f"  Restormer_FT_TTA: {len(df_res)} rows, cols={list(df_res.columns)}")

# Target alphas
target_alphas = [0.705, 0.71, 0.715, 0.725]

# Load blocks as arrays
print("Decoding NAFNet blocks...")
naf_blocks = []
for b64 in df_naf['BLOCK']:
    img = np.frombuffer(base64.b64decode(b64), dtype=np.uint8).reshape(256, 256, 3)
    naf_blocks.append(img)
naf_blocks = np.array(naf_blocks)
print(f"  NAFNet shape: {naf_blocks.shape}")

print("Decoding Restormer_FT_TTA blocks...")
res_blocks = []
for b64 in df_res['BLOCK']:
    img = np.frombuffer(base64.b64decode(b64), dtype=np.uint8).reshape(256, 256, 3)
    res_blocks.append(img)
res_blocks = np.array(res_blocks)
print(f"  Restormer_FT_TTA shape: {res_blocks.shape}")

# Generate for each alpha
for alpha in target_alphas:
    out_name = f'{BASE}/SubmitSrgb_2model_ftsearch_{alpha:.3f}.csv'
    if os.path.exists(out_name):
        print(f"Skip {alpha}: already exists")
        continue
    
    print(f"\nGenerating alpha={alpha}...")
    t0 = time.time()
    
    # Weighted average
    ens = alpha * naf_blocks.astype(float) + (1 - alpha) * res_blocks.astype(float)
    ens = np.clip(ens, 0, 255).round().astype(np.uint8)
    
    # Encode to CSV
    rows = []
    for i in range(ens.shape[0]):
        b64 = base64.b64encode(ens[i].tobytes()).decode('utf-8')
        rows.append({'ID': i, 'BLOCK': b64})
    
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_name, index=False)
    
    elapsed = time.time() - t0
    print(f"  Saved: {out_name} ({os.path.getsize(out_name)/1024/1024:.1f} MB) in {elapsed:.1f}s")
    
    # Verify format
    df_v = pd.read_csv(out_name)
    b64_len = len(df_v['BLOCK'].iloc[0])
    print(f"  Verify: rows={len(df_v)}, cols={list(df_v.columns)}, b64_len={b64_len}")
    assert len(df_v) == 1280, f"Wrong row count: {len(df_v)}"
    assert list(df_v.columns) == ['ID', 'BLOCK'], f"Wrong cols: {list(df_v.columns)}"
    assert df_v['ID'].tolist() == list(range(1280)), "ID mismatch"
    assert b64_len == 262144, f"Wrong b64 length: {b64_len}"
    print(f"  ✅ Format OK")

print("\nAll done!")