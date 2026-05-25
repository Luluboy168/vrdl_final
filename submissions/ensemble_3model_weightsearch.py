#!/usr/bin/env python3
"""3-model ensemble: NAFTta + CGNet-TTA + RFT-TTA8 weight search + local validation"""
import base64, pandas as pd, numpy as np, scipy.io as sio, os, json, sys

PYTHON = '/home/luluboy/miniconda3/bin/python3'
BASE = '/home/luluboy/projects/vrdl_final/submissions'
PROJ = '/home/luluboy/projects/vrdl_final'
DATA = f'{PROJ}/data'

def psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0: return 100.0
    return 10 * np.log10(255**2 / mse)

print("Loading GT for local validation...")
gt_mat = sio.loadmat(f'{DATA}/ValidationGtBlocksSrgb.mat')
gt = gt_mat['ValidationGtBlocksSrgb']  # (40, 32, 256, 256, 3)
print(f"GT shape: {gt.shape}")

def decode_blocks(series):
    raw = [base64.b64decode(s) for s in series]
    arr = np.array([np.frombuffer(b, dtype=np.uint8) for b in raw], dtype=np.uint8)
    return arr

# Load CSVs
print("Loading CSVs (this may take ~1 min)...")
df_naftta  = pd.read_csv(f'{BASE}/SubmitSrgb_nafnet_tta8_fixed.csv')
df_cgnet   = pd.read_csv(f'{BASE}/SubmitSrgb_cgnet_tta_fixed.csv')
df_rft     = pd.read_csv(f'{BASE}/SubmitSrgb_restormer_ft_tta8.csv')

print(f"  NAFTta:  {len(df_naftta)} rows")
print(f"  CGNet:   {len(df_cgnet)} rows")
print(f"  RFT-TTA: {len(df_rft)} rows")

assert (df_naftta.ID == df_cgnet.ID).all()
assert (df_naftta.ID == df_rft.ID).all()
print("  IDs aligned ✓")

print("Decoding blocks (3x ~335MB CSVs)... this takes ~60-90s")
blocks_naftta = decode_blocks(df_naftta.BLOCK)
print(f"  NAFTta decoded: {blocks_naftta.shape}")
blocks_cgnet  = decode_blocks(df_cgnet.BLOCK)
print(f"  CGNet decoded")
blocks_rft    = decode_blocks(df_rft.BLOCK)
print(f"  RFT decoded")

# Reshape to (40, 32, 256, 256, 3) for validation
b_naftta = blocks_naftta.reshape(40, 32, 256, 256, 3)
b_cgnet  = blocks_cgnet.reshape(40, 32, 256, 256, 3)
b_rft    = blocks_rft.reshape(40, 32, 256, 256, 3)

# Per-model local PSNR
print("\n=== Per-model Local PSNR ===")
for name, arr in [('NAFTta', b_naftta), ('CGNet', b_cgnet), ('RFT', b_rft)]:
    psnrs = []
    for k in range(1280):
        i, j = k // 32, k % 32
        psnrs.append(psnr(arr[i, j], gt[i, j]))
    print(f"  {name}: mean={np.mean(psnrs):.4f} dB")

# Weight search: fine grid around known good weights
# Best so far: NAFTta+RFT alpha=0.745
# CGNet is typically worse, so give it lower weight
print("\n=== 3-Model Weight Search ===")

results = []
# Grid: w1 (NAFTta) × w2 (CGNet) × w3 (RFT) = 1.0
# Focus on CGNet weights 0.05-0.25, NAFTta dominant
for w1 in [0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76]:
    for w2 in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25]:
        w3 = round(1.0 - w1 - w2, 2)
        if w3 < 0.03 or w3 > 0.35:
            continue
        # Blend in float32
        blended = w1 * b_naftta.astype(np.float32) + \
                  w2 * b_cgnet.astype(np.float32) + \
                  w3 * b_rft.astype(np.float32)
        blended = np.clip(blended, 0, 255).round().astype(np.uint8)
        
        # Local PSNR
        psnrs = []
        for k in range(1280):
            i, j = k // 32, k % 32
            psnrs.append(psnr(blended[i, j], gt[i, j]))
        mean_psnr = np.mean(psnrs)
        results.append((w1, w2, w3, mean_psnr))
        print(f"  w=({w1:.2f},{w2:.2f},{w3:.2f}) -> local PSNR={mean_psnr:.4f}")

# Sort by local PSNR
results.sort(key=lambda x: -x[3])
print("\n=== Top 10 by Local PSNR ===")
for r in results[:10]:
    print(f"  w=({r[0]:.2f},{r[1]:.2f},{r[2]:.2f}) -> local={r[3]:.4f}")

best = results[0]
print(f"\nBest: w1={best[0]}, w2={best[1]}, w3={best[2]} -> local={best[3]:.4f}")

# Generate best CSV
print("\nGenerating best ensemble CSV...")
w1, w2, w3 = best[0], best[1], best[2]
# blocks_* are still in (1280, 196608) shape
blended_blocks = w1 * blocks_naftta.astype(np.float32) + \
                 w2 * blocks_cgnet.astype(np.float32) + \
                 w3 * blocks_rft.astype(np.float32)
blended_blocks = np.clip(blended_blocks, 0, 255).round().astype(np.uint8)

# Build submission CSV
output_rows = []
for idx, row in df_naftta.iterrows():
    img_bytes = blended_blocks[idx].tobytes()
    b64 = base64.b64encode(img_bytes).decode('utf-8')
    output_rows.append({'ID': int(row['ID']), 'BLOCK': b64})

out_df = pd.DataFrame(output_rows)
fname = f'{BASE}/SubmitSrgb_3model_ncr_best.csv'
out_df.to_csv(fname, index=False)

# Validate format
print(f"\nValidating {fname}...")
validate_df = pd.read_csv(fname)
assert list(validate_df.columns) == ['ID', 'BLOCK']
assert validate_df['ID'].tolist() == list(range(1280))
assert all(len(b) == 262144 for b in validate_df['BLOCK'])
fsize = os.path.getsize(fname)
print(f"✅ Format OK! File size: {fsize/1024/1024:.1f} MB")

# Save best weights
with open(f'{BASE}/best_3model_ncr_weights.json', 'w') as f:
    json.dump({'w_naftta': w1, 'w_cgnet': w2, 'w_rft': w3, 'local_psnr': float(best[3])}, f)

print("\nDone! Best weights:", best)
