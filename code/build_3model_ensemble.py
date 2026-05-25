#!/usr/bin/env python3
"""Build and submit 3-model ensemble with NAFNet + Restormer + Restormer-FT."""
import numpy as np
import scipy.io as sio
import base64, pandas as pd, os

PYTHON = '/home/luluboy/miniconda3/bin/python3'
BASE = '/home/luluboy/projects/vrdl_final/submissions'
PROJ = '/home/luluboy/projects/vrdl_final'
DATA = f'{PROJ}/data'

def psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0: return 100.0
    return 10 * np.log10(255**2 / mse)

# Load ground truth for local validation
print("Loading GT for local validation...")
gt_mat = sio.loadmat(f'{DATA}/ValidationGtBlocksSrgb.mat')
gt = gt_mat['ValidationGtBlocksSrgb']  # (40, 32, 256, 256, 3)

def load_csv_blocks(csv_path):
    df = pd.read_csv(csv_path)
    blocks = []
    for _, row in df.iterrows():
        b64 = row['BLOCK']
        img_bytes = base64.b64decode(b64)
        arr = np.frombuffer(img_bytes, dtype=np.uint8).reshape(256, 256, 3)
        blocks.append(arr)
    return np.array(blocks)

# Load 3 models: NAFNet, Restormer, Restormer-FT
print("Loading NAFNet baseline...")
naf = load_csv_blocks(f'{BASE}/SubmitSrgb_baseline_official.csv')

print("Loading Restormer baseline...")
rest = load_csv_blocks(f'{BASE}/SubmitSrgb_restormer_official.csv')

print("Loading Restormer-FT...")
rest_ft = load_csv_blocks(f'{BASE}/SubmitSrgb_restormer_ft.csv')

print(f"NAFNet: {naf.shape}, Restormer: {rest.shape}, Restormer-FT: {rest_ft.shape}")

# Local validation: per-model PSNR
print("\n=== Per-model Local Validation ===")
for name, arr in [('NAFNet', naf), ('Restormer', rest), ('Restormer-FT', rest_ft)]:
    psnrs = []
    for k in range(1280):
        i, j = k // 32, k % 32
        psnrs.append(psnr(arr[k], gt[i, j]))
    print(f"  {name}: {np.mean(psnrs):.4f}")

# Grid search: find best (w1, w2, w3) for NAFNet + Restormer + Restormer-FT
# Keep NAFNet dominant since it's best (40.3675)
print("\n=== 3-Model Ensemble Grid Search ===")
best_score = -1
best_weights = (0.7, 0.3, 0.0)

for w1 in [0.60, 0.65, 0.70, 0.72, 0.75]:  # NAFNet weight
    for w2 in [0.10, 0.15, 0.20, 0.25, 0.30]:  # Restormer weight
        w3 = round(1.0 - w1 - w2, 2)
        if w3 < 0 or w3 > 0.30:
            continue
        ens = (w1 * naf.astype(float) + w2 * rest.astype(float) + w3 * rest_ft.astype(float)).astype(np.uint8)
        psnrs_ens = [psnr(ens[k], gt[k//32, k%32]) for k in range(1280)]
        mean_psnr = np.mean(psnrs_ens)
        if mean_psnr > best_score:
            best_score = mean_psnr
            best_weights = (w1, w2, w3)
            print(f"  NEW BEST: w=({w1:.2f},{w2:.2f},{w3:.2f}) local_PSNR={mean_psnr:.4f}")
        else:
            print(f"  w=({w1:.2f},{w2:.2f},{w3:.2f}) local_PSNR={mean_psnr:.4f}")

print(f"\nBest local weights: {best_weights} with PSNR={best_score:.4f}")

# NOTE: local validation has mismatch with Kaggle. Trust Kaggle over local.
# But this gives relative ranking. NAFNet is best (40.37 on Kaggle) so NAFNet should dominate.

# Build the best ensemble CSV based on Kaggle-trusted weights
# Current best Kaggle: 0.7 NAFNet + 0.3 Restormer = 40.4874
# Try: 0.65 NAFNet + 0.25 Restormer + 0.10 Restormer-FT
print("\n=== Building submission CSVs ===")

# For 3-model, use NAFNet-heavy weights
# Based on Kaggle: NAFNet=40.37 > Restormer=40.09 > Restormer-FT=40.09
# But Restormer-FT might have different error patterns → useful for ensemble diversity
combos = [
    ('ensemble_v5_65_25_10', 0.65, 0.25, 0.10),
    ('ensemble_v6_60_20_20', 0.60, 0.20, 0.20),
]

for name, w1, w2, w3 in combos:
    csv_path = f'{BASE}/SubmitSrgb_{name}.csv'
    if os.path.exists(csv_path):
        print(f"  {name}: already exists, skipping")
        continue
    print(f"  Building {name}: NAFNet={w1}, Restormer={w2}, Restormer-FT={w3}")
    ens = (w1 * naf.astype(float) + w2 * rest.astype(float) + w3 * rest_ft.astype(float)).astype(np.uint8)
    # Save as CSV
    output = []
    for k in range(1280):
        output.append({'ID': k, 'BLOCK': base64.b64encode(ens[k].tobytes()).decode('utf-8')})
    df_out = pd.DataFrame(output)
    df_out.to_csv(csv_path, index=False)
    # Validate format
    df_check = pd.read_csv(csv_path)
    b64_len = len(df_check['BLOCK'].iloc[0])
    print(f"    Saved: {csv_path} ({os.path.getsize(csv_path)/1e6:.1f} MB), b64_len={b64_len}")
    assert b64_len == 262144, f"ERROR: expected 262144, got {b64_len}"
    assert df_check['ID'].tolist() == list(range(1280))
    print(f"    ✅ Format OK")

print("\nDone! Next: submit to Kaggle with:")
print(f"  kaggle competitions submit -c sidd-benchmark-srgb-psnr -f {BASE}/SubmitSrgb_ensemble_v5_65_25_10.csv -m 'Ensemble v5: 0.65 NAFNet + 0.25 Restormer + 0.10 Restormer-FT'")