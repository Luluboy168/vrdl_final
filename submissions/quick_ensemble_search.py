#!/usr/bin/env python3
"""4-model ensemble grid search using local validation GT."""
import numpy as np
import scipy.io as sio
import base64, pandas as pd, os

BASE = '/home/luluboy/projects/vrdl_final/submissions'
DATA = '/home/luluboy/projects/vrdl_final/data'

def psnr_metric(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0: return 100.0
    return 10 * np.log10(255**2 / mse)

def load_csv_blocks(csv_path):
    df = pd.read_csv(csv_path)
    blocks = []
    for _, row in df.iterrows():
        b64 = row['BLOCK']
        img_bytes = base64.b64decode(b64)
        arr = np.frombuffer(img_bytes, dtype=np.uint8).reshape(256, 256, 3)
        blocks.append(arr)
    return np.array(blocks)

# Load GT
print("Loading GT...")
gt_mat = sio.loadmat(f'{DATA}/ValidationGtBlocksSrgb.mat')
gt = gt_mat['ValidationGtBlocksSrgb']  # (40, 32, 256, 256, 3)

# Load all model outputs
print("Loading models...")
models = {}
model_files = {
    'NAFNet': 'SubmitSrgb_baseline_official.csv',
    'NAFNet_TTA': 'SubmitSrgb_nafnet_tta8_fixed.csv',
    'Restormer': 'SubmitSrgb_restormer_official.csv',
    'Restormer_TTA': 'SubmitSrgb_restormer_tta8.csv',
    'Restormer_FT': 'SubmitSrgb_restormer_ft.csv',
    'Restormer_FT_TTA': 'SubmitSrgb_restormer_ft_tta8.csv',
    'MIRNetv2': 'SubmitSrgb_mirnetv2_fixed.csv',
}
for name, fname in model_files.items():
    path = os.path.join(BASE, fname)
    if os.path.exists(path):
        models[name] = load_csv_blocks(path)
        # Quick local validation
        psnrs = [psnr_metric(models[name][k], gt[k//32, k%32]) for k in range(1280)]
        print(f"  {name}: local_PSNR={np.mean(psnrs):.4f}")
    else:
        print(f"  {name}: NOT FOUND ({fname})")

print(f"\nLoaded {len(models)} models")

# 4-model ensemble: NAFNet_TTA + Restormer_TTA + Restormer_FT_TTA + MIRNetv2
# NAFNet_TTA is our best single model, should dominate
print("\n=== 4-Model Ensemble Grid Search ===")
best_score = -1
best_weights = None
results = []

key_models = ['NAFNet_TTA', 'Restormer_TTA', 'Restormer_FT_TTA', 'MIRNetv2']
available = [m for m in key_models if m in models]
print(f"Using models: {available}")

for w1 in [0.50, 0.55, 0.60, 0.65, 0.70, 0.72, 0.75, 0.78]:
    for w2 in [0.05, 0.10, 0.15, 0.20]:
        for w3 in [0.05, 0.10, 0.15, 0.20, 0.25]:
            w4 = round(1.0 - w1 - w2 - w3, 2)
            if w4 < 0 or w4 > 0.20:
                continue
            ens = sum(w * models[m].astype(float) for w, m in zip([w1,w2,w3,w4], available))
            ens = np.clip(ens, 0, 255).round().astype(np.uint8)
            psnrs_ens = [psnr_metric(ens[k], gt[k//32, k%32]) for k in range(1280)]
            mean_psnr = np.mean(psnrs_ens)
            if mean_psnr > best_score:
                best_score = mean_psnr
                best_weights = (w1, w2, w3, w4)
                print(f"  NEW BEST: w={best_weights} local_PSNR={mean_psnr:.4f}")
            results.append((mean_psnr, (w1,w2,w3,w4)))

# Also try 3-model: NAFNet_TTA + Restormer_FT_TTA + MIRNetv2
print("\n=== 3-Model (NAFNet_TTA + Restormer_FT_TTA + MIRNetv2) ===")
for w1 in [0.50, 0.55, 0.60, 0.65, 0.70, 0.72, 0.75, 0.78, 0.80]:
    for w2 in [0.10, 0.15, 0.20, 0.25, 0.30]:
        w3 = round(1.0 - w1 - w2, 2)
        if w3 < 0 or w3 > 0.30:
            continue
        ens = (w1 * models['NAFNet_TTA'].astype(float) + 
               w2 * models['Restormer_FT_TTA'].astype(float) + 
               w3 * models['MIRNetv2'].astype(float))
        ens = np.clip(ens, 0, 255).round().astype(np.uint8)
        psnrs_ens = [psnr_metric(ens[k], gt[k//32, k%32]) for k in range(1280)]
        mean_psnr = np.mean(psnrs_ens)
        if mean_psnr > best_score:
            best_score = mean_psnr
            best_weights = (w1, w2, w3, '3-model')
            print(f"  NEW BEST: w=({w1},{w2},{w3}) local_PSNR={mean_psnr:.4f}")

# Also try 2-model NAFNet_TTA + Restormer_FT_TTA only
print("\n=== 2-Model (NAFNet_TTA + Restormer_FT_TTA) ===")
for w1 in [0.60, 0.65, 0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80, 0.82, 0.85]:
    w2 = round(1.0 - w1, 2)
    ens = (w1 * models['NAFNet_TTA'].astype(float) + 
           w2 * models['Restormer_FT_TTA'].astype(float))
    ens = np.clip(ens, 0, 255).round().astype(np.uint8)
    psnrs_ens = [psnr_metric(ens[k], gt[k//32, k%32]) for k in range(1280)]
    mean_psnr = np.mean(psnrs_ens)
    if mean_psnr > best_score:
        best_score = mean_psnr
        best_weights = (w1, w2, '2-model')
        print(f"  NEW BEST: w=({w1},{w2}) local_PSNR={mean_psnr:.4f}")
    else:
        print(f"  w=({w1},{w2}) local_PSNR={mean_psnr:.4f}")

print(f"\n=== BEST OVERALL: w={best_weights} local_PSNR={best_score:.4f} ===")

# Save best weights for next step
import json
with open('/home/luluboy/projects/vrdl_final/submissions/best_ensemble_weights.json', 'w') as f:
    json.dump({'weights': best_weights, 'local_psnr': best_score}, f)
print("Saved to best_ensemble_weights.json")