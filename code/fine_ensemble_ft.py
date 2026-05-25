#!/usr/bin/env python3
"""
Fine-grained ensemble: NAFNet baseline + Restormer-FT
(Kaggle best was α=0.72 → 40.4816)
"""
import pandas as pd, numpy as np, base64, os

BASE = "/home/luluboy/projects/vrdl_final/submissions"

print("Loading CSVs...")
naf_df = pd.read_csv(f"{BASE}/SubmitSrgb_baseline_official.csv")
rest_df = pd.read_csv(f"{BASE}/SubmitSrgb_restormer_ft.csv")

print(f"NAF: {len(naf_df)} rows, REST: {len(rest_df)} rows")

# Decode all blocks
print("Decoding NAFNet blocks...")
naf_arr = np.zeros((1280, 256, 256, 3), dtype=np.uint8)
for i in range(1280):
    naf_arr[i] = np.frombuffer(base64.b64decode(naf_df.loc[i,'BLOCK']), dtype=np.uint8).reshape(256,256,3)

print("Decoding Restormer-FT blocks...")
rest_arr = np.zeros((1280, 256, 256, 3), dtype=np.uint8)
for i in range(1280):
    rest_arr[i] = np.frombuffer(base64.b64decode(rest_df.loc[i,'BLOCK']), dtype=np.uint8).reshape(256,256,3)

print(f"NAF: {naf_arr.shape}, REST: {rest_arr.shape}")

# Test fine alphas 0.700 to 0.760 in 0.002 steps (focused around 0.72)
alphas = [round(a,3) for a in np.arange(0.700, 0.762, 0.002)]
print(f"\nTesting {len(alphas)} alphas around the best known value (0.72)...")

results = []
for alpha in alphas:
    blended = np.clip(np.round(alpha * naf_arr.astype(np.float32) + (1-alpha) * rest_arr.astype(np.float32)), 0, 255).astype(np.uint8)
    
    # Local MSE vs NAFNet baseline (for relative ranking only - NOT Kaggle PSNR)
    mse_vs_naf = np.mean((blended.astype(np.float32) - naf_arr.astype(np.float32))**2)
    psnr_vs_naf = 10 * np.log10(255**2 / (mse_vs_naf + 1e-10))
    
    # MSE vs Restormer
    mse_vs_rest = np.mean((blended.astype(np.float32) - rest_arr.astype(np.float32))**2)
    psnr_vs_rest = 10 * np.log10(255**2 / (mse_vs_rest + 1e-10))
    
    # Weighted MSE (proxy since we don't have GT) - closer to GT is between these
    results.append((alpha, psnr_vs_naf, psnr_vs_rest))
    print(f"  alpha={alpha:.3f}  psnr_vs_NAF={psnr_vs_naf:.3f}  psnr_vs_REST={psnr_vs_rest:.3f}")

# Also test a few broader alphas for contrast
broader = [0.68, 0.70, 0.72, 0.74, 0.76, 0.78]
print("\nBroader alpha test (for reference):")
for alpha in broader:
    if alpha in [round(a,3) for a in np.arange(0.700, 0.762, 0.002)]:
        continue
    blended = np.clip(np.round(alpha * naf_arr.astype(np.float32) + (1-alpha) * rest_arr.astype(np.float32)), 0, 255).astype(np.uint8)
    mse_vs_naf = np.mean((blended.astype(np.float32) - naf_arr.astype(np.float32))**2)
    psnr_vs_naf = 10 * np.log10(255**2 / (mse_vs_naf + 1e-10))
    results.append((alpha, psnr_vs_naf, 0))
    print(f"  alpha={alpha:.3f}  psnr_vs_NAF={psnr_vs_naf:.3f}")

# Generate top 3 alphas as CSVs
results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
print("\nTop alphas (by psnr_vs_NAF):")
for a, p, r in results_sorted[:5]:
    print(f"  alpha={a:.3f}  psnr_vs_NAF={p:.3f}")

# Also pick alphas that are most DIFFERENT from 0.72 (explore boundaries)
print("\nGenerating CSVs for alpha=0.720 and alpha=0.730...")
for target_alpha in [0.720, 0.730]:
    alpha = target_alpha
    blended = np.clip(np.round(alpha * naf_arr.astype(np.float32) + (1-alpha) * rest_arr.astype(np.float32)), 0, 255).astype(np.uint8)
    
    block_strings = [base64.b64encode(blended[i].tobytes()).decode('utf-8') for i in range(1280)]
    out_df = pd.DataFrame({'ID': list(range(1280)), 'BLOCK': block_strings})
    
    fname = f"{BASE}/SubmitSrgb_2model_ftsearch_{alpha:.3f}.csv"
    out_df.to_csv(fname, index=False)
    size_mb = os.path.getsize(fname) / (1024*1024)
    
    # Validate
    check = pd.read_csv(fname)
    assert list(check.columns) == ['ID','BLOCK'], f"Columns wrong: {check.columns}"
    assert check['ID'].tolist() == list(range(1280)), f"IDs wrong"
    assert all(len(b) == 262144 for b in check['BLOCK']), f"Block sizes wrong"
    print(f"  ✅ {fname} ({size_mb:.1f}MB)")

print("\nAll done!")
print("NOTE: This local metric measures similarity to NAFNet, NOT Kaggle PSNR.")
print("Known Kaggle results: α=0.72→40.4816, α=0.74→40.4455, α=0.68→40.4467, α=0.80→40.4284")