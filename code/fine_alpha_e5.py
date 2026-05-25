#!/usr/bin/env python3
"""
Fine alpha search: NAFNet + Restormer-Epoch5
Try alphas from 0.700 to 0.760 in 0.005 steps
Generate submissions for top alphas
"""
import os, base64, numpy as np, pandas as pd

BASE = '/home/luluboy/projects/vrdl_final/submissions'

naf_df = pd.read_csv(f'{BASE}/SubmitSrgb_baseline_official.csv')
rest5_df = pd.read_csv(f'{BASE}/SubmitSrgb_restormer_ft_epoch5.csv')

# Decode to arrays
print('Decoding NAFNet baseline...')
naf_arr = np.array([np.frombuffer(base64.b64decode(naf_df.loc[i,'BLOCK']), dtype=np.uint8).reshape(256,256,3) for i in range(1280)])
print('Decoding Restormer-Epoch5...')
rest5_arr = np.array([np.frombuffer(base64.b64decode(rest5_df.loc[i,'BLOCK']), dtype=np.uint8).reshape(256,256,3) for i in range(1280)])
print(f'Shapes: NAF {naf_arr.shape}, REST5 {rest5_arr.shape}')

# Test alphas
alphas = [round(a, 3) for a in np.arange(0.700, 0.760, 0.005)]
print(f'Testing {len(alphas)} alphas...')

results = []
for alpha in alphas:
    blended = np.clip(np.round(alpha * naf_arr.astype(np.float64) + (1-alpha) * rest5_arr.astype(np.float64)), 0, 255).astype(np.uint8)
    # Local metric: MSE vs NAFNet (NOT Kaggle PSNR!)
    mse = np.mean((blended.astype(np.float64) - naf_arr.astype(np.float64))**2)
    psnr = 10 * np.log10(255**2 / (mse + 1e-10))
    results.append((alpha, psnr, blended))
    print(f'  alpha={alpha:.3f}  local_psnr={psnr:.3f}')

results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
print('\nTop 5 by local PSNR:')
for a, p, _ in results_sorted[:5]:
    print(f'  alpha={a:.3f}  {p:.3f}')

# Generate submissions for top 2 and a few around the best known value
top_alphas = [r[0] for r in results_sorted[:2]]
# Also add alphas near the best known Kaggle value (0.72)
for target in [0.718, 0.720, 0.722]:
    if target not in top_alphas:
        top_alphas.append(target)
top_alphas = sorted(set(top_alphas))

print(f'\nGenerating CSVs for: {top_alphas}')
for alpha in top_alphas:
    blended = np.clip(np.round(alpha * naf_arr.astype(np.float64) + (1-alpha) * rest5_arr.astype(np.float64)), 0, 255).astype(np.uint8)
    rows = [{'ID': i, 'BLOCK': base64.b64encode(blended[i].tobytes()).decode()} for i in range(1280)]
    out_path = f'{BASE}/SubmitSrgb_2model_e5_alpha_{alpha:.3f}.csv'
    pd.DataFrame(rows).to_csv(out_path, index=False)
    sz = os.path.getsize(out_path)/1e6
    print(f'  Saved: {out_path} ({sz:.1f} MB)')

print('EXIT_CODE:0')