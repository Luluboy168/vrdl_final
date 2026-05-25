#!/usr/bin/env python3
"""
Ensemble NAFNet baseline + Restormer-FT Multi-Scale TTA
Optimal alpha from prior runs: NAFNet α=0.735 (best LB score)
Also generate a few nearby alphas for grid-search
"""
import os, sys, time
import pandas as pd
import base64
import numpy as np

PYTHON = '/home/luluboy/miniconda3/bin/python3'
NAFNET   = '/home/luluboy/projects/vrdl_final/submissions/SubmitSrgb_baseline_official.csv'
RESTORMS = '/home/luluboy/projects/vrdl_final/submissions/SubmitSrgb_restormer_ft_ms_tta.csv'
OUTDIR   = '/home/luluboy/projects/vrdl_final/submissions'

assert os.path.exists(NAFNET),   f'NAFNet baseline not found: {NAFNET}'
assert os.path.exists(RESTORMS), f'Restormer MS-TTA not found: {RESTORMS}'

print('Loading NAFNet baseline...')
t0 = time.time()
nafnet = pd.read_csv(NAFNET)
print(f'  Loaded NAFNet: {len(nafnet)} rows, first BLOCK len={len(nafnet.iloc[0]["BLOCK"])}')

print('Loading Restormer-MS-TTA...')
restorms = pd.read_csv(RESTORMS)
print(f'  Loaded Restormer: {len(restorms)} rows, first BLOCK len={len(restorms.iloc[0]["BLOCK"])}')

# Sort by ID
nafnet  = nafnet.sort_values('ID').reset_index(drop=True)
restorms = restorms.sort_values('ID').reset_index(drop=True)

assert nafnet['ID'].tolist() == list(range(1280)), 'NAFNet ID mismatch'
assert restorms['ID'].tolist() == list(range(1280)), 'Restormer ID mismatch'

def b64_to_img(b64):
    return np.frombuffer(base64.b64decode(b64), dtype=np.uint8).reshape(256, 256, 3).copy()

def img_to_b64(img):
    return base64.b64encode(img.tobytes()).decode('utf-8')

# Alphas to try (NAFNet weight)
# Prior best: α=0.735
alphas = [0.725, 0.730, 0.735, 0.740, 0.745]
best_alpha = 0.735

print(f'\nGenerating ensembles for alphas={alphas} ...')
print(f'Best alpha (will submit first): {best_alpha}')

for alpha in alphas:
    out_name = f'SubmitSrgb_2model_msfttta_{str(alpha).replace(".", "")}.csv'
    out_path = os.path.join(OUTDIR, out_name)
    
    if os.path.exists(out_path):
        print(f'  [SKIP] {out_name} already exists')
        continue
    
    print(f'  Building alpha={alpha}...')
    t1 = time.time()
    new_blocks = []
    for i in range(len(nafnet)):
        img_n = b64_to_img(nafnet.loc[i, 'BLOCK'])
        img_r = b64_to_img(restorms.loc[i, 'BLOCK'])
        # Blend in uint8 space
        blended = np.clip(
            np.round(alpha * img_n.astype(np.float32) + (1 - alpha) * img_r.astype(np.float32)),
            0, 255
        ).astype(np.uint8)
        new_blocks.append(img_to_b64(blended))
    
    result = pd.DataFrame({'ID': nafnet['ID'], 'BLOCK': new_blocks})
    result.to_csv(out_path, index=False)
    elapsed = time.time() - t1
    size_mb = os.path.getsize(out_path) / 1e6
    first_len = len(new_blocks[0])
    print(f'  Saved {out_name} ({size_mb:.1f} MB, {elapsed:.1f}s, first BLOCK len={first_len})')

print(f'\n✅ All ensembles done in {(time.time()-t0)/60:.1f} min')
print(f'Best alpha: {best_alpha}')
print(f'Recommend submitting: SubmitSrgb_2model_msfttta_{str(best_alpha).replace(".", "")}.csv')