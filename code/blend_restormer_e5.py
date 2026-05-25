#!/usr/bin/env python3
"""
Blend NAFNet (baseline official) + Restormer-Epoch5 with alpha=0.72
and NAFNet-TTA + Restormer-Epoch5 with alpha=0.72
"""
import os, base64, numpy as np, pandas as pd

BASE = '/home/luluboy/projects/vrdl_final/submissions'

# Load NAFNet baseline (non-TTA)
naf_df = pd.read_csv(f'{BASE}/SubmitSrgb_baseline_official.csv')
# Load NAFNet-TTA (8-way TTA)
naftta_df = pd.read_csv(f'{BASE}/SubmitSrgb_nafnet_tta8_fixed.csv')
# Load Restormer Epoch5 (non-TTA)
rest5_df = pd.read_csv(f'{BASE}/SubmitSrgb_restormer_ft_epoch5.csv')

print(f'NAF: {len(naf_df)}, NAFTTA: {len(naftta_df)}, REST5: {len(rest5_df)}')

def blend_csv(df1, df2, alpha, out_path):
    """Blend two CSVs with alpha weight for df1."""
    rows = []
    for i in range(1280):
        b1 = np.frombuffer(base64.b64decode(df1.loc[i,'BLOCK']), dtype=np.uint8).reshape(256,256,3).astype(np.float64)
        b2 = np.frombuffer(base64.b64decode(df2.loc[i,'BLOCK']), dtype=np.uint8).reshape(256,256,3).astype(np.float64)
        blended = (alpha * b1 + (1-alpha) * b2).clip(0,255).astype(np.uint8)
        rows.append({'ID': i, 'BLOCK': base64.b64encode(blended.tobytes()).decode()})
        if (i+1) % 200 == 0:
            print(f'  {i+1}/1280')
    pd.DataFrame(rows).to_csv(out_path, index=False)
    sz = os.path.getsize(out_path)/1e6
    b64_len = len(rows[0]['BLOCK'])
    print(f'  Saved: {out_path} ({sz:.1f} MB), b64_len={b64_len}')
    assert b64_len == 262144

# Blend 1: NAFNet + Restormer-Epoch5, alpha=0.72
blend_csv(naf_df, rest5_df, 0.720,
          f'{BASE}/SubmitSrgb_2model_e5blend_072.csv')

# Blend 2: NAFTta + Restormer-Epoch5, alpha=0.72
blend_csv(naftta_df, rest5_df, 0.720,
          f'{BASE}/SubmitSrgb_2model_naftta_e5_072.csv')

print('All done! EXIT_CODE:0')