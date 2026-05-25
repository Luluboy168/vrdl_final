#!/usr/bin/env python3
"""3-model Top-3 ensemble: NAFTta + CGNet + RFT-TTA (uint8 format)"""
import pandas as pd
import base64
import numpy as np

SUB = '/home/luluboy/projects/vrdl_final/submissions'

# Load CSVs
print("Loading CSVs...")
nafta  = pd.read_csv(f'{SUB}/SubmitSrgb_nafnet_tta8_fixed.csv')
rft    = pd.read_csv(f'{SUB}/SubmitSrgb_restormer_ft_tta8.csv')
cgnet  = pd.read_csv(f'{SUB}/SubmitSrgb_cgnet_tta_fixed.csv')

# Decode block: all existing CSVs use uint8 (196608 bytes) format
def decode_block(b):
    arr = np.frombuffer(base64.b64decode(b), dtype=np.uint8)
    return arr.reshape(256, 256, 3)

def encode_block(arr):
    return base64.b64encode(arr.astype(np.uint8).tobytes()).decode()

weight_combos = [
    (0.60, 0.20, 0.20, '0.60_0.20_0.20'),
    (0.65, 0.18, 0.17, '0.65_0.18_0.17'),
    (0.55, 0.25, 0.20, '0.55_0.25_0.20'),
    (0.70, 0.20, 0.10, '0.70_0.20_0.10'),
    (0.58, 0.22, 0.20, '0.58_0.22_0.20'),
]

for wa, wc, wr, tag in weight_combos:
    print(f"Doing {tag}...")
    rows = []
    for i in range(len(nafta)):
        a = decode_block(nafta.iloc[i]['BLOCK']).astype(np.float32)
        c = decode_block(cgnet.iloc[i]['BLOCK']).astype(np.float32)
        r = decode_block(rft.iloc[i]['BLOCK']).astype(np.float32)
        blended = wa * a + wc * c + wr * r
        rows.append({'ID': int(nafta.iloc[i]['ID']), 'BLOCK': encode_block(blended)})
    
    out = pd.DataFrame(rows)
    out_path = f'{SUB}/SubmitSrgb_3model_top3_{tag}.csv'
    out.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}, {len(out)} rows, block_len={out['BLOCK'].str.len().iloc[0]}")

print("Done!")