#!/usr/bin/env python3
"""
Blend NAFTta+RFT-TTA blend (SubmitSrgb_2model_fttta_0.745.csv) with
Restormer-FT-TTA (SubmitSrgb_restormer_ft_tta8.csv) to test 2-model ensemble.
Result ≈ alpha*NAFTta + (1-alpha)*RFT + (1-alpha)*(Restormer-FT) blending is complex,
but we treat (NAFTta+RFT) as one component and Restormer-FT as second.
Blend: result = (1-w)*blend745 + w*restormer_ft
"""
import pandas as pd, numpy as np, base64, sys, os

SUBS = '/home/luluboy/projects/vrdl_final/submissions'
BLEND745 = os.path.join(SUBS, 'SubmitSrgb_2model_fttta_0.745.csv')
RFT8     = os.path.join(SUBS, 'SubmitSrgb_restormer_ft_tta8.csv')

print('Loading blend745...')
df745 = pd.read_csv(BLEND745)
print('Loading RFT8...')
dfRFT = pd.read_csv(RFT8)

print('Decoding blocks...')
blocks745 = []
blocksRFT = []
for i in range(1280):
    b745 = np.frombuffer(base64.b64decode(df745['BLOCK'].iloc[i]), dtype=np.uint8).reshape(256,256,3)
    brft = np.frombuffer(base64.b64decode(dfRFT['BLOCK'].iloc[i]), dtype=np.uint8).reshape(256,256,3)
    blocks745.append(b745)
    blocksRFT.append(brft)
print('Decoding done')

# Test a range of w values (weight for Restormer-FT in the blend)
for w in [0.10, 0.12, 0.15, 0.18, 0.20]:
    print(f'\nGenerating blend w={w}...')
    new_blocks = []
    for i in range(1280):
        blended = (blocks745[i].astype(np.float32) * (1-w) + 
                   blocksRFT[i].astype(np.float32) * w).clip(0,255).astype(np.uint8)
        new_blocks.append(base64.b64encode(blended.tobytes()).decode('utf-8'))
    
    out = pd.DataFrame({'ID': np.arange(1280), 'BLOCK': new_blocks})
    fname = f'SubmitSrgb_blend745_rft8_w{int(w*100)}.csv'
    out.to_csv(os.path.join(SUBS, fname), index=False)
    # Quick validation
    assert len(out['BLOCK'].iloc[0]) == 262144, f'Bad format at w={w}'
    print(f'  Saved {fname}')

print('\nAll done!')