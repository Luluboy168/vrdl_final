#!/usr/bin/env python3
"""3-model ensemble: NAFTta + RestormerFTTTA + RestormerFTMSTTA"""
import pandas as pd, base64, numpy as np, sys

OUTDIR = '/home/luluboy/projects/vrdl_final/submissions'

NAFTA   = f'{OUTDIR}/SubmitSrgb_nafnet_tta8_fixed.csv'
RFTTA   = f'{OUTDIR}/SubmitSrgb_restormer_ft_tta8.csv'
RFMS    = f'{OUTDIR}/SubmitSrgb_restormer_ft_ms_tta.csv'

print("Loading CSVs...")
nafta_df = pd.read_csv(NAFTA).sort_values('ID').reset_index(drop=True)
rfttta_df = pd.read_csv(RFTTA).sort_values('ID').reset_index(drop=True)
rfms_df   = pd.read_csv(RFMS).sort_values('ID').reset_index(drop=True)

print(f"  NAFTta:      {len(nafta_df)} rows,  BLOCK len={len(nafta_df.loc[0,'BLOCK'])}")
print(f"  RestormerFTTTA: {len(rfttta_df)} rows, BLOCK len={len(rfttta_df.loc[0,'BLOCK'])}")
print(f"  RestormerFTMS:  {len(rfms_df)} rows,  BLOCK len={len(rfms_df.loc[0,'BLOCK'])}")

def b64_to_img(b64):
    return np.frombuffer(base64.b64decode(b64), dtype=np.uint8).reshape(256, 256, 3)

def img_to_b64(img):
    return base64.b64encode(img.tobytes()).decode('utf-8')

# Weight search: 3 models, sum=1.0
# NAFTta / RestormerFTTTA / RestormerFTMS
weight_configs = [
    (0.55, 0.25, 0.20, 'naftta_rfttta_rfms'),
    (0.50, 0.30, 0.20, 'naftta_rfttta_rfms'),
    (0.45, 0.30, 0.25, 'naftta_rfttta_rfms'),
    (0.50, 0.25, 0.25, 'naftta_rfttta_rfms'),
    (0.45, 0.25, 0.30, 'naftta_rfttta_rfms'),
]

for w_n, w_r1, w_r2, label in weight_configs:
    out_name = f'SubmitSrgb_3model_v4_{w_n:.2f}_{w_r1:.2f}_{w_r2:.2f}.csv'
    out_path = f'{OUTDIR}/{out_name}'
    
    print(f"\nGenerating 3-model ensemble NAF={w_n}, RFT-TTA={w_r1}, RFT-MS={w_r2}...")
    new_blocks = []
    for i in range(len(nafta_df)):
        img_n  = b64_to_img(nafta_df.loc[i, 'BLOCK']).astype(np.float32)
        img_r1 = b64_to_img(rfttta_df.loc[i, 'BLOCK']).astype(np.float32)
        img_r2 = b64_to_img(rfms_df.loc[i, 'BLOCK']).astype(np.float32)
        blended = w_n * img_n + w_r1 * img_r1 + w_r2 * img_r2
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        new_blocks.append(img_to_b64(blended))
    
    result = pd.DataFrame({'ID': nafta_df['ID'], 'BLOCK': new_blocks})
    result.to_csv(out_path, index=False)
    print(f'  Saved {out_name}, rows={len(result)}, BLOCK len={len(new_blocks[0])}')
    
    # Quick validation
    df_check = pd.read_csv(out_path)
    assert list(df_check.columns) == ['ID', 'BLOCK'], f"Column mismatch: {list(df_check.columns)}"
    assert df_check['ID'].tolist() == list(range(1280)), f"ID range mismatch"
    assert all(len(b) == 262144 for b in df_check['BLOCK']), f"Block size mismatch!"
    print(f'  ✅ Format check PASSED')

print("\nAll done!")