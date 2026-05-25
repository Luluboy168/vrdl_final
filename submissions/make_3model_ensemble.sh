#!/bin/bash
PYTHON=/home/luluboy/miniconda3/bin/python3

$PYTHON - <<PY
import pandas as pd, base64, numpy as np

NAFTta = '/home/luluboy/projects/vrdl_final/submissions/SubmitSrgb_nafnet_tta8_fixed.csv'
RestormerTTA = '/home/luluboy/projects/vrdl_final/submissions/SubmitSrgb_restormer_tta8.csv'
RestormerFT = '/home/luluboy/projects/vrdl_final/submissions/SubmitSrgb_restormer_ft.csv'
OUTDIR = '/home/luluboy/projects/vrdl_final/submissions'

print("Loading CSVs...")
nafta_df = pd.read_csv(NAFTta).sort_values('ID').reset_index(drop=True)
rttta_df = pd.read_csv(RestormerTTA).sort_values('ID').reset_index(drop=True)
rft_df   = pd.read_csv(RestormerFT).sort_values('ID').reset_index(drop=True)

print(f"  NAFTta: {len(nafta_df)} rows")
print(f"  RestormerTTA: {len(rttta_df)} rows")
print(f"  RestormerFT: {len(rft_df)} rows")

def b64_to_img(b64):
    return np.frombuffer(base64.b64decode(b64), dtype=np.uint8).reshape(256, 256, 3)

def img_to_b64(img):
    return base64.b64encode(img.tobytes()).decode('utf-8')

# 0.55 NAFTta + 0.30 RestormerTTA + 0.15 RestormerFT
# Also try 0.60/0.25/0.15
for w_nafta, w_rttta, w_rft, label in [
    (0.55, 0.30, 0.15, '553015'),
    (0.60, 0.25, 0.15, '602515'),
]:
    print(f"\nGenerating 3-model ensemble w=({w_nafta},{w_rttta},{w_rft})...")
    new_blocks = []
    for i in range(len(nafta_df)):
        img_n = b64_to_img(nafta_df.loc[i, 'BLOCK']).astype(np.float32)
        img_r1 = b64_to_img(rttta_df.loc[i, 'BLOCK']).astype(np.float32)
        img_r2 = b64_to_img(rft_df.loc[i, 'BLOCK']).astype(np.float32)
        blended = w_nafta * img_n + w_rttta * img_r1 + w_rft * img_r2
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        new_blocks.append(img_to_b64(blended))

    result = pd.DataFrame({'ID': nafta_df['ID'], 'BLOCK': new_blocks})
    fname = f'{OUTDIR}/SubmitSrgb_3model_{label}.csv'
    result.to_csv(fname, index=False)
    print(f'Saved {fname}, rows={len(result)}, first BLOCK len={len(new_blocks[0])}')

    # Quick validation
    df_check = pd.read_csv(fname)
    assert list(df_check.columns) == ['ID', 'BLOCK']
    assert df_check['ID'].tolist() == list(range(1280))
    assert all(len(b) == 262144 for b in df_check['BLOCK']), "Block size mismatch!"
    print(f'  ✅ Format check PASSED ({len(df_check)} rows)')

print("\nAll done!")
PY
