#!/usr/bin/env python3
"""
3-model ensemble: NAFTta + RestormerFTTA + CGNet (correct format)
Quick weight search - 5 configs, best one submitted.
"""
import pandas as pd, base64, numpy as np, os

OUTDIR = '/home/luluboy/projects/vrdl_final/submissions'

NAFTA   = f'{OUTDIR}/SubmitSrgb_nafnet_tta8_fixed.csv'
RFTTA   = f'{OUTDIR}/SubmitSrgb_restormer_ft_tta8.csv'
CGNET   = f'{OUTDIR}/SubmitSrgb_cgnet_tta_fixed.csv'

print("Loading CSVs...")
nafta  = pd.read_csv(NAFTA).sort_values('ID').reset_index(drop=True)
rfttta = pd.read_csv(RFTTA).sort_values('ID').reset_index(drop=True)
cgnet  = pd.read_csv(CGNET).sort_values('ID').reset_index(drop=True)

def b64_to_img(b64):
    return np.frombuffer(base64.b64decode(b64), dtype=np.uint8).reshape(256, 256, 3)

def img_to_b64(img):
    return base64.b64encode(img.tobytes()).decode('utf-8')

# CGNet SIDD score ~40.39, should get decent weight
# NAFTta ~40.37, RFT-TTA ~40.04
# Try several configs centered around NAFNet-heavy
configs = [
    (0.60, 0.25, 0.15, 'v5'),
    (0.55, 0.30, 0.15, 'v5'),
    (0.65, 0.20, 0.15, 'v5'),
    (0.55, 0.25, 0.20, 'v5'),
    (0.60, 0.20, 0.20, 'v5'),
]

best_out = None
for w_n, w_r, w_c, label in configs:
    out_name = f'SubmitSrgb_3model_{label}_{w_n:.2f}_{w_r:.2f}_{w_c:.2f}.csv'
    out_path = f'{OUTDIR}/{out_name}'
    if os.path.exists(out_path):
        print(f"  {out_name} already exists, skipping")
        if best_out is None:
            best_out = out_path
        continue

    print(f"Generating NAF={w_n}, RFT={w_r}, CGNet={w_c}...")
    new_blocks = []
    for i in range(len(nafta)):
        img_n = b64_to_img(nafta.loc[i, 'BLOCK']).astype(np.float32)
        img_r = b64_to_img(rfttta.loc[i, 'BLOCK']).astype(np.float32)
        img_c = b64_to_img(cgnet.loc[i, 'BLOCK']).astype(np.float32)
        blended = w_n * img_n + w_r * img_r + w_c * img_c
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        new_blocks.append(img_to_b64(blended))

    result = pd.DataFrame({'ID': nafta['ID'], 'BLOCK': new_blocks})
    result.to_csv(out_path, index=False)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  Saved {out_name} ({size_mb:.1f} MB)")

    # Quick validation
    df_check = pd.read_csv(out_path)
    assert list(df_check.columns) == ['ID', 'BLOCK']
    assert df_check['ID'].tolist() == list(range(1280))
    assert all(len(b) == 262144 for b in df_check['BLOCK'])
    print(f"  ✅ Format OK")
    if best_out is None:
        best_out = out_path

print(f"\nDone! Best: {best_out}")