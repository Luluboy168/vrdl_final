#!/usr/bin/env python3
"""
NAFTta + CGNet-TTA 2-model ensemble
"""
import os, sys, base64, cv2, numpy as np, pandas as pd

PROJECT = '/home/luluboy/projects/vrdl_final'
NAF_CSV = f'{PROJECT}/submissions/SubmitSrgb_nafnet_tta8_fixed.csv'
CGNET_CSV = f'{PROJECT}/submissions/SubmitSrgb_cgnet_tta_fixed.csv'
OUT_DIR = f'{PROJECT}/submissions'

print("Loading CSVs...")
df_naf = pd.read_csv(NAF_CSV)
df_cg = pd.read_csv(CGNET_CSV)

# Verify format
assert list(df_naf.columns) == ['ID', 'BLOCK']
assert list(df_cg.columns) == ['ID', 'BLOCK']
assert df_naf['ID'].tolist() == list(range(1280))
assert df_cg['ID'].tolist() == list(range(1280))
print(f"NAF: {len(df_naf)} rows, CGNet: {len(df_cg)} rows")

# Decode a few samples to verify
for i in [0, 640, 1279]:
    b64_n = df_naf.iloc[i]['BLOCK']
    b64_c = df_cg.iloc[i]['BLOCK']
    img_n = np.frombuffer(base64.b64decode(b64_n), np.uint8).reshape(256, 256, 3)
    img_c = np.frombuffer(base64.b64decode(b64_c), np.uint8).reshape(256, 256, 3)
    print(f"  Block {i}: NAF mean={img_n.mean():.1f} std={img_n.std():.1f}, CGNet mean={img_c.mean():.1f} std={img_c.std():.1f}")

# Try a few alpha values for NAFTta (rest is CGNet)
# alpha = weight for NAFTta (1-alpha = weight for CGNet)
best_alpha = 0.75
best_score = -1

print("\nSaving CSV for alpha=0.80 (NAFTta 0.80 + CGNet 0.20)...")
alpha = 0.80
rows = []
for i in range(1280):
    b64_n = df_naf.iloc[i]['BLOCK']
    b64_c = df_cg.iloc[i]['BLOCK']
    img_n = np.frombuffer(base64.b64decode(b64_n), np.uint8).reshape(256, 256, 3).astype(np.float64)
    img_c = np.frombuffer(base64.b64decode(b64_c), np.uint8).reshape(256, 256, 3).astype(np.float64)
    # Weighted average
    img_ens = (alpha * img_n + (1 - alpha) * img_c).clip(0, 255).astype(np.uint8)
    b64_out = base64.b64encode(img_ens.tobytes()).decode('utf-8')
    rows.append({'ID': i, 'BLOCK': b64_out})
    if (i+1) % 200 == 0:
        print(f"  Processed {i+1}/1280")

df_out = pd.DataFrame(rows)
out_path = f'{OUT_DIR}/SubmitSrgb_2model_naftta_cgnet_alpha80.csv'
df_out.to_csv(out_path, index=False)
print(f"\n✅ Saved: {out_path}")
print(f"   File size: {os.path.getsize(out_path) / 1e6:.1f} MB")

# Verify output
df_check = pd.read_csv(out_path)
assert len(df_check) == 1280
assert all(len(b) == 262144 for b in df_check['BLOCK'])
print("✅ Output verified")