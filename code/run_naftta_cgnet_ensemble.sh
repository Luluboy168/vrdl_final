#!/usr/bin/env python3
"""
NAFTta + CGNet-TTA ensemble - generate multiple alpha values + submit.
Usage: tmux new -s naftta_cgnet_ensemble "bash -c '...'"
"""
import os, sys, base64, numpy as np, pandas as pd, subprocess, time

PROJECT = '/home/luluboy/projects/vrdl_final'
NAF_CSV = f'{PROJECT}/submissions/SubmitSrgb_nafnet_tta8_fixed.csv'
CGNET_CSV = f'{PROJECT}/submissions/SubmitSrgb_cgnet_tta_fixed.csv'
OUT_DIR = f'{PROJECT}/submissions'

ALPHAS = [0.70, 0.75, 0.80]  # alpha = weight for NAFTta

print("Loading CSVs (this takes ~1-2 min)...")
df_naf = pd.read_csv(NAF_CSV)
df_cg = pd.read_csv(CGNET_CSV)

assert list(df_naf.columns) == ['ID', 'BLOCK']
assert list(df_cg.columns) == ['ID', 'BLOCK']
assert df_naf['ID'].tolist() == list(range(1280))
print(f"Loaded: NAF={len(df_naf)}, CGNet={len(df_cg)}")

def make_ensemble(df_naf, df_cg, alpha, out_path):
    rows = []
    for i in range(1280):
        img_n = np.frombuffer(base64.b64decode(df_naf.iloc[i]['BLOCK']), np.uint8).reshape(256, 256, 3).astype(np.float64)
        img_c = np.frombuffer(base64.b64decode(df_cg.iloc[i]['BLOCK']), np.uint8).reshape(256, 256, 3).astype(np.float64)
        img_ens = (alpha * img_n + (1 - alpha) * img_c).clip(0, 255).astype(np.uint8)
        rows.append({'ID': i, 'BLOCK': base64.b64encode(img_ens.tobytes()).decode('utf-8')})
        if (i+1) % 256 == 0:
            print(f"  alpha={alpha:.2f}: {i+1}/1280")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  Saved {out_path} ({size_mb:.1f} MB)")

# Generate all alphas
for alpha in ALPHAS:
    out_name = f'SubmitSrgb_2model_naftta_cgnet_alpha{int(alpha*100)}.csv'
    out_path = f'{OUT_DIR}/{out_name}'
    if os.path.exists(out_path):
        print(f"  Already exists: {out_name}, skipping generation")
    else:
        print(f"\nGenerating alpha={alpha:.2f}...")
        make_ensemble(df_naf, df_cg, alpha, out_path)

# Verify all files
print("\n=== Verification ===")
for alpha in ALPHAS:
    out_name = f'SubmitSrgb_2model_naftta_cgnet_alpha{int(alpha*100)}.csv'
    out_path = f'{OUT_DIR}/{out_name}'
    df = pd.read_csv(out_path)
    b64_len = len(df['BLOCK'].iloc[0])
    status = "✅" if b64_len == 262144 else f"❌ len={b64_len}"
    print(f"  {out_name}: {status} ({os.path.getsize(out_path)/1e6:.1f} MB)")

print("\nDone generating. Next: manually submit via kaggle CLI.")