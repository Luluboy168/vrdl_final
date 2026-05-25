#!/usr/bin/env python3
"""Generate missing alpha values 0.747 and 0.748 for NAFTta + RFT-TTA blend."""
import numpy as np, pandas as pd, base64, os, subprocess

SUB = "/home/luluboy/projects/vrdl_final/submissions"
naf_csv = f"{SUB}/SubmitSrgb_nafnet_tta8_fixed.csv"
rft_csv = f"{SUB}/SubmitSrgb_restormer_ft_tta8.csv"

print("Loading CSVs...")
naf_df = pd.read_csv(naf_csv)
rft_df = pd.read_csv(rft_csv)

# Fix 1-indexed IDs
for df in [naf_df, rft_df]:
    if df['ID'].min() == 1:
        df['ID'] -= 1
naf_df = naf_df.sort_values('ID').reset_index(drop=True)
rft_df = rft_df.sort_values('ID').reset_index(drop=True)

print("Decoding blocks...")
naf_blocks = [np.frombuffer(base64.b64decode(naf_df['BLOCK'].iloc[i]), dtype=np.uint8).reshape(256,256,3) for i in range(1280)]
rft_blocks = [np.frombuffer(base64.b64decode(rft_df['BLOCK'].iloc[i]), dtype=np.uint8).reshape(256,256,3) for i in range(1280)]
print("Done decoding")

for alpha in [0.747, 0.748]:
    beta = 1 - alpha
    print(f"\nBlending alpha={alpha}...")
    out_blocks = []
    for i in range(1280):
        fused = alpha * naf_blocks[i].astype(np.float64) + beta * rft_blocks[i].astype(np.float64)
        fused = np.clip(fused, 0, 255).round().astype(np.uint8)
        out_blocks.append(base64.b64encode(fused.tobytes()).decode('utf-8'))
    
    fname = f"SubmitSrgb_2model_fttta_{alpha:.3f}.csv"
    fpath = f"{SUB}/{fname}"
    df = pd.DataFrame({'ID': np.arange(1280), 'BLOCK': out_blocks})
    df.to_csv(fpath, index=False)
    size = os.path.getsize(fpath)/1e6
    
    # Validate
    assert len(out_blocks[0]) == 262144, f"Bad b64 length at alpha={alpha}"
    print(f"  ✅ Saved {fname} ({size:.1f} MB)")
    
    # Submit
    print(f"  Submitting to Kaggle...")
    result = subprocess.run(
        ['kaggle', 'competitions', 'submit', '-c', 'sidd-benchmark-srgb-psnr',
         '-f', fpath, '-m', f'Fine search: NAFTta+RFT-TTA alpha={alpha:.3f} (fttta)'],
        capture_output=True, text=True, timeout=180
    )
    if 'COMPLETE' in result.stdout or result.returncode == 0:
        print(f"  ✅ Submitted successfully!")
    else:
        print(f"  ❌ Submit failed: {result.stdout[:200]} {result.stderr[:200]}")
