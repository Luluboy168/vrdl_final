"""
Fine alpha search for NAFTta + RFT-TTA ensemble (using NEW Restormer-FT TTA8).
Finer search around optimal region 0.735-0.740.
"""
import numpy as np, pandas as pd, base64, os, sys

SUB = "/home/luluboy/projects/vrdl_final/submissions"

naf_df = pd.read_csv(os.path.join(SUB, "SubmitSrgb_nafnet_tta8_fixed.csv"))
rft_df = pd.read_csv(os.path.join(SUB, "SubmitSrgb_restormer_ft_tta8.csv"))

# Fix 1-indexed IDs
for df in [naf_df, rft_df]:
    if df['ID'].min() == 1:
        df['ID'] = df['ID'] - 1

naf_df = naf_df.sort_values('ID').reset_index(drop=True)
rft_df = rft_df.sort_values('ID').reset_index(drop=True)

# Pre-decode
print("Decoding blocks...")
naf_blocks = [np.frombuffer(base64.b64decode(r), np.uint8).reshape(256,256,3) 
              for r in naf_df['BLOCK']]
rft_blocks = [np.frombuffer(base64.b64decode(r), np.uint8).reshape(256,256,3) 
              for r in rft_df['BLOCK']]
print("Done.")

# Finer alphas around known good range
alphas = [0.735, 0.736, 0.737, 0.738, 0.739]

for alpha in alphas:
    beta = 1 - alpha
    output_blocks = []
    for i in range(1280):
        fused = alpha * naf_blocks[i].astype(np.float64) + beta * rft_blocks[i].astype(np.float64)
        fused = np.clip(fused, 0, 255).round().astype(np.uint8)
        b64 = base64.b64encode(fused.tobytes()).decode('utf-8')
        output_blocks.append(b64)
    df = pd.DataFrame({'ID': np.arange(1280), 'BLOCK': output_blocks})
    fname = f"SubmitSrgb_2model_fttta8_{alpha:.3f}.csv"
    df.to_csv(os.path.join(SUB, fname), index=False)
    print(f"  alpha={alpha:.3f}: {fname}")

print("Done!")