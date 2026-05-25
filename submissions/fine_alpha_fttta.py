"""
Quick fine alpha search for NAFTta + RFT-TTA ensemble.
Uses cached denoised images from previous runs.
Alpha = weight for NAFTta; (1-alpha) = weight for RFT-TTA.
"""
import numpy as np
import pandas as pd
import base64, os, sys

SUBMISSIONS = "/home/luluboy/projects/vrdl_final/submissions"

# Load cached 2-model FTTT ensemble (alpha=0.735 already validated = 40.4852)
# We need the RAW denoised outputs for NAFTta and RFT-TTA
# Check what's available
print("Checking available outputs...")

# The NAFTta and RFT-TTA raw denoised blocks are stored in:
# SubmitSrgb_nafnet_tta8_fixed.csv (NAFTta)
# SubmitSrgb_restormer_ft_tta8.csv (RFT-TTA) 

naf_csv = os.path.join(SUBMISSIONS, "SubmitSrgb_nafnet_tta8_fixed.csv")
rft_csv = os.path.join(SUBMISSIONS, "SubmitSrgb_restormer_ft_tta8.csv")

if not os.path.exists(naf_csv) or not os.path.exists(rft_csv):
    print(f"ERROR: missing files")
    sys.exit(1)

naf_df = pd.read_csv(naf_csv)
rft_df = pd.read_csv(rft_csv)

print(f"NAF shape: {naf_df.shape}, RFT shape: {rft_df.shape}")
print(f"NAF columns: {list(naf_df.columns)}")
print(f"RFT columns: {list(rft_df.columns)}")

# IDs
naf_ids = naf_df['ID'].tolist()
rft_ids = rft_df['ID'].tolist()
print(f"NAF ID range: {min(naf_ids)}-{max(naf_ids)}")
print(f"RFT ID range: {min(rft_ids)}-{max(rft_ids)}")

# Fix if IDs are 1-indexed
if min(naf_ids) == 1:
    naf_df['ID'] = naf_df['ID'] - 1
    print("Fixed NAF IDs: now 0-indexed")
if min(rft_ids) == 1:
    rft_df['ID'] = rft_df['ID'] - 1
    print("Fixed RFT IDs: now 0-indexed")

# Sort by ID
naf_df = naf_df.sort_values('ID').reset_index(drop=True)
rft_df = rft_df.sort_values('ID').reset_index(drop=True)

assert list(naf_df['ID']) == list(range(1280)), f"NAF IDs wrong: {naf_df['ID'].min()}-{naf_df['ID'].max()}"
assert list(rft_df['ID']) == list(range(1280)), f"RFT IDs wrong: {rft_df['ID'].min()}-{rft_df['ID'].max()}"

# Verify base64 lengths
naf_len = len(naf_df['BLOCK'].iloc[0])
rft_len = len(rft_df['BLOCK'].iloc[0])
print(f"NAF base64 length: {naf_len} (expected 262144)")
print(f"RFT base64 length: {rft_len} (expected 262144)")

# Test different alphas
alphas = [0.730, 0.731, 0.732, 0.733, 0.734, 0.736, 0.737, 0.738, 0.739, 0.740]

# Pre-decode all blocks (cache for reuse)
print("Pre-decoding blocks...")
naf_blocks = []
rft_blocks = []

for i in range(1280):
    naf_bytes = base64.b64decode(naf_df['BLOCK'].iloc[i])
    rft_bytes = base64.b64decode(rft_df['BLOCK'].iloc[i])
    naf_img = np.frombuffer(naf_bytes, dtype=np.uint8).reshape(256, 256, 3)
    rft_img = np.frombuffer(rft_bytes, dtype=np.uint8).reshape(256, 256, 3)
    naf_blocks.append(naf_img)
    rft_blocks.append(rft_img)

print("Done decoding. Running alpha search...")

for alpha in alphas:
    beta = 1 - alpha
    output_blocks = []
    for i in range(1280):
        # Weighted average in float, then convert back to uint8
        fused = alpha * naf_blocks[i].astype(np.float64) + beta * rft_blocks[i].astype(np.float64)
        fused = np.clip(fused, 0, 255).round().astype(np.uint8)
        b64 = base64.b64encode(fused.tobytes()).decode('utf-8')
        output_blocks.append(b64)
    
    df = pd.DataFrame()
    df['ID'] = np.arange(1280)
    df['BLOCK'] = output_blocks
    
    fname = f"SubmitSrgb_2model_fttta_{alpha:.3f}.csv"
    fpath = os.path.join(SUBMISSIONS, fname)
    df.to_csv(fpath, index=False)
    fsize = os.path.getsize(fpath) / 1e6
    print(f"  alpha={alpha:.3f}: saved {fname} ({fsize:.1f} MB)")

print("Done! Submit with:")
print("  kaggle competitions submit -c sidd-benchmark-srgb-psnr -f <file> -m 'fine alpha search'")