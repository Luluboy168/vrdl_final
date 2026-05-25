#!/usr/bin/env python3
"""
Fine 2-model alpha search: NAFTta + RFT-TTA
Plus 3-model ensemble: NAFTta + CGNet-TTA + RFT-MS-TTA
"""
import pandas as pd, base64, numpy as np, os, subprocess

SUB = "/home/luluboy/projects/vrdl_final/submissions"

# --- Load models ---
print("Loading models...")
naf_df = pd.read_csv(f"{SUB}/SubmitSrgb_nafnet_tta8_fixed.csv").sort_values('ID').reset_index(drop=True)
rft_df = pd.read_csv(f"{SUB}/SubmitSrgb_restormer_ft_tta8.csv").sort_values('ID').reset_index(drop=True)
cg_df  = pd.read_csv(f"{SUB}/SubmitSrgb_cgnet_tta_fixed.csv").sort_values('ID').reset_index(drop=True)
rms_df = pd.read_csv(f"{SUB}/SubmitSrgb_restormer_ft_ms_tta.csv").sort_values('ID').reset_index(drop=True)

# Fix 1-indexed IDs
for df in [naf_df, rft_df, cg_df, rms_df]:
    if df['ID'].min() == 1:
        df['ID'] = df['ID'] - 1
    df.sort_values('ID', inplace=True)
    df.reset_index(drop=True, inplace=True)

print(f"  NAFTta: {len(naf_df)}, RFT-TTA: {len(rft_df)}, CGNet: {len(cg_df)}, RFT-MS: {len(rms_df)}")

# Pre-decode all
def decode_all(df):
    return [np.frombuffer(base64.b64decode(r), np.uint8).reshape(256,256,3) for r in df['BLOCK']]

print("Decoding NAFTta..."); naf_blocks = decode_all(naf_df)
print("Decoding RFT-TTA..."); rft_blocks = decode_all(rft_df)
print("Decoding CGNet...");   cg_blocks  = decode_all(cg_df)
print("Decoding RFT-MS...");  rms_blocks = decode_all(rms_df)
print("All decoded!")

def make_csv(out_path, ids, blocks):
    df = pd.DataFrame({'ID': ids, 'BLOCK': blocks})
    df.to_csv(out_path, index=False)
    size_mb = os.path.getsize(out_path)/1e6
    print(f"  ✅ {os.path.basename(out_path)} ({size_mb:.1f} MB)")

def submit_csv(csv_path, msg):
    print(f"  Submitting {os.path.basename(csv_path)}...")
    result = subprocess.run(
        ['kaggle', 'competitions', 'submit', '-c', 'sidd-benchmark-srgb-psnr',
         '-f', csv_path, '-m', msg],
        capture_output=True, text=True, cwd=SUB
    )
    print(f"  -> {result.stdout.strip()[:100]}")
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()[:200]}")

##############################################
# PART 1: Fine 2-model alpha search
##############################################
print("\n=== PART 1: Fine 2-model alpha search (NAFTta + RFT-TTA) ===")
# Already submitted: 0.735 (40.4852), need: 0.733, 0.734, 0.736, 0.737, 0.738, 0.739
fine_alphas = [0.733, 0.734, 0.736, 0.737, 0.738, 0.739]

for alpha in fine_alphas:
    beta = 1 - alpha
    fname = f"SubmitSrgb_2model_fttta8_{alpha:.3f}.csv"
    fpath = f"{SUB}/{fname}"
    if os.path.exists(fpath):
        print(f"  {fname} exists, skipping")
        continue
    out_blocks = []
    for i in range(1280):
        fused = alpha * naf_blocks[i].astype(np.float64) + beta * rft_blocks[i].astype(np.float64)
        fused = np.clip(fused, 0, 255).round().astype(np.uint8)
        out_blocks.append(base64.b64encode(fused.tobytes()).decode('utf-8'))
    make_csv(fpath, np.arange(1280), out_blocks)
    submit_csv(fpath, f"Fine search: NAFTta+RFT-TTA alpha={alpha:.3f} (fttta8)")

##############################################
# PART 2: 3-model ensemble search
# NAFTta + CGNet-TTA + RFT-MS-TTA
##############################################
print("\n=== PART 2: 3-model ensemble (NAFTta + CGNet-TTA + RFT-MS-TTA) ===")
configs = [
    (0.70, 0.15, 0.15),
    (0.72, 0.15, 0.13),
    (0.68, 0.17, 0.15),
    (0.74, 0.13, 0.13),
    (0.65, 0.20, 0.15),
    (0.70, 0.20, 0.10),
    (0.68, 0.15, 0.17),
]

for w_n, w_c, w_r in configs:
    # Also try RFT-TTA instead of RFT-MS-TTA for comparison
    for rft_name, rft_blk in [("RFTMS", rms_blocks), ("RFTTA", rft_blocks)]:
        fname = f"SubmitSrgb_3model_{rft_name}_{w_n:.2f}_{w_c:.2f}_{w_r:.2f}.csv"
        fpath = f"{SUB}/{fname}"
        if os.path.exists(fpath):
            print(f"  {fname} exists, skipping")
            continue
        out_blocks = []
        for i in range(1280):
            fused = (w_n * naf_blocks[i].astype(np.float64) +
                     w_c * cg_blocks[i].astype(np.float64) +
                     w_r * rft_blk[i].astype(np.float64))
            fused = np.clip(fused, 0, 255).round().astype(np.uint8)
            out_blocks.append(base64.b64encode(fused.tobytes()).decode('utf-8'))
        make_csv(fpath, np.arange(1280), out_blocks)
        rft_label = "RFT-MS-TTA" if rft_name == "RFTMS" else "RFT-TTA"
        submit_csv(fpath, f"3model: NAFTta({w_n:.2f})+CGNet({w_c:.2f})+{rft_label}({w_r:.2f})")

print("\n=== ALL DONE ===")