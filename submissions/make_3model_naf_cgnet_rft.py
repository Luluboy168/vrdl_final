#!/usr/bin/env python3
"""
3-model ensemble: NAFTta + CGNet-TTA + RFT-TTA8
Blend: w1*NAFTta + w2*CGNet-TTA + w3*RFT-TTA8 (float32)
Clip to [0,255], round to uint8, base64 encode.
"""
import base64, pandas as pd, numpy as np

BASE = '/home/luluboy/projects/vrdl_final/submissions'
OUT  = BASE

# Load CSVs
print("Loading CSVs...")
df_naftta  = pd.read_csv(f'{BASE}/SubmitSrgb_nafnet_tta8_fixed.csv')
df_cgnet   = pd.read_csv(f'{BASE}/SubmitSrgb_cgnet_tta_fixed.csv')
df_rft     = pd.read_csv(f'{BASE}/SubmitSrgb_restormer_ft_tta8.csv')

print(f"  NAFTta:  {len(df_naftta)} rows")
print(f"  CGNet:   {len(df_cgnet)} rows")
print(f"  RFT-TTA: {len(df_rft)} rows")

# Verify ID alignment
assert (df_naftta.ID == df_cgnet.ID).all()
assert (df_naftta.ID == df_rft.ID).all()
print("  IDs aligned ✓")

# Decode each block to uint8 numpy arrays
def decode_blocks(series):
    """Decode base64 strings to uint8 arrays (H*W*C)."""
    raw = [base64.b64decode(s) for s in series]
    arr = np.array([np.frombuffer(b, dtype=np.uint8) for b in raw], dtype=np.uint8)
    return arr

print("Decoding blocks...")
blocks_naftta = decode_blocks(df_naftta.BLOCK)   # (1280, 196608)
blocks_cgnet  = decode_blocks(df_cgnet.BLOCK)
blocks_rft    = decode_blocks(df_rft.BLOCK)

print(f"  Shapes: NAFTta={blocks_naftta.shape}, CGNet={blocks_cgnet.shape}, RFT={blocks_rft.shape}")

# Weight combos
WEIGHTS = [
    (0.72, 0.18, 0.10, '0.72_0.18_0.10'),
    (0.70, 0.20, 0.10, '0.70_0.20_0.10'),
    (0.74, 0.16, 0.10, '0.74_0.16_0.10'),
]

def make_ensemble(w1, w2, w3, label):
    print(f"\n=== Ensembling: w1={w1}, w2={w2}, w3={w3} ===")
    # Convert to float32
    e_naftta = blocks_naftta.astype(np.float32)
    e_cgnet  = blocks_cgnet.astype(np.float32)
    e_rft    = blocks_rft.astype(np.float32)

    # Blend
    blended = w1 * e_naftta + w2 * e_cgnet + w3 * e_rft

    # Clip + round + uint8
    blended = np.clip(blended, 0, 255)
    blended = np.round(blended).astype(np.uint8)

    print(f"  Blended range: [{blended.min()}, {blended.max()}]")

    # Encode back to base64
    print("  Encoding to base64...")
    block_strings = [
        base64.b64encode(row.tobytes()).decode('utf-8')
        for row in blended
    ]

    # Build output DF
    out_df = pd.DataFrame()
    out_df['ID'] = df_naftta.ID.values
    out_df['BLOCK'] = block_strings

    # Validate before saving
    n_invalid = sum(1 for s in out_df.BLOCK if len(s) != 262144)
    print(f"  Invalid blocks (len != 262144): {n_invalid}")

    fname = f'{OUT}/SubmitSrgb_3model_ncr_{label}.csv'
    out_df.to_csv(fname, index=False)
    print(f"  Saved: {fname}")

    # Quick validation
    df_check = pd.read_csv(fname)
    assert len(df_check) == 1280, f"Row count mismatch: {len(df_check)}"
    assert list(df_check.columns) == ['ID', 'BLOCK'], f"Column mismatch: {list(df_check.columns)}"
    assert df_check.ID.min() == 0 and df_check.ID.max() == 1279, "ID range error"
    sample_len = len(df_check.BLOCK.iloc[0])
    assert sample_len == 262144, f"Block length mismatch: {sample_len}"
    print(f"  Validation ✓  (rows={len(df_check)}, IDs={df_check.ID.min()}-{df_check.ID.max()}, block_len={sample_len})")

    return fname

saved = []
for w1, w2, w3, label in WEIGHTS:
    fname = make_ensemble(w1, w2, w3, label)
    saved.append((w1, w2, w3, fname))

print("\n\n=== All done! ===")
print("Generated files:")
for w1, w2, w3, fname in saved:
    print(f"  {fname}")