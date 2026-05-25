#!/usr/bin/env python3
"""
Fix CGNet TTA CSV: PNG-encoded blocks → raw bytes base64
Result: SubmitSrgb_cgnet_tta_fixed.csv
"""
import pandas as pd, base64, numpy as np, cv2, sys

SUBS = '/home/luluboy/projects/vrdl_final/submissions'
src  = f'{SUBS}/SubmitSrgb_cgnet_tta.csv'
dst  = f'{SUBS}/SubmitSrgb_cgnet_tta_fixed.csv'

print(f"Reading {src}...")
df = pd.read_csv(src)
print(f"  rows={len(df)}, columns={list(df.columns)}")

# Ensure 0-indexed
if df['ID'].min() == 1:
    df['ID'] -= 1
    print("  Adjusted ID: 1-indexed → 0-indexed")

# Convert each PNG-encoded block to raw bytes base64
print("Converting PNG → raw bytes...")
new_blocks = []
for i, b64 in enumerate(df['BLOCK']):
    img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Block {i} failed to decode as PNG")
    assert img.shape == (256, 256, 3), f"Block {i} wrong shape: {img.shape}"
    assert img.dtype == np.uint8
    new_blocks.append(base64.b64encode(img.tobytes()).decode('utf-8'))
    if i % 200 == 0:
        print(f"  processed {i}/1280...")

df['BLOCK'] = new_blocks

# Validate
print("Validating...")
assert list(df.columns) == ['ID', 'BLOCK']
assert df['ID'].tolist() == list(range(1280))
first_len = len(df['BLOCK'].iloc[0])
print(f"  first block len={first_len} (expected 262144)")
assert first_len == 262144, f"Unexpected block length: {first_len}"
assert all(len(b) == 262144 for b in df['BLOCK']), "Inconsistent block sizes!"

# Save
df.to_csv(dst, index=False)
size_mb = __import__('os').path.getsize(dst) / 1e6
print(f"Saved {dst} ({size_mb:.1f} MB)")
print("✅ CGNet fix complete!")