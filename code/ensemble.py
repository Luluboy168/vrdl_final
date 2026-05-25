#!/usr/bin/env python3
"""
ensemble.py
Weighted average ensemble: NAFNet_TTA + Restormer (no TTA)
Grid search α ∈ {0.3, 0.4, 0.5, 0.6, 0.7}
Output: SubmitSrgb_ensemble_v1.csv (NAFNet TTA column format)
"""
import os
import sys
import numpy as np
import pandas as pd
import base64
import cv2
from tqdm import tqdm

PROJECT_DIR = '/home/luluboy/projects/vrdl_final'
SUBS_DIR    = os.path.join(PROJECT_DIR, 'submissions')

NAFNET_CSV = os.path.join(SUBS_DIR, 'SubmitSrgb_nafnet_tta8.csv')
RESTORMER_CSV = os.path.join(SUBS_DIR, 'SubmitSrgb_restormer.csv')
OUTPUT_DIR = os.path.join(SUBS_DIR, 'ensemble_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🔍 讀取 NAFNet TTA CSV...")
df_naf = pd.read_csv(NAFNET_CSV)
print(f"   NAFNet: {len(df_naf)} rows, columns: {list(df_naf.columns)}")
print(f"   ID range: {df_naf['ID'].min()} – {df_naf['ID'].max()}")

print("🔍 讀取 Restormer CSV...")
df_res = pd.read_csv(RESTORMER_CSV)
print(f"   Restormer: {len(df_res)} rows, columns: {list(df_res.columns)}")
print(f"   Id range: {df_res['Id'].min()} – {df_res['Id'].max()}")

# Build Restormer lookup: Id -> base64
res_lookup = dict(zip(df_res['Id'], df_res['Base64EncodedBlocks']))

def decode_base64_block(b64_str, model='nafnet'):
    """Decode base64 string to (256,256,3) uint8 numpy array."""
    raw = base64.b64decode(b64_str)
    if model == 'nafnet':
        # cv2.imencode format: numpy array serialised via cv2.imencode/png
        nparr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            # fallback: try as raw bytes (256*256*3)
            arr = np.frombuffer(raw[:256*256*3], np.uint8).reshape(256, 256, 3)
            img = arr
    else:
        # Restormer: raw bytes (256*256*3)
        arr = np.frombuffer(raw, np.uint8).reshape(256, 256, 3)
        img = arr
    return img

def encode_nafnet_base64(img):
    """Encode (256,256,3) uint8 as cv2.imencode base64 (matching NAFNet TTA format)."""
    # Encode as PNG via cv2.imencode
    _, enc = cv2.imencode('.png', img)
    return base64.b64encode(enc.tobytes()).decode('utf-8')

def array_to_base64_raw(img):
    """Raw bytes base64 (matching Restormer format)."""
    return base64.b64encode(img.tobytes()).decode('utf-8')

# Pre-decode all Restormer blocks into memory
print("📦 預解碼 Restormer blocks...")
res_blocks = {}
for idx, row in df_res.iterrows():
    block_id = row['Id']
    try:
        img = decode_base64_block(row['Base64EncodedBlocks'], model='restormer')
        res_blocks[block_id] = img
    except Exception as e:
        print(f"   ⚠️ Restormer block {block_id} decode error: {e}")
        res_blocks[block_id] = None

print(f"   解碼完成: {len(res_blocks)} blocks")

# Grid search over alpha
ALPHAS = [0.3, 0.4, 0.5, 0.6, 0.7]
best_alpha = None
best_count = 0

print("\n🔄 Grid search α...")
for alpha in ALPHAS:
    output_csv = os.path.join(OUTPUT_DIR, f'ensemble_alpha_{alpha}.csv')
    records = []
    
    for idx, row in tqdm(df_naf.iterrows(), total=len(df_naf), desc=f'α={alpha}'):
        block_id = row['ID']
        naf_b64 = row['BLOCK']
        
        # Decode NAFNet block
        try:
            naf_img = decode_base64_block(naf_b64, model='nafnet')
        except Exception as e:
            print(f"\n   ⚠️ NAFNet block {block_id} decode error: {e}")
            continue
        
        # Get Restormer block (1-based in Restormer CSV)
        res_id = block_id + 1  # Convert NAFNet 0-based to Restormer 1-based
        res_img = res_blocks.get(res_id)
        
        if res_img is None:
            print(f"\n   ⚠️ Missing Restormer block {res_id}, using NAFNet only")
            ensemble_img = naf_img
        else:
            # Weighted average
            naf_f = naf_img.astype(np.float32)
            res_f = res_img.astype(np.float32)
            ensemble_f = alpha * naf_f + (1 - alpha) * res_f
            ensemble_img = np.clip(ensemble_f, 0, 255).round().astype(np.uint8)
        
        # Encode in NAFNet TTA format
        b64 = encode_nafnet_base64(ensemble_img)
        records.append({'ID': block_id, 'BLOCK': b64})
    
    # Save
    df_out = pd.DataFrame(records)
    df_out.to_csv(output_csv, index=False)
    size_mb = os.path.getsize(output_csv) / 1e6
    print(f"   α={alpha}: {len(records)} blocks → {output_csv} ({size_mb:.1f} MB)")
    
    if len(records) > best_count:
        best_count = len(records)
        best_alpha = alpha

print(f"\n✅ Grid search 完成")
print(f"   最佳 α (by block count): {best_alpha} ({best_count} blocks)")

# Use alpha=0.5 as default (most balanced), produce final CSV
FINAL_ALPHA = 0.5
final_csv = os.path.join(SUBS_DIR, 'SubmitSrgb_ensemble_v1.csv')
src_csv = os.path.join(OUTPUT_DIR, f'ensemble_alpha_{FINAL_ALPHA}.csv')
import shutil
shutil.copy(src_csv, final_csv)
print(f"\n✅ 最終 ensemble CSV: {final_csv} (α={FINAL_ALPHA})")
print(f"   備份各 α 版本於: {OUTPUT_DIR}/")