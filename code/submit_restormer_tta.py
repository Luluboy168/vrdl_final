#!/usr/bin/env python3
"""Monitor Restormer TTA completion + auto-validate + auto-submit Kaggle."""

import subprocess, sys, time, os, base64, numpy as np, pandas as pd

BASE = "/home/luluboy/projects/vrdl_final/submissions"

def wait_for_csv(filename, timeout_min=20):
    path = os.path.join(BASE, filename)
    start = time.time()
    while time.time() - start < timeout_min * 60:
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"[OK] {filename} exists ({size:,} bytes)", flush=True)
            return path
        print(f"[waiting...] {filename} not found yet, retrying in 15s", flush=True)
        time.sleep(15)
    raise TimeoutError(f"{filename} did not appear within {timeout_min} min")

def validate_csv(path):
    """Validate format: ID=0-1279, BLOCK len=262144, shape (1280,2)."""
    df = pd.read_csv(path)
    cols = list(df.columns)
    print(f"  Columns: {cols}", flush=True)
    assert cols == ['ID', 'BLOCK'], f"Expected ['ID','BLOCK'], got {cols}"
    ids = df['ID'].tolist()
    assert ids == list(range(1280)), f"ID range wrong: {min(ids)}-{max(ids)}"
    b64_sample = df['BLOCK'].iloc[0]
    b64_len = len(b64_sample)
    print(f"  First BLOCK len: {b64_len}", flush=True)
    assert b64_len == 262144, f"Expected 262144, got {b64_len}"
    print(f"  ✅ Format OK", flush=True)
    return True

def kaggle_submit(csv_path, message):
    cmd = [
        "kaggle", "competitions", "submit",
        "-c", "sidd-benchmark-srgb-psnr",
        "-f", csv_path,
        "-m", message
    ]
    print(f"[SUBMIT] {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"[SUBMIT stdout]: {result.stdout}", flush=True)
    if result.stderr:
        print(f"[SUBMIT stderr]: {result.stderr}", flush=True)
    if result.returncode != 0:
        print(f"[ERROR] Submit failed with code {result.returncode}", flush=True)
        return False
    print("  ✅ Submit OK", flush=True)
    return True

def kaggle_submissions():
    cmd = ["kaggle", "competitions", "submissions", "sidd-benchmark-srgb-psnr"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("[SUBMISSIONS]:\n" + result.stdout, flush=True)

# ── Restormer TTA v2 (vrdl_restormer_tta_v2 session) ──
print("=== Restormer TTA v2 ===", flush=True)
v2_csv = wait_for_csv("SubmitSrgb_restormer_tta8.csv", timeout_min=20)
validate_csv(v2_csv)
# Rename to something meaningful
out_csv = os.path.join(BASE, "SubmitSrgb_restormer_tta8_v2.csv")
os.rename(v2_csv, out_csv)
print(f"[RENAMED] to SubmitSrgb_restormer_tta8_v2.csv", flush=True)
# Verify rename
if os.path.exists(out_csv):
    validate_csv(out_csv)
    kaggle_submit(out_csv, "Restormer + 8-way TTA v2 (batch_size=1, 串行 augmentation)")
else:
    print("[WARN] rename failed, submitting original path", flush=True)
    kaggle_submit(v2_csv, "Restormer + 8-way TTA v2")

# ── also check vrdl_tta session ──
print("\n=== Restormer TTA vrdl_tta ===", flush=True)
tta_csv = wait_for_csv("SubmitSrgb_restormer_tta.csv", timeout_min=5)
if tta_csv:
    validate_csv(tta_csv)
    kaggle_submit(tta_csv, "Restormer + 4-way TTA")

print("\n=== Done ===", flush=True)
kaggle_submissions()