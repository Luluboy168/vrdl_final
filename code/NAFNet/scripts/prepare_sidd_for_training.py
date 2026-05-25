#!/usr/bin/env python3
"""
Prepare SIDD Medium sRGB for NAFNet fine-tuning.
Creates paired folder structure: train/gt/, train/lq/, val/gt/, val/lq/
Each with matching filenames for PairedImageDataset (folder backend).

Usage: python prepare_sidd_for_training.py
"""
import os
import shutil
import random
from pathlib import Path

# Paths
DATA_ROOT = Path("/home/luluboy/projects/vrdl_final/data/SIDD_Medium_sRGB/Data")
OUTPUT_ROOT = Path("/home/luluboy/projects/vrdl_final/code/NAFNet/datasets/SIDD")
TRAIN_GT = OUTPUT_ROOT / "train" / "gt"
TRAIN_LQ = OUTPUT_ROOT / "train" / "lq"
VAL_GT = OUTPUT_ROOT / "val" / "gt"
VAL_LQ = OUTPUT_ROOT / "val" / "lq"

SPLIT_RATIO = 0.9  # 90% train, 10% val

def prepare():
    # Create directories
    for d in [TRAIN_GT, TRAIN_LQ, VAL_GT, VAL_LQ]:
        d.mkdir(parents=True, exist_ok=True)
        print(f"Created {d}")

    # Scan all scene instances
    scenes = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir()])
    print(f"Found {len(scenes)} scene instances")

    # Each scene has 2 GT and 2 NOISY images
    pairs = []
    for scene in scenes:
        gt_files = sorted([f for f in scene.glob("*_GT_SRGB_*.PNG")])
        noisy_files = sorted([f for f in scene.glob("*_NOISY_SRGB_*.PNG")])
        if len(gt_files) == 2 and len(noisy_files) == 2:
            # Use first pair per scene (both GT and NOISY have 2 images)
            pairs.append((gt_files[0], noisy_files[0]))
            pairs.append((gt_files[1], noisy_files[1]))
        else:
            print(f"  WARNING: {scene.name} has {len(gt_files)} GT, {len(noisy_files)} NOISY — SKIP")

    print(f"Total pairs: {len(pairs)}")

    # Shuffle and split
    random.seed(42)
    random.shuffle(pairs)
    n_train = int(len(pairs) * SPLIT_RATIO)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]

    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    # Copy train pairs
    idx = 0
    for gt_path, lq_path in train_pairs:
        gt_dst = TRAIN_GT / f"{idx:05d}.PNG"
        lq_dst = TRAIN_LQ / f"{idx:05d}.PNG"
        shutil.copy2(gt_path, gt_dst)
        shutil.copy2(lq_path, lq_dst)
        idx += 1

    # Copy val pairs
    idx = 0
    for gt_path, lq_path in val_pairs:
        gt_dst = VAL_GT / f"{idx:05d}.PNG"
        lq_dst = VAL_LQ / f"{idx:05d}.PNG"
        shutil.copy2(gt_path, gt_dst)
        shutil.copy2(lq_path, lq_dst)
        idx += 1

    print(f"\n✅ Done! Train: {len(list(TRAIN_GT.glob('*.PNG')))} pairs, Val: {len(list(VAL_GT.glob('*.PNG')))} pairs")
    print(f"Train GT: {TRAIN_GT}")
    print(f"Train LQ: {TRAIN_LQ}")
    print(f"Val GT: {VAL_GT}")
    print(f"Val LQ: {VAL_LQ}")

if __name__ == "__main__":
    prepare()