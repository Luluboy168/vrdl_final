#!/usr/bin/env python3
"""從 SIDD Medium sRGB 資料集擷取 training patches."""
import os, cv2, numpy as np, torch
from pathlib import Path

SRC_NOISY = '/home/luluboy/projects/vrdl_final/data/SIDD_Medium_sRGB/Data'
DST_INPUT = '/home/luluboy/projects/vrdl_final/code/Restormer/Denoising/Datasets/train/SIDD/input_crops'
DST_TARGET = '/home/luluboy/projects/vrdl_final/code/Restormer/Denoising/Datasets/train/SIDD/target_crops'
PATCH_SIZE = 128
NUM_PATCHES_PER_IMAGE = 20  # 每張圖片取 20 個 random patches
np.random.seed(42)

os.makedirs(DST_INPUT, exist_ok=True)
os.makedirs(DST_TARGET, exist_ok=True)

folders = sorted([f for f in os.listdir(SRC_NOISY) if os.path.isdir(os.path.join(SRC_NOISY, f))])
print(f'Found {len(folders)} scene folders')

patch_count = 0
for folder in folders:
    folder_path = os.path.join(SRC_NOISY, folder)
    files = sorted(os.listdir(folder_path))
    noisy_files = [f for f in files if 'NOISY' in f]
    gt_files = [f for f in files if 'GT' in f]

    if len(noisy_files) < 1 or len(gt_files) < 1:
        continue

    noisy_imgs = [cv2.imread(os.path.join(folder_path, f)) for f in noisy_files]
    gt_imgs = [cv2.imread(os.path.join(folder_path, f)) for f in gt_files]

    for ni, (noisy_img, gt_img) in enumerate(zip(noisy_imgs, gt_imgs)):
        noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        H, W = noisy_img.shape[:2]

        for pi in range(NUM_PATCHES_PER_IMAGE):
            if H > PATCH_SIZE and W > PATCH_SIZE:
                y = np.random.randint(0, H - PATCH_SIZE)
                x = np.random.randint(0, W - PATCH_SIZE)

                noisy_patch = noisy_img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                gt_patch = gt_img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

                patch_name = f'{patch_count:06d}_ni{ni}.png'
                cv2.imwrite(os.path.join(DST_INPUT, patch_name), cv2.cvtColor(noisy_patch, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(DST_TARGET, patch_name), cv2.cvtColor(gt_patch, cv2.COLOR_RGB2BGR))
                patch_count += 1

print(f'Created {patch_count} training patches in {DST_INPUT}')
print(f'Training patches available: {len(os.listdir(DST_INPUT))} input, {len(os.listdir(DST_TARGET))} target')