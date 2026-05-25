#!/usr/bin/env python3
"""
train_restormer_finetune.py
Fine-tune Restormer (RealDenoising_Restormer.pth) on SIDD sRGB denoising.
"""

import os
import sys
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.cuda.amp as amp
import cv2
from tqdm import tqdm

# ===== Config =====
PROJECT_DIR    = '/home/luluboy/projects/vrdl_final'
CODE_DIR       = os.path.join(PROJECT_DIR, 'code')
RESTORMER_DIR  = os.path.join(CODE_DIR, 'Restormer')
WEIGHT_FILE    = os.path.join(PROJECT_DIR, 'weights', 'RealDenoising_Restormer.pth')
DATA_DIR       = os.path.join(PROJECT_DIR, 'data', 'SIDD_Medium_sRGB', 'Data')
LOG_DIR        = os.path.join(PROJECT_DIR, 'logs')
CKPT_DIR       = os.path.join(PROJECT_DIR, 'checkpoints')
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# Training hyperparams
CROP_SIZE      = 256
BATCH_SIZE     = 1           # Conservative: 1 for stability on RTX 3080 Ti 12GB
NUM_EPOCHS     = 5
LR             = 5e-5
NUM_WORKERS    = 2          # Reduce to save pinned memory
SEED           = 42
ACCUM_STEPS    = 8          # Effective batch = 1*8 = 8 (accumulate over 8 steps)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

# ===== Seed =====
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

# ===== Model =====
def build_model():
    sys.path.insert(0, os.path.join(RESTORMER_DIR, 'basicsr', 'models', 'archs'))
    from restormer_arch import Restormer

    model = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='BiasFree',
        dual_pixel_task=False,
    )

    print(f'Loading pretrained weights from {WEIGHT_FILE}')
    ckpt = torch.load(WEIGHT_FILE, map_location='cpu')
    if 'params' in ckpt:
        state_dict = ckpt['params']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f'Loaded. Missing keys: {len(missing)}, Unexpected: {len(unexpected)}')

    # Freeze norm / bias-free LN params (optional - don't freeze too much for fine-tune)
    # Actually for fine-tune, let everything train but use lower LR for pretrained params
    return model


# ===== Dataset =====
class SIDDDataset(Dataset):
    """
    Loads full-size PNG pairs from SIDD_Medium_sRGB.
    Each scene dir has: GT_SRGB_010.PNG, GT_SRGB_011.PNG, NOISY_SRGB_010.PNG, NOISY_SRGB_011.PNG
    Random 256x256 crops with flip/rotate augmentation on-the-fly.
    """
    def __init__(self, data_dir, crop_size=256, augment=True):
        self.crop_size = crop_size
        self.augment = augment

        # Collect all (gt_path, noisy_path) pairs
        self.pairs = []
        scene_dirs = sorted(glob.glob(os.path.join(data_dir, '*')))
        for scene_dir in scene_dirs:
            gt_files  = sorted(glob.glob(os.path.join(scene_dir, '*_GT_SRGB_*.PNG')))
            noisy_files = sorted(glob.glob(os.path.join(scene_dir, '*_NOISY_SRGB_*.PNG')))
            # Match by scene
            for gt in gt_files:
                # extract base to match noisy
                # e.g. 0001_GT_SRGB_010.PNG -> 0001_NOISY_SRGB_010.PNG
                basename = os.path.basename(gt)  # 0001_GT_SRGB_010.PNG
                parts = basename.split('_')
                prefix = parts[0]  # 0001
                suffix = '_'.join(parts[2:])  # SRGB_010.PNG
                noisy_name = basename.replace('GT_', 'NOISY_')
                noisy_path = os.path.join(scene_dir, noisy_name)
                if os.path.exists(noisy_path):
                    self.pairs.append((gt, noisy_path))

        print(f'Dataset: {len(self.pairs)} image pairs from {len(scene_dirs)} scenes')

    def __len__(self):
        return len(self.pairs) * 4  # multiple crops per image

    def _augment(self, gt, noisy):
        # Random horizontal flip
        if random.random() > 0.5:
            gt = cv2.flip(gt, 1)
            noisy = cv2.flip(noisy, 1)
        # Random vertical flip
        if random.random() > 0.5:
            gt = cv2.flip(gt, 0)
            noisy = cv2.flip(noisy, 0)
        # Random 90° rotation
        k = random.randint(0, 3)
        if k > 0:
            gt = np.rot90(gt, k)
            noisy = np.rot90(noisy, k)
        return gt, noisy

    def _random_crop(self, gt, noisy):
        h, w = gt.shape[:2]
        if h < self.crop_size or w < self.crop_size:
            # Upscale if needed
            scale = max(self.crop_size / h, self.crop_size / w) + 0.1
            gt = cv2.resize(gt, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
            noisy = cv2.resize(noisy, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
            h, w = gt.shape[:2]

        # Random crop coordinates
        top  = random.randint(0, h - self.crop_size)
        left = random.randint(0, w - self.crop_size)
        gt_crop   = gt[top:top+self.crop_size, left:left+self.crop_size]
        noisy_crop = noisy[top:top+self.crop_size, left:left+self.crop_size]
        return gt_crop, noisy_crop

    def __getitem__(self, idx):
        idx = idx % len(self.pairs)
        gt_path, noisy_path = self.pairs[idx]

        gt    = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        noisy = cv2.imread(noisy_path, cv2.IMREAD_COLOR)
        gt    = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)

        if self.augment:
            gt, noisy = self._augment(gt, noisy)

        gt, noisy = self._random_crop(gt, noisy)

        # to float [0,1]
        gt    = gt.astype(np.float32)    / 255.0
        noisy = noisy.astype(np.float32) / 255.0

        # CHW
        gt_t    = torch.from_numpy(gt).permute(2, 0, 1).float()
        noisy_t = torch.from_numpy(noisy).permute(2, 0, 1).float()

        return noisy_t, gt_t


# ===== Training =====
def main():
    model = build_model()
    model.to(DEVICE)

    dataset  = SIDDDataset(DATA_DIR, crop_size=CROP_SIZE, augment=True)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Loss
    criterion = nn.L1Loss()  # L1 is generally better for image restoration than MSE

    # Optimizer with different LR for pretrained vs new params
    # Since we load with strict=False and mostly match pretrained weights,
    # use one LR for all with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # AMP scaler
    scaler = amp.GradScaler()

    print(f'\nDataset: {len(dataset)} samples, {len(dataloader)} batches/epoch')
    print(f'Batch size: {BATCH_SIZE}, Accum steps: {ACCUM_STEPS}, Effective BS: {BATCH_SIZE * ACCUM_STEPS}')
    print(f'LR: {LR}, Epochs: {NUM_EPOCHS}')
    print(f'Starting training...\n')

    global_step = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')

        optimizer.zero_grad()
        for batch_idx, (noisy, gt) in enumerate(pbar):
            noisy = noisy.to(DEVICE)
            gt    = gt.to(DEVICE)

            with amp.autocast():
                pred = model(noisy)
                loss = criterion(pred, gt) / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (batch_idx + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * ACCUM_STEPS
            pbar.set_postfix({'loss': f'{loss.item() * ACCUM_STEPS:.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch+1} done. Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}')

        # Save checkpoint
        ckpt_path = os.path.join(CKPT_DIR, f'restormer_ft_epoch{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, ckpt_path)
        print(f'  -> Checkpoint saved: {ckpt_path}')

    # Save final
    final_path = os.path.join(CKPT_DIR, 'restormer_ft_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f'\nTraining complete. Final model: {final_path}')


if __name__ == '__main__':
    main()