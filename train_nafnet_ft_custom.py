#!/usr/bin/env python3
"""
train_nafnet_ft_custom.py
Fine-tune NAFNet-SIDD on SIDD Medium sRGB for a few epochs.
Quick custom script - no basicsr framework needed.
"""

import os, sys, glob, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.cuda.amp as amp
import cv2
from tqdm import tqdm

PROJECT_DIR = '/home/luluboy/projects/vrdl_final'
CODE_DIR    = os.path.join(PROJECT_DIR, 'code', 'NAFNet')
DATA_DIR    = os.path.join(PROJECT_DIR, 'data', 'SIDD_Medium_sRGB', 'Data')
WEIGHT_FILE = os.path.join(PROJECT_DIR, 'weights', 'NAFNet-SIDD-width64.pth')
CKPT_DIR    = os.path.join(PROJECT_DIR, 'checkpoints')
LOG_DIR     = os.path.join(PROJECT_DIR, 'logs')

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Hyperparams
CROP_SIZE   = 256
BATCH_SIZE  = 4
NUM_EPOCHS  = 3       # Just 3 epochs for quick improvement
LR          = 1e-4
NUM_WORKERS = 4
SEED        = 42
ACCUM_STEPS = 2       # Effective batch = 4*2 = 8

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

# ===== Seed =====
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

# ===== Model =====
class LayerNorm2d(nn.Module):
    def forward(self, x):
        u = x.mean(dim=1, keepdim=True)
        s = (x - u).square().mean(dim=1, keepdim=True).sqrt() + 1e-2
        return (x - u) / s

sys.path.insert(0, os.path.join(CODE_DIR))
from basicsr.models.archs.NAFNet_arch import NAFNet, SimpleGate

def build_model():
    model = NAFNet(
        img_channel=3,
        width=64,
        enc_blk_nums=[2, 2, 4, 8],
        middle_blk_num=12,
        dec_blk_nums=[2, 2, 2, 2],
    )
    ckpt = torch.load(WEIGHT_FILE, map_location='cpu', weights_only=False)
    if 'params' in ckpt:
        sd = ckpt['params']
    elif 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    else:
        sd = ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'Loaded NAFNet-SIDD (params). Missing={len(missing)}, Unexpected={len(unexpected)}')
    return model.to(DEVICE)

# ===== Dataset =====
class SIDD_SRGB_Dataset(Dataset):
    def __init__(self, data_dir, gt_suffix='GT_SRGB', noisy_suffix='NOISY_SRGB'):
        # Files are like: <scene>/<id>_GT_SRGB_XXX.png (with scene number prefix)
        self.gt_images = sorted(glob.glob(os.path.join(data_dir, '*/*' + gt_suffix + '_*.png')))
        self.noisy_images = sorted(glob.glob(os.path.join(data_dir, '*/*' + noisy_suffix + '_*.png')))
        assert len(self.gt_images) == len(self.noisy_images), \
            f'Mismatch: {len(self.gt_images)} GT vs {len(self.noisy_images)} NOISY'
        print(f'SIDD Dataset: {len(self.gt_images)} pairs')
    
    def __len__(self):
        return len(self.gt_images)
    
    def augment(self, gt, noisy):
        # Random flip/rotate
        if random.random() > 0.5:
            gt = cv2.flip(gt, 0); noisy = cv2.flip(noisy, 0)
        if random.random() > 0.5:
            gt = cv2.flip(gt, 1); noisy = cv2.flip(noisy, 1)
        k = random.randint(0, 3)
        if k:
            gt = np.rot90(gt, k); noisy = np.rot90(noisy, k)
        return gt, noisy
    
    def __getitem__(self, idx):
        gt_path = self.gt_images[idx]
        noisy_path = self.noisy_images[idx]
        
        gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        noisy = cv2.imread(noisy_path, cv2.IMREAD_COLOR)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
        
        gt, noisy = self.augment(gt, noisy)
        
        # Random crop
        h, w = gt.shape[:2]
        if h > CROP_SIZE and w > CROP_SIZE:
            y = random.randint(0, h - CROP_SIZE)
            x = random.randint(0, w - CROP_SIZE)
            gt = gt[y:y+CROP_SIZE, x:x+CROP_SIZE]
            noisy = noisy[y:y+CROP_SIZE, x:x+CROP_SIZE]
        
        gt = gt.astype(np.float32) / 255.0
        noisy = noisy.astype(np.float32) / 255.0
        
        gt_t = torch.from_numpy(gt.transpose(2, 0, 1)).float()
        noisy_t = torch.from_numpy(noisy.transpose(2, 0, 1)).float()
        return noisy_t, gt_t

# ===== Metrics =====
def calc_psnr(img1, img2):
    mse = ((img1 - img2) ** 2).mean()
    return 20 * np.log10(1.0 / np.sqrt(mse))

# ===== Main training =====
print('Building model...')
model = build_model()

print('Loading dataset...')
train_ds = SIDD_SRGB_Dataset(DATA_DIR)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS * len(train_loader))
scaler = amp.GradScaler()
criterion = nn.L1Loss()

print(f'Starting fine-tuning: {NUM_EPOCHS} epochs, {len(train_loader)} batches/epoch')
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
    
    optimizer.zero_grad()
    for batch_idx, (noisy, gt) in enumerate(pbar):
        noisy, gt = noisy.to(DEVICE), gt.to(DEVICE)
        
        with amp.autocast():
            pred = model(noisy)
            loss = criterion(pred, gt) / ACCUM_STEPS
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        
        total_loss += loss.item() * ACCUM_STEPS
        pbar.set_postfix({'loss': f'{loss.item()*ACCUM_STEPS:.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})
    
    avg_loss = total_loss / len(train_loader)
    elapsed = time.time() - start_time
    print(f'Epoch {epoch+1}: avg_loss={avg_loss:.4f}, elapsed={elapsed/60:.1f}min')
    
    # Save checkpoint per epoch
    ckpt_path = os.path.join(CKPT_DIR, f'nafnet_ft_epoch{epoch+1}.pth')
    torch.save(model.state_dict(), ckpt_path)
    print(f'  Saved: {ckpt_path}')

# Save final
final_path = os.path.join(CKPT_DIR, 'nafnet_ft_final.pth')
torch.save(model.state_dict(), final_path)
print(f'Fine-tuning complete! Final: {final_path}')
print(f'Total time: {(time.time()-start_time)/60:.1f} minutes')