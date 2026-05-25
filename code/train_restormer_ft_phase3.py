#!/usr/bin/env python3
"""Phase 3 Restormer Fine-tuning: 5 more epochs from restormer_ft_epoch10.pth"""
import os, sys, glob, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.cuda.amp as amp
import cv2
from tqdm import tqdm

PROJECT_DIR   = '/home/luluboy/projects/vrdl_final'
CODE_DIR      = os.path.join(PROJECT_DIR, 'code')
RESTORMER_DIR = os.path.join(CODE_DIR, 'Restormer')

# Start from epoch 10 checkpoint (latest)
START_EPOCH  = 11
NUM_EPOCHS   = 15
WEIGHT_FILE  = os.path.join(PROJECT_DIR, 'checkpoints', 'restormer_ft_epoch10.pth')
DATA_DIR     = os.path.join(PROJECT_DIR, 'data', 'SIDD_Medium_sRGB', 'Data')
CKPT_DIR     = os.path.join(PROJECT_DIR, 'checkpoints')
LOG_DIR      = os.path.join(PROJECT_DIR, 'submissions', 'restormer_ft_phase3.log')

os.makedirs(CKPT_DIR, exist_ok=True)

CROP_SIZE   = 256
BATCH_SIZE  = 1
LR          = 1e-5      # even lower LR for Phase3 continuation
NUM_WORKERS = 2
SEED        = 42
ACCUM_STEPS = 8         # effective batch = 8
DEVICE      = 'cuda'

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

def build_model():
    sys.path.insert(0, os.path.join(RESTORMER_DIR, 'basicsr', 'models', 'archs'))
    from restormer_arch import Restormer
    model = Restormer(
        inp_channels=3, out_channels=3, dim=48,
        num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type='BiasFree', dual_pixel_task=False
    )
    print(f'Loading from {WEIGHT_FILE}')
    ckpt = torch.load(WEIGHT_FILE, map_location='cpu')
    sd = ckpt.get('params', ckpt.get('state_dict', ckpt))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'  Missing: {len(missing)}, Unexpected: {len(unexpected)}')
    return model

class SIDDDataset(Dataset):
    def __init__(self, data_dir, crop_size=256, augment=True):
        self.crop_size, self.augment = crop_size, augment
        self.pairs = []
        for scene in sorted(glob.glob(os.path.join(data_dir, '*'))):
            gt_files    = sorted(glob.glob(os.path.join(scene, '*GT*SRGB*.png'))
                               + glob.glob(os.path.join(scene, '*GT*SRGB*.PNG')))
            noisy_files = []
            for gf in gt_files:
                nid = gf.replace('GT_SRGB', 'NOISY_SRGB').replace('gt_srgb', 'noisy_srgb')
                if os.path.exists(nid):
                    noisy_files.append(nid)
            for nf, gf in zip(noisy_files, gt_files[:len(noisy_files)]):
                self.pairs.append((nf, gf))
        print(f'Dataset: {len(self.pairs)} pairs')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, gt_path = self.pairs[idx]
        noisy = cv2.imread(noisy_path); gt = cv2.imread(gt_path)
        noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
        gt    = cv2.cvtColor(gt,    cv2.COLOR_BGR2RGB)

        h, w = noisy.shape[:2]
        if h > self.crop_size or w > self.crop_size:
            top  = random.randint(0, h - self.crop_size)
            left = random.randint(0, w - self.crop_size)
            noisy = noisy[top:top+self.crop_size, left:left+self.crop_size]
            gt    = gt[   top:top+self.crop_size, left:left+self.crop_size]

        if self.augment:
            if random.random() < 0.5:
                noisy = np.flip(noisy, axis=1).copy()
                gt    = np.flip(gt,    axis=1).copy()
            if random.random() < 0.5:
                noisy = np.flip(noisy, axis=0).copy()
                gt    = np.flip(gt,    axis=0).copy()
            if random.random() < 0.5:
                noisy = np.rot90(noisy, k=random.randint(1,3)).copy()
                gt    = np.rot90(gt,    k=random.randint(1,3)).copy()

        noisy = torch.from_numpy(noisy.astype(np.float32) / 255.0).permute(2, 0, 1)
        gt    = torch.from_numpy(gt.astype(np.float32)    / 255.0).permute(2, 0, 1)
        return noisy, gt

def main():
    print(f"[Restormer Phase3] Start epoch {START_EPOCH} → {NUM_EPOCHS}, LR={LR}")
    print(f"[Restormer Phase3] Weight: {WEIGHT_FILE}")

    model = build_model().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler    = amp.GradScaler()

    # Resume from START_EPOCH-1
    resume_epoch = START_EPOCH - 1
    if resume_epoch >= 1:
        resume_path = os.path.join(CKPT_DIR, f'restormer_ft_epoch{resume_epoch}.pth')
        if os.path.exists(resume_path):
            ckpt = torch.load(resume_path, map_location='cpu')
            sd   = ckpt.get('params', ckpt.get('state_dict', ckpt))
            model.load_state_dict(sd, strict=False)
            opt_ckpt = os.path.join(CKPT_DIR, f'optimizer_epoch{resume_epoch}.pth')
            if os.path.exists(opt_ckpt):
                optimizer.load_state_dict(torch.load(opt_ckpt, map_location='cpu'))
            sched_ckpt = os.path.join(CKPT_DIR, f'scheduler_epoch{resume_epoch}.pth')
            if os.path.exists(sched_ckpt):
                scheduler.load_state_dict(torch.load(sched_ckpt, map_location='cpu'))
            print(f'  Resumed from epoch {resume_epoch}')

    train_dataset = SIDDDataset(DATA_DIR, crop_size=CROP_SIZE)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    for epoch in range(START_EPOCH, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f'Epoch {epoch}/{NUM_EPOCHS}')

        for batch_idx, (noisy, gt) in pbar:
            noisy, gt = noisy.to(DEVICE), gt.to(DEVICE)
            with amp.autocast():
                pred   = model(noisy)
                loss   = nn.L1Loss(reduction='mean')(pred, gt)
                loss_v = loss.item() / ACCUM_STEPS

            scaler.scale(loss / ACCUM_STEPS).backward()

            if (batch_idx + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss_v * ACCUM_STEPS
            pbar.set_postfix({'loss': f'{loss_v * ACCUM_STEPS:.5f}'})

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        print(f'  Epoch {epoch}: avg_loss={avg_loss:.5f}, lr={scheduler.get_last_lr()[0]:.2e}')

        # Save checkpoint
        ckpt_path = os.path.join(CKPT_DIR, f'restormer_ft_epoch{epoch}.pth')
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, ckpt_path)
        torch.save(optimizer.state_dict(), os.path.join(CKPT_DIR, f'optimizer_epoch{epoch}.pth'))
        torch.save(scheduler.state_dict(), os.path.join(CKPT_DIR, f'scheduler_epoch{epoch}.pth'))
        print(f'  Saved: {ckpt_path}')

    print('[Restormer Phase3] Done!')

if __name__ == '__main__':
    main()