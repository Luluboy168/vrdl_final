#!/usr/bin/env python3
"""
train_restormer_finetune2.py
Resume fine-tuning from restormer_ft_final.pth (epoch 5)
Continue for 15 total epochs (10 more)
"""
import os, sys, glob, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.cuda.amp as amp
import cv2
from tqdm import tqdm

PROJECT_DIR   = '/home/luluboy/projects/vrdl_final'
CODE_DIR      = os.path.join(PROJECT_DIR, 'code')
RESTORMER_DIR = os.path.join(CODE_DIR, 'Restormer')
START_EPOCH   = 6        # Resume from epoch 6 (restormer_ft_final.pth was after epoch 5)
NUM_EPOCHS    = 15        # Total epochs target
WEIGHT_FILE   = os.path.join(PROJECT_DIR, 'weights', 'restormer_ft_final.pth')  # resume from ft5
DATA_DIR      = os.path.join(PROJECT_DIR, 'data', 'SIDD_Medium_sRGB', 'Data')
CKPT_DIR      = os.path.join(PROJECT_DIR, 'checkpoints')
LOG_DIR       = os.path.join(PROJECT_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

CROP_SIZE   = 256
BATCH_SIZE  = 1
LR          = 2e-5        # Lower LR for continuation
NUM_WORKERS = 2
SEED        = 42
ACCUM_STEPS = 8
DEVICE = 'cuda'
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

def build_model():
    sys.path.insert(0, os.path.join(RESTORMER_DIR, 'basicsr', 'models', 'archs'))
    from restormer_arch import Restormer
    model = Restormer(inp_channels=3, out_channels=3, dim=48,
                      num_blocks=[4,6,6,8], num_refinement_blocks=4,
                      heads=[1,2,4,8], ffn_expansion_factor=2.66,
                      bias=False, LayerNorm_type='BiasFree', dual_pixel_task=False)
    print(f'Loading from {WEIGHT_FILE}')
    ckpt = torch.load(WEIGHT_FILE, map_location='cpu')
    sd = ckpt.get('params', ckpt.get('state_dict', ckpt))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'Missing: {len(missing)}, Unexpected: {len(unexpected)}')
    return model

class SIDDDataset(Dataset):
    def __init__(self, data_dir, crop_size=256, augment=True):
        self.crop_size, self.augment = crop_size, augment
        self.pairs = []
        for scene in sorted(glob.glob(os.path.join(data_dir, '*'))):
            gt_files  = sorted(glob.glob(os.path.join(scene, '*GT*SRGB*.png')) + glob.glob(os.path.join(scene, '*GT*SRGB*.PNG')))
            noisy_files = []
            for gf in gt_files:
                nid = gf.replace('GT_SRGB', 'NOISY_SRGB').replace('gt_srgb', 'noisy_srgb')
                if os.path.exists(nid): noisy_files.append(nid)
            for nf, gf in zip(noisy_files, gt_files[:len(noisy_files)]):
                self.pairs.append((nf, gf))
        print(f'Dataset: {len(self.pairs)} pairs from {data_dir}')
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        noisy_path, gt_path = self.pairs[idx]
        noisy = cv2.imread(noisy_path); gt = cv2.imread(gt_path)
        noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
        gt    = cv2.cvtColor(gt,    cv2.COLOR_BGR2RGB)
        h, w = noisy.shape[:2]
        if h < self.crop_size or w < self.crop_size:
            noisy = cv2.resize(noisy, (self.crop_size, self.crop_size))
            gt    = cv2.resize(gt,    (self.crop_size, self.crop_size))
            h, w  = self.crop_size, self.crop_size
        y, x = random.randint(0,h-self.crop_size), random.randint(0,w-self.crop_size)
        n_crop = noisy[y:y+self.crop_size, x:x+self.crop_size]
        g_crop = gt[y:y+self.crop_size, x:x+self.crop_size]
        if self.augment:
            for _ in range(random.randint(0,3)):
                k = random.choice([0,1,2,3])
                if k: n_crop = np.rot90(n_crop, k); g_crop = np.rot90(g_crop, k)
            if random.random() > 0.5:
                n_crop = np.fliplr(n_crop).copy(); g_crop = np.fliplr(g_crop).copy()
        n_t = torch.from_numpy(n_crop.astype(np.float32)/255.0).permute(2,0,1)
        g_t = torch.from_numpy(g_crop.astype(np.float32)/255.0).permute(2,0,1)
        return n_t, g_t

model = build_model().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS-START_EPOCH+1, eta_min=1e-7)
criterion = nn.L1Loss()
scaler = amp.GradScaler()
dataset = SIDDDataset(DATA_DIR, CROP_SIZE, augment=True)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

print(f'Training from epoch {START_EPOCH} to {NUM_EPOCHS}')
for epoch in range(START_EPOCH, NUM_EPOCHS+1):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f'Epoch {epoch}/{NUM_EPOCHS}')
    for i, (noisy, gt) in enumerate(pbar):
        noisy, gt = noisy.to(DEVICE), gt.to(DEVICE)
        with amp.autocast():
            pred = model(noisy)
            loss = criterion(pred, gt) / ACCUM_STEPS
        scaler.scale(loss).backward()
        if (i+1) % ACCUM_STEPS == 0:
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
        total_loss += loss.item() * ACCUM_STEPS
        pbar.set_postfix({'loss': f'{loss.item()*ACCUM_STEPS:.4f}'})
    scheduler.step()
    avg_loss = total_loss / len(loader)
    ckpt_path = os.path.join(CKPT_DIR, f'restormer_ft2_epoch{epoch}.pth')
    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'loss': avg_loss}, ckpt_path)
    print(f'Epoch {epoch} done. Avg loss: {avg_loss:.4f}. LR: {scheduler.get_last_lr()[0]:.2e}. Saved to {ckpt_path}')

final_path = os.path.join(CKPT_DIR, 'restormer_ft2_final.pth')
torch.save({'epoch': NUM_EPOCHS, 'state_dict': model.state_dict(), 'loss': avg_loss}, final_path)
print(f'Fine-tuning complete! Final: {final_path}')
