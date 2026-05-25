#!/usr/bin/env python3
"""Validate ensemble PSNR on the validation set."""
import sys, os
sys.path.insert(0, '/home/luluboy/projects/vrdl_final/code')

import numpy as np
import scipy.io as sio
import torch
import cv2
import base64
import pandas as pd
from tqdm import tqdm

PYTHON = '/home/luluboy/miniconda3/bin/python3'

def psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return 100.0
    return 10 * np.log10(255**2 / mse)

# Load validation data
print("Loading data...")
noisy_mat = sio.loadmat('/home/luluboy/projects/vrdl_final/data/BenchmarkNoisyBlocksSrgb.mat')
gt_mat = sio.loadmat('/home/luluboy/projects/vrdl_final/data/ValidationGtBlocksSrgb.mat')

noisy = noisy_mat['BenchmarkNoisyBlocksSrgb']   # (40, 32, 256, 256, 3)
gt = gt_mat['ValidationGtBlocksSrgb']             # (40, 32, 256, 256, 3)
print(f"Noisy: {noisy.shape}, GT: {gt.shape}")

# Load NAFNet
print("Loading NAFNet...")
sys.path.insert(0, '/home/luluboy/projects/vrdl_final/code/NAFNet/basicsr')
from NAFNet import NAFNet
model_naf = NAFNet(img_channel=3, width=64, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1])
model_naf.load_state_dict(torch.load('/home/luluboy/projects/vrdl_final/weights/NAFNet-SIDD-width64.pth', map_location='cuda:0'))
model_naf.cuda().eval()

# Load Restormer
print("Loading Restormer...")
sys.path.insert(0, '/home/luluboy/projects/vrdl_final/code/Restormer')
from Restormer import Restormer
model_rest = Restormer(inp_channels=3)
model_rest.load_state_dict(torch.load('/home/luluboy/projects/vrdl_final/weights/RealDenoising_Restormer.pth', map_location='cuda:0'))
model_rest.cuda().eval()

def run_nafnet(block):
    with torch.no_grad():
        inp = torch.from_numpy(block).permute(2,0,1).unsqueeze(0).float().cuda() / 255.0
        out = model_naf(inp).squeeze(0).permute(1,2,0).clamp(0,1).cpu().numpy()
    return (out * 255).astype(np.uint8)

def run_restormer(block):
    with torch.no_grad():
        inp = torch.from_numpy(block).permute(2,0,1).unsqueeze(0).float().cuda() / 255.0
        out = model_rest(inp).squeeze(0).permute(1,2,0).clamp(0,1).cpu().numpy()
    return (out * 255).astype(np.uint8)

print("Running inference on 1280 blocks...")
results = []
for i in range(40):
    for j in range(32):
        noisy_block = noisy[i, j]  # (256, 256, 3) uint8
        gt_block = gt[i, j]         # (256, 256, 3) uint8

        # NAFNet
        denoised_naf = run_nafnet(noisy_block)
        psnr_naf = psnr(denoised_naf, gt_block)

        # Restormer
        denoised_rest = run_restormer(noisy_block)
        psnr_rest = psnr(denoised_rest, gt_block)

        # Ensemble v2: 0.7 NAFNet + 0.3 Restormer
        denoised_ens = (0.7 * denoised_naf.astype(float) + 0.3 * denoised_rest.astype(float)).astype(np.uint8)
        psnr_ens = psnr(denoised_ens, gt_block)

        results.append({'psnr_naf': psnr_naf, 'psnr_rest': psnr_rest, 'psnr_ens': psnr_ens})

        if (i*32+j+1) % 64 == 0:
            print(f"  [{i*32+j+1}/1280] NAFNet PSNR: {psnr_naf:.4f}, Restormer PSNR: {psnr_rest:.4f}, Ensemble PSNR: {psnr_ens:.4f}")

df = pd.DataFrame(results)
print(f"\n=== Validation Results ===")
print(f"NAFNet avg PSNR: {df['psnr_naf'].mean():.4f}")
print(f"Restormer avg PSNR: {df['psnr_rest'].mean():.4f}")
print(f"Ensemble v2 (0.7+0.3) avg PSNR: {df['psnr_ens'].mean():.4f}")

# Also try different alpha values
print("\n=== Alpha Grid Search ===")
for alpha in [0.5, 0.6, 0.7, 0.8]:
    psnr_alpha = (alpha * df['psnr_naf'] + (1-alpha) * df['psnr_rest']).mean()
    print(f"Alpha={alpha}: PSNR={psnr_alpha:.4f}")
