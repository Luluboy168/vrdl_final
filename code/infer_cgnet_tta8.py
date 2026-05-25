#!/usr/bin/env python3
"""
CGNet SIDD inference - generates SubmitSrgb_cgnet_tta8.csv
Uses 8-way TTA (horizontal flip, vertical flip, 90/180/270 rotations)
"""
import os, sys, base64
import numpy as np
import torch
import scipy.io as sio
from pathlib import Path

PROJECT = '/home/luluboy/projects/vrdl_final'
CGNET_REPO = os.path.join(PROJECT, 'code', 'CGNet_repo')
CGNET_WEIGHT = os.path.join(PROJECT, 'weights', 'CGNet_SIDD.pth')
MAT_FILE = os.path.join(PROJECT, 'data', 'BenchmarkNoisyBlocksSrgb.mat')
OUT_CSV = os.path.join(PROJECT, 'submissions', 'SubmitSrgb_cgnet_tta8.csv')

sys.path.insert(0, os.path.join(CGNET_REPO))
from basicsr.models.archs.CGNet_arch import CascadedGaze

def build_cgnet():
    model = CascadedGaze(
        img_channel=3,
        width=60,
        enc_blk_nums=[2, 2, 4, 6],
        middle_blk_num=10,
        dec_blk_nums=[2, 2, 2, 2],
        GCE_CONVS_nums=[3, 3, 2, 2]
    )
    sd = torch.load(CGNET_WEIGHT, map_location='cpu', weights_only=False)
    if 'params' in sd:
        sd = sd['params']
    elif 'state_dict' in sd:
        sd = sd['state_dict']
    # Handle DDP wrapper
    if 'module' in list(sd.keys())[0]:
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'CGNet loaded. Missing={len(missing)}, Unexpected={len(unexpected)}')
    return model.cuda().eval()

def apply_tta8(img):
    """8-way TTA: identity, rot90/180/270, h-flip, v-flip, h-flip+rot90"""
    augs = [
        lambda x: x,
        lambda x: np.rot90(x, 1),
        lambda x: np.rot90(x, 2),
        lambda x: np.rot90(x, 3),
        lambda x: np.fliplr(x),
        lambda x: np.flipud(x),
        lambda x: np.rot90(np.fliplr(x), 1),
        lambda x: np.rot90(np.flipud(x), 1),
    ]
    deaugs = [
        lambda x: x,
        lambda x: np.rot90(x, -1),
        lambda x: np.rot90(x, -2),
        lambda x: np.rot90(x, -3),
        lambda x: np.fliplr(x),
        lambda x: np.flipud(x),
        lambda x: np.rot90(np.fliplr(x), -1),
        lambda x: np.rot90(np.flipud(x), -1),
    ]
    return augs, deaugs

def main():
    print('Building CGNet model...')
    model = build_cgnet()
    
    print(f'Loading test data from {MAT_FILE}...')
    mat = sio.loadmat(MAT_FILE)
    # The mat file has 'BenchmarkNoisyBlocks', shape (40, 32, 256, 256, 3), uint8
    inputs = mat['BenchmarkNoisyBlocksSrgb']
    print(f'inputs shape: {inputs.shape}, dtype: {inputs.dtype}')
    assert inputs.shape == (40, 32, 256, 256, 3), f'Unexpected shape: {inputs.shape}'
    
    augs, deaugs = apply_tta8(None)
    
    blocks_b64 = []
    total = 40 * 32
    count = 0
    
    print(f'Running CGNet inference on {total} blocks with 8-way TTA...')
    with torch.no_grad(), torch.cuda.amp.autocast():
        for i in range(40):
            row = []
            for j in range(32):
                noisy = inputs[i, j]  # (256, 256, 3), uint8
                
                # TTA: average 8 augmented versions
                preds = []
                for k, (aug, deaug) in enumerate(zip(augs, deaugs)):
                    aug_img = aug(noisy.copy()).astype(np.float32) / 255.0
                    aug_t = torch.from_numpy(aug_img.transpose(2, 0, 1)).float().cuda().unsqueeze(0)
                    pred = model(aug_t)
                    pred = pred.squeeze().cpu().numpy().transpose(1, 2, 0)
                    pred = deaug((pred * 255.0).clip(0, 255).astype(np.uint8))
                    preds.append(pred)
                
                # Average TTA predictions
                avg_pred = np.mean(preds, axis=0).astype(np.uint8)
                assert avg_pred.shape == (256, 256, 3) and avg_pred.dtype == np.uint8
                
                b64 = base64.b64encode(avg_pred.tobytes()).decode('utf-8')
                row.append(b64)
                count += 1
                
                if count % 128 == 0:
                    print(f'  Processed {count}/{total} blocks')
            
            row_b64 = []
            for b64 in row:
                row_b64.append(b64)
            blocks_b64.extend(row_b64)
    
    print(f'Creating submission CSV...')
    import pandas as pd
    df = pd.DataFrame()
    df['ID'] = np.arange(1280)
    df['BLOCK'] = blocks_b64
    
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f'Saved: {OUT_CSV}')
    
    # Validate
    df2 = pd.read_csv(OUT_CSV)
    assert list(df2.columns) == ['ID', 'BLOCK'], f'Wrong columns: {list(df2.columns)}'
    assert len(df2) == 1280, f'Wrong rows: {len(df2)}'
    assert all(len(b) == 262144 for b in df2['BLOCK']), f'Wrong b64 length!'
    print(f'✅ CSV valid: {len(df2)} rows, each b64={len(df2["BLOCK"].iloc[0])} chars')

if __name__ == '__main__':
    main()