#!/usr/bin/env python3
"""
SwinIR inference on SIDD Benchmark sRGB test blocks.
Generates SubmitSrgb_swinir.csv for use in 3-model ensemble.
"""
import os, sys, base64, time
import numpy as np
import torch
import scipy.io

PROJECT_DIR = '/home/luluboy/projects/vrdl_final'
SWINIR_REPO = os.path.join(PROJECT_DIR, 'swinir_repo')
SWINIR_MODEL = os.path.join(PROJECT_DIR, 'swinir_model', 'swinir_dn.pth')
DATA_FILE = os.path.join(PROJECT_DIR, 'data', 'BenchmarkNoisyBlocksSrgb.mat')
OUT_CSV = os.path.join(PROJECT_DIR, 'submissions', 'SubmitSrgb_swinir.csv')

# Add swinir repo to path
sys.path.insert(0, SWINIR_REPO)
from models.network_swinir import SwinIR

def build_model():
    model = SwinIR(
        img_size=64,  # 64 for classical denoising
        patch_size=1,
        in_chans=3,
        embed_dim=96,
        depths=[6, 6, 6, 6],
        num_heads=[6, 6, 6, 6],
        window_size=8,
        mlp_ratio=4.,
        upsampler='pixel unshuffle',  # for real image denoising
        resi_connection='1conv'
    )
    
    print(f'Loading SwinIR weights from {SWINIR_MODEL}')
    state_dict = torch.load(SWINIR_MODEL, map_location='cpu', weights_only=True)
    
    # Handle different state_dict formats
    if 'params' in state_dict:
        state_dict = state_dict['params']
    if 'module' in str(list(state_dict.keys())[0]):
        new_sd = {}
        for k, v in state_dict.items():
            new_sd[k[7:]] = v
        state_dict = new_sd
    
    model.load_state_dict(state_dict, strict=False)
    print('✅ SwinIR model loaded')
    return model

def img2tensor(img):
    """HWC uint8 → CHW float [0,1]"""
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img

def tensor2img(t):
    """CHW float [0,1] → HWC uint8 [0,255]"""
    out = t.detach().cpu().clamp(0, 1).mul(255).round().byte()
    return out.permute(1, 2, 0).numpy()

def denoise_block(model, block, device):
    """Denoise a single 256x256x3 block using SwinIR with pixel unshuffle."""
    img_t = img2tensor(block).unsqueeze(0).to(device)  # (1,3,256,256)
    
    with torch.no_grad():
        # SwinIR with upsampler='pixel unshuffle' expects img_size parameter
        # Use window_size=8 for 256x256 with 64 img_size
        # Actually for 256x256 with window_size=8: H=32, W=32 windows
        # This should fit in GPU
        out = model(img_t)
    
    return tensor2img(out[0])

def main():
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    # Load model
    model = build_model().to(device)
    model.eval()
    
    # Load data
    print(f'Loading {DATA_FILE}...')
    mat = scipy.io.loadmat(DATA_FILE)
    inputs = mat['BenchmarkNoisyBlocksSrgb']
    print(f'  Shape: {inputs.shape}, dtype: {inputs.dtype}')
    n_i, n_j = inputs.shape[0], inputs.shape[1]
    
    # Process all blocks
    results = []
    total = n_i * n_j
    print(f'Processing {total} blocks...')
    
    for i in range(n_i):
        for j in range(n_j):
            block = inputs[i, j, :, :, :]  # (256,256,3) uint8
            denoised = denoise_block(model, block, device)
            
            # Convert to raw bytes base64 (NOT PNG)
            b64 = base64.b64encode(denoised.tobytes()).decode('utf-8')
            block_id = i * n_j + j
            results.append({'ID': block_id, 'BLOCK': b64})
            
            if block_id % 200 == 0:
                elapsed = time.time() - start_time
                print(f'  [{block_id}/{total}] elapsed={elapsed:.0f}s')
    
    # Save CSV
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)
    
    elapsed = time.time() - start_time
    size_mb = os.path.getsize(OUT_CSV) / 1e6
    print(f'\n✅ Saved {OUT_CSV}')
    print(f'   {len(df)} rows, {size_mb:.1f} MB')
    print(f'   Total time: {elapsed:.0f}s')
    
    # Quick validation
    df_check = pd.read_csv(OUT_CSV)
    assert list(df_check.columns) == ['ID', 'BLOCK']
    assert df_check['ID'].tolist() == list(range(1280))
    assert all(len(b) == 262144 for b in df_check['BLOCK'])
    print('   ✅ Format validation passed')

if __name__ == '__main__':
    main()