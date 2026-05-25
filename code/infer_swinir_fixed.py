#!/usr/bin/env python3
"""
Fixed SwinIR inference on SIDD Benchmark sRGB test blocks.
Uses correct color_dn config: img_size=128, embed_dim=180, depths=[6,6,6,6,6,6].
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
LOG_FILE = os.path.join(PROJECT_DIR, 'swinir_fixed_inference.log')

sys.path.insert(0, SWINIR_REPO)
from models.network_swinir import SwinIR

def build_model():
    # Correct color_dn config matching swinir_dn.pth
    model = SwinIR(
        img_size=128,
        patch_size=1,
        in_chans=3,
        embed_dim=180,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6],
        window_size=8,
        mlp_ratio=2.,
        upsampler='',  # empty upsampler for color_dn
        resi_connection='1conv'
    )
    
    print(f'Loading SwinIR weights from {SWINIR_MODEL}')
    state_dict = torch.load(SWINIR_MODEL, map_location='cpu', weights_only=True)
    
    if 'params' in state_dict:
        state_dict = state_dict['params']
    if 'module' in str(list(state_dict.keys())[0]):
        new_sd = {}
        for k, v in state_dict.items():
            new_sd[k[7:]] = v
        state_dict = new_sd
    
    # Load with strict=False to allow missing/unexpected keys
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'  Missing keys (ignored): {len(missing)}')
    if unexpected:
        print(f'  Unexpected keys (ignored): {len(unexpected)}')
    print('SwinIR model loaded successfully')
    return model

def img2tensor(img):
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img

def tensor2img(t):
    out = t.detach().cpu().clamp(0, 1).mul(255).round().byte()
    return out.permute(1, 2, 0).numpy()

def denoise_block(model, block, device):
    img_t = img2tensor(block).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_t)
    return tensor2img(out[0])

def main():
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    model = build_model().to(device)
    model.eval()
    
    print(f'Loading {DATA_FILE}...')
    mat = scipy.io.loadmat(DATA_FILE)
    inputs = mat['BenchmarkNoisyBlocksSrgb']
    print(f'  Shape: {inputs.shape}, dtype: {inputs.dtype}')
    n_i, n_j = inputs.shape[0], inputs.shape[1]
    
    total = n_i * n_j
    print(f'Processing {total} blocks...')
    
    results = []
    for i in range(n_i):
        for j in range(n_j):
            block = inputs[i, j, :, :, :]
            denoised = denoise_block(model, block, device)
            results.append(denoised)
            
            block_idx = i * n_j + j + 1
            if block_idx % 100 == 0:
                elapsed = time.time() - start_time
                print(f'  Processed {block_idx}/{total} ({elapsed:.1f}s)')
    
    print(f'Done inference. Total time: {time.time()-start_time:.1f}s')
    
    # Build submission CSV
    print('Building submission CSV...')
    output_blocks_base64string = []
    for img in results:
        assert img.shape == (256, 256, 3) and img.dtype == np.uint8
        b64 = base64.b64encode(img.tobytes()).decode('utf-8')
        output_blocks_base64string.append(b64)
    
    import pandas as pd
    df = pd.DataFrame()
    df['ID'] = np.arange(1280)
    df['BLOCK'] = output_blocks_base64string
    df.to_csv(OUT_CSV, index=False)
    
    # Validate
    print('Validating CSV...')
    df2 = pd.read_csv(OUT_CSV)
    assert list(df2.columns) == ['ID', 'BLOCK']
    assert df2['ID'].tolist() == list(range(1280))
    all_ok = all(len(b) == 262144 for b in df2['BLOCK'])
    print(f'  Columns OK: {list(df2.columns) == ["ID","BLOCK"]}')
    print(f'  IDs OK: {df2["ID"].tolist() == list(range(1280))}')
    print(f'  Block sizes OK: {all_ok}')
    print(f'  File size: {os.path.getsize(OUT_CSV)/1e6:.1f} MB')
    
    # Verify first/last base64 length
    for idx in [0, 639, 1279]:
        print(f'  Block {idx} b64 len: {len(df2["BLOCK"].iloc[idx])}')
    
    print(f'Output written to {OUT_CSV}')
    print(f'Total time: {time.time()-start_time:.1f}s')

if __name__ == '__main__':
    main()