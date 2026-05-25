#!/usr/bin/env python3
"""
Restormer-FT + Multi-Scale 8-way TTA inference.
Scales: 0.9x, 1.0x, 1.1x — each scale runs 8-way TTA, results resized back to 256x256 and averaged.
"""
import os, sys, gc, time
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import scipy.io
import numpy as np
import pandas as pd
import cv2
import base64
import torch.backends.cudnn as cudnn

DEVICE = torch.device('cuda')
print(f'[DEVICE] Using: {DEVICE}')

PROJECT_DIR   = '/home/luluboy/projects/vrdl_final'
CODE_DIR      = os.path.join(PROJECT_DIR, 'code')
RESTORMER_DIR = os.path.join(CODE_DIR, 'Restormer')
DATA_DIR      = os.path.join(PROJECT_DIR, 'data')
SUBS_DIR      = os.path.join(PROJECT_DIR, 'submissions')
CKPT_DIR      = os.path.join(PROJECT_DIR, 'checkpoints')

MAT_FILE    = os.path.join(DATA_DIR, 'BenchmarkNoisyBlocksSrgb.mat')
WEIGHT_FILE = os.path.join(CKPT_DIR, 'restormer_ft_final.pth')
OUTPUT_CSV  = os.path.join(SUBS_DIR, 'SubmitSrgb_restormer_ft_ms_tta.csv')

# Multi-scale factors
SCALES = [0.875, 1.0, 1.125]  # 224, 256, 288 — all divisible by 16 (fixes pixel_unshuffle dimension error at 0.9x=230)

# 8-way TTA transforms
def rot90(img): return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
def rot180(img): return cv2.rotate(img, cv2.ROTATE_180)
def rot270(img): return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
def hflip(img): return cv2.flip(img, 1)
def inv_rot90(img): return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
def inv_rot180(img): return cv2.rotate(img, cv2.ROTATE_180)
def inv_rot270(img): return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

TTA_VARIANTS = [
    (lambda x: x,                                   lambda x: x),
    (rot90,                                         inv_rot90),
    (rot180,                                        inv_rot180),
    (rot270,                                        inv_rot270),
    (hflip,                                         hflip),
    (lambda x: hflip(rot90(x)),                     lambda x: inv_rot90(hflip(x))),
    (lambda x: hflip(rot180(x)),                    lambda x: inv_rot180(hflip(x))),
    (lambda x: hflip(rot270(x)),                    lambda x: inv_rot270(hflip(x))),
]


def img2tensor(img):
    return torch.from_numpy(img.astype(np.float32)/255.0).permute(2,0,1).float()


def build_model():
    sys.path.insert(0, os.path.join(RESTORMER_DIR, 'basicsr/models/archs'))
    from restormer_arch import Restormer

    model = Restormer(
        inp_channels=3, out_channels=3, dim=48,
        num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type='BiasFree', dual_pixel_task=False,
    )

    print(f'Loading Restormer-FT weights from: {WEIGHT_FILE}')
    ckpt = torch.load(WEIGHT_FILE, map_location='cpu', weights_only=False)
    sd = ckpt.get('params', ckpt.get('state_dict', ckpt))
    model.load_state_dict(sd, strict=True)
    model.to(DEVICE)
    model.eval()
    print('Model loaded successfully (FP32)')
    return model


@torch.no_grad()
def infer_block_single_pass(model, block):
    """Single pass (no TTA) for speed check."""
    t = img2tensor(block).unsqueeze(0).to(DEVICE)
    with torch.amp.autocast('cuda'):
        out = model(t)
    out = out.float().squeeze(0).permute(1, 2, 0).cpu().numpy()
    del t, out
    torch.cuda.empty_cache()
    return (block.astype(np.float32) * 0 + 128).astype(np.uint8)  # dummy


@torch.no_grad()
def run_ms_tta_block(model, block):
    """
    Multi-scale 8-way TTA for one 256x256 block.
    At each scale: resize → 8-way TTA → resize back to 256x256 → accumulate
    """
    accum = np.zeros((256, 256, 3), dtype=np.float64)
    target_size = 256

    for scale in SCALES:
        # Resize input
        new_size = int(256 * scale)
        resized = cv2.resize(block, (new_size, new_size), interpolation=cv2.INTER_CUBIC)

        # 8-way TTA
        scale_accum = np.zeros((new_size, new_size, 3), dtype=np.float64)
        for fwd, inv in TTA_VARIANTS:
            xformed = fwd(resized)
            t = img2tensor(xformed).unsqueeze(0).to(DEVICE)
            with torch.amp.autocast('cuda'):
                out = model(t)
            out = out.float()
            out_np = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
            out_np = out_np * 255.0
            restored = inv(out_np.astype(np.float32))
            scale_accum += restored
            del t, out, out_np, xformed
            torch.cuda.empty_cache()

        scale_avg = scale_accum / len(TTA_VARIANTS)

        # Resize back to 256x256
        resized_back = cv2.resize(scale_avg, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        accum += resized_back

    # Average across scales
    avg = accum / len(SCALES)
    return np.clip(avg, 0, 255).round().astype(np.uint8)


def main():
    t_start = time.time()
    print('='*60)
    print('Restormer-FT + Multi-Scale 8-way TTA Inference')
    print(f'Scales: {SCALES}, TTA: {len(TTA_VARIANTS)}-way')
    print('='*60)

    if not os.path.exists(MAT_FILE):
        print(f'ERROR: {MAT_FILE} not found')
        sys.exit(1)
    if not os.path.exists(WEIGHT_FILE):
        print(f'ERROR: {WEIGHT_FILE} not found')
        sys.exit(1)

    model = build_model()

    # Sanity check
    print('Running sanity check...')
    test_block = np.random.randint(0, 255, (256,256,3), dtype=np.uint8)
    t = img2tensor(test_block).unsqueeze(0).to(DEVICE)
    with torch.amp.autocast('cuda'):
        out = model(t)
    out = out.float()
    print(f'  Sanity OK. Peak: {torch.cuda.max_memory_allocated()/1e9:.3f}GB')
    del t, out
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    # Load data
    print(f'Loading data from {MAT_FILE}...')
    mat = scipy.io.loadmat(MAT_FILE)
    keys = [k for k in mat.keys() if not k.startswith('__')]
    inputs = mat[keys[0]]
    n_i, n_j = inputs.shape[0], inputs.shape[1]
    n_blocks = n_i * n_j
    print(f'  Shape: {inputs.shape} -> {n_blocks} blocks')

    # Quick timing estimate with 5 blocks
    print('Estimating speed...')
    t0 = time.time()
    for ii in range(min(2, n_i)):
        for jj in range(min(2, n_j)):
            block = inputs[ii, jj, :, :, :]
            _ = run_ms_tta_block(model, block)
    t_estimate = time.time() - t0
    per_block = t_estimate / 4
    total_est = per_block * n_blocks
    print(f'  ~{per_block:.1f}s/block, estimated total: {total_est/60:.1f} min')
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    print(f'\nRunning multi-scale TTA inference on {n_blocks} blocks...')
    results = []

    for i in range(n_i):
        for j in range(n_j):
            block = inputs[i, j, :, :, :]
            denoised = run_ms_tta_block(model, block)
            b64 = base64.b64encode(denoised.tobytes()).decode('utf-8')
            block_id = i * n_j + j
            results.append({'ID': block_id, 'BLOCK': b64})
            gc.collect()

            if (block_id + 1) % 50 == 0 or block_id + 1 == n_blocks:
                elapsed = time.time() - t_start
                peak = torch.cuda.max_memory_allocated()/1e9
                eta = (elapsed / (block_id + 1)) * (n_blocks - block_id - 1)
                print(f'  [{block_id+1}/{n_blocks}] '
                      f'elapsed={elapsed/60:.1f}min eta={eta/60:.1f}min '
                      f'peak={peak:.2f}GB')

    os.makedirs(SUBS_DIR, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    size_mb = os.path.getsize(OUTPUT_CSV) / 1e6
    total_time = time.time() - t_start
    print(f'\n✅ Done! Saved {OUTPUT_CSV} ({size_mb:.1f} MB, {len(df)} rows)')
    print(f'   Total time: {total_time/60:.1f} min, Peak GPU: {torch.cuda.max_memory_allocated()/1e9:.3f} GB')

    # Quick format validation
    print('Validating format...')
    first_len = len(df.iloc[0]['BLOCK'])
    print(f'   First BLOCK length: {first_len} (expected 262144)')
    if first_len == 262144:
        print('   ✅ Format check PASSED')
    else:
        print(f'   ⚠️  WARNING: Expected 262144, got {first_len}')


if __name__ == '__main__':
    main()