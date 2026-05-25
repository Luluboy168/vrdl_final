#!/usr/bin/env python3
"""
MIRNetv2 batch inference — faster than sequential by processing multiple blocks at once.
"""
import os, sys, torch, scipy.io, numpy as np, pandas as pd, base64, time

PROJECT_DIR = '/home/luluboy/projects/vrdl_final'
CODE_DIR    = os.path.join(PROJECT_DIR, 'code', 'MIRNetv2')
DATA_DIR    = os.path.join(PROJECT_DIR, 'data')
WEIGHTS_DIR = os.path.join(PROJECT_DIR, 'weights')
SUBS_DIR    = os.path.join(PROJECT_DIR, 'submissions')

MAT_FILE    = os.path.join(DATA_DIR, 'BenchmarkNoisyBlocksSrgb.mat')
WEIGHT_FILE = os.path.join(WEIGHTS_DIR, 'real_denoising.pth')
OUTPUT_CSV  = os.path.join(SUBS_DIR, 'SubmitSrgb_mirnetv2.csv')

DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16   # adjust if OOM (3080Ti 12GB should be fine with 16)

print(f'🚀 Device: {DEVICE}, batch_size: {BATCH_SIZE}')

# Build model
sys.path.insert(0, os.path.join(CODE_DIR, 'basicsr', 'models', 'archs'))
from mirnet_v2_arch import MIRNet_v2

model = MIRNet_v2(
    inp_channels=3, out_channels=3, n_feat=80, chan_factor=1.5,
    n_RRG=4, n_MRB=2, height=3, width=2, scale=1, bias=False, task='real_denoising'
)
ckpt = torch.load(WEIGHT_FILE, map_location='cpu')
state_dict = ckpt['params'] if 'params' in ckpt else ckpt
model.load_state_dict(state_dict, strict=True)
model.to(DEVICE)
model.eval()
print('✅ Model loaded')

# Load mat
print(f'📂 Loading {MAT_FILE}...')
mat = scipy.io.loadmat(MAT_FILE)
keys = [k for k in mat.keys() if not k.startswith('__')]
noisy_blocks = mat[keys[0]]  # (40, 32, 256, 256, 3), uint8
rows, cols = noisy_blocks.shape[0], noisy_blocks.shape[1]
total = rows * cols
print(f'   {rows}×{cols}={total} blocks')

def img2tensor(img):
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).float()

def tensor2img(tensor):
    out = tensor.cpu().clamp(0, 1).mul(255).round().byte()
    return out.permute(1, 2, 0).numpy()

def array_to_base64(x):
    return base64.b64encode(x.tobytes()).decode('utf-8')

# Batch inference
records = []
n_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

print(f'🔄 Starting batch inference ({n_batches} batches of size {BATCH_SIZE})...')
t_start = time.time()

for batch_idx in range(n_batches):
    start_idx = batch_idx * BATCH_SIZE
    end_idx   = min(start_idx + BATCH_SIZE, total)
    batch_i = [(start_idx + k) // cols for k in range(end_idx - start_idx)]
    batch_j = [(start_idx + k) %  cols for k in range(end_idx - start_idx)]

    # Stack batch: (B, 3, 256, 256), [0,1]
    imgs = [noisy_blocks[bi, bj] for bi, bj in zip(batch_i, batch_j)]
    batch_t = torch.stack([img2tensor(img) for img in imgs]).to(DEVICE)

    with torch.no_grad():
        out_batch = model(batch_t)  # (B, 3, 256, 256)

    for k in range(end_idx - start_idx):
        block_id = start_idx + k
        out_img = tensor2img(out_batch[k])
        out_img = np.clip(out_img, 0, 255).astype(np.uint8)
        records.append({'ID': block_id, 'BLOCK': array_to_base64(out_img)})

    elapsed = time.time() - t_start
    pct = (block_id + 1) / total * 100
    eta = elapsed / (block_id + 1) * (total - block_id - 1)
    print(f'   [{block_id+1}/{total}] {pct:.1f}% | elapsed {elapsed:.1f}s, ETA {eta:.1f}s')

t_total = time.time() - t_start
print(f'\n💾 Writing CSV ({len(records)} rows)...')
df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)
size_mb = os.path.getsize(OUTPUT_CSV) / 1e6
print(f'✅ Done! {t_total:.1f}s total, CSV {size_mb:.1f} MB')

# Quick validation
df2 = pd.read_csv(OUTPUT_CSV)
assert list(df2.columns) == ['ID', 'BLOCK']
assert len(df2) == 1280
assert all(len(b) == 262144 for b in df2['BLOCK'])
print('✅ Validation passed — ready for Kaggle submit!')