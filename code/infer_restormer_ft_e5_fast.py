#!/usr/bin/env python3
"""Quick Restormer-FT Epoch5 inference (no TTA) for ensemble testing."""
import os, sys, torch, scipy.io, numpy as np, pandas as pd, base64

PROJECT_DIR = '/home/luluboy/projects/vrdl_final'
CODE_DIR = os.path.join(PROJECT_DIR, 'code')
RESTORMER_DIR = os.path.join(CODE_DIR, 'Restormer')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
CKPT_DIR = os.path.join(PROJECT_DIR, 'checkpoints')
SUBS_DIR = os.path.join(PROJECT_DIR, 'submissions')

MAT_FILE = os.path.join(DATA_DIR, 'BenchmarkNoisyBlocksSrgb.mat')
WEIGHT_FILE = os.path.join(CKPT_DIR, 'restormer_ft_epoch5.pth')
OUTPUT_CSV = os.path.join(SUBS_DIR, 'SubmitSrgb_restormer_ft_epoch5.csv')

DEVICE = 'cuda'
print(f'[DEVICE] {DEVICE}')
print(f'Loading weights: {WEIGHT_FILE}')

sys.path.insert(0, os.path.join(RESTORMER_DIR, 'basicsr/models/archs'))
from restormer_arch import Restormer
model = Restormer(inp_channels=3, out_channels=3, dim=48,
    num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
    heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
    bias=False, LayerNorm_type='BiasFree', dual_pixel_task=False)

ckpt = torch.load(WEIGHT_FILE, map_location='cpu')
state_dict = ckpt.get('params', ckpt.get('state_dict', ckpt))
model.load_state_dict(state_dict, strict=False)
model.to(DEVICE)
model.eval()
print('Model loaded OK')

mat = scipy.io.loadmat(MAT_FILE)
keys = [k for k in mat.keys() if not k.startswith('__')]
blocks = mat[keys[0]]
print(f'Blocks shape: {blocks.shape}')

def img2tensor(img):
    return torch.from_numpy(img.astype(np.float32)/255.0).permute(2,0,1).float()

def tensor2img(t):
    return t.cpu().clamp(0,1).mul(255).round().byte().permute(1,2,0).numpy()

records = []
total = blocks.shape[0] * blocks.shape[1]
for i in range(blocks.shape[0]):
    for j in range(blocks.shape[1]):
        bid = i * blocks.shape[1] + j
        img = blocks[i, j]
        t = img2tensor(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(t)
        out_img = np.clip(tensor2img(out[0]), 0, 255).astype(np.uint8)
        records.append({'ID': bid, 'BLOCK': base64.b64encode(out_img.tobytes()).decode()})
        if (bid+1) % 200 == 0 or bid == total-1:
            print(f'  [{bid+1}/{total}] done')

df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)
print(f'Saved: {OUTPUT_CSV} ({os.path.getsize(OUTPUT_CSV)/1e6:.1f} MB)')
assert len(df['BLOCK'].iloc[0]) == 262144
print('EXIT_CODE:0')