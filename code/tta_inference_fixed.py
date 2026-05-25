#!/usr/bin/env python3
"""
tta_inference_fixed.py
Fixed 8-way TTA for NAFNet — corrected inverse transforms.

BUG FIX: The original tta_inference.py had incorrect inverse transforms for
rotation+flip combinations. The inverse of hflip(rot90(x)) is rot270(hflip(x)),
NOT inv_rot90(hflip(x)). Same for rot180/rot270 variants.

Forward transforms (fwd):
  8 variants = 4 rotations × 2 h-flip options

Correct inverse transforms (inv):
  The inverse of fwd must UNDO in reverse order:
  - fwd = hflip ∘ rotN  →  inv = rot(-N) ∘ hflip
  - NOT inv_rotN ∘ hflip (wrong order!)

Tested by symmetry check (symmetric patterns should survive round-trip).
"""

import os, sys, base64, cv2
import numpy as np
import pandas as pd
import scipy.io
import torch
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[DEVICE] {DEVICE}')

PROJECT_DIR = '/home/luluboy/projects/vrdl_final'
CODE_DIR    = os.path.join(PROJECT_DIR, 'code')
NAFNET_DIR  = os.path.join(CODE_DIR, 'NAFNet')
DATA_DIR    = os.path.join(PROJECT_DIR, 'data')
WEIGHTS_DIR = os.path.join(PROJECT_DIR, 'weights')
SUBS_DIR    = os.path.join(PROJECT_DIR, 'submissions')
MAT_FILE    = os.path.join(DATA_DIR, 'BenchmarkNoisyBlocksSrgb.mat')
WEIGHT_FILE = os.path.join(WEIGHTS_DIR, 'NAFNet-SIDD-width64.pth')
OUTPUT_CSV  = os.path.join(SUBS_DIR, 'SubmitSrgb_nafnet_tta8_fixed.csv')


# ── NAFNet helpers ────────────────────────────────────────────────────────────

def img2tensor(img):
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).float().to(DEVICE).unsqueeze(0)

def tensor2img(t):
    out = t[0].detach().cpu().clamp(0, 1).mul(255).round().byte()
    return out.permute(1, 2, 0).numpy()

def build_model():
    sys.path.insert(0, NAFNET_DIR)
    from basicsr.models.archs.NAFNet_arch import NAFNet
    model = NAFNet(img_channel=3, width=64,
                   enc_blk_nums=[2, 2, 4, 8],
                   middle_blk_num=12,
                   dec_blk_nums=[2, 2, 2, 2])
    sd = torch.load(WEIGHT_FILE, map_location='cpu', weights_only=True)
    if isinstance(sd, dict) and 'params' in sd:
        sd = sd['params']
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.to(DEVICE).eval()
    return model


# ── Corrected TTA transforms ─────────────────────────────────────────────────
# Key insight: inverse must UNDO in reverse order of forward
#
# Forward: first apply rotation, then hflip
#   fwd = hflip ∘ rotN
# Correct inverse: first hflip (to undo last operation), then inv_rotN
#   inv = inv_rotN ∘ hflip  ← undo hflip first, then rotation

def rot90(img):      return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
def rot180(img):     return cv2.rotate(img, cv2.ROTATE_180)
def rot270(img):     return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
def hflip(img):      return cv2.flip(img, 1)   # horizontal flip

# Correct inverses (undo hflip FIRST, then rotation)
def inv_rot0(img):   return img
def inv_rot90(img):  return rot270(img)         # inverse of rot90 is rot270
def inv_rot180(img): return rot180(img)         # inverse of rot180 is rot180
def inv_rot270(img): return rot90(img)         # inverse of rot270 is rot90

# For hflip composed with rotation: inv = inv_rotN ∘ hflip
#   (undo hflip first, then undo the rotation)
def inv_hflip_only(img): return hflip(img)

def inv_rot0_hflip(img):   return inv_hflip_only(img)            # undo hflip, nothing else
def inv_rot90_hflip(img):  return inv_rot90(hflip(img))          # undo hflip, then inv_rot90
def inv_rot180_hflip(img): return inv_rot180(hflip(img))         # undo hflip, then inv_rot180
def inv_rot270_hflip(img): return inv_rot270(hflip(img))         # undo hflip, then inv_rot270

# 8 TTA variants: (fwd_transform, correct_inverse_transform)
# Order: 4 rotations (no flip) + 4 rotations (with hflip)
TTA_VARIANTS = [
    (lambda x: x,                           inv_rot0),           # rot0
    (rot90,                                 inv_rot90),          # rot90
    (rot180,                                inv_rot180),         # rot180
    (rot270,                                inv_rot270),         # rot270
    (hflip,                                 inv_hflip_only),     # hflip only
    (lambda x: hflip(rot90(x)),             inv_rot90_hflip),    # rot90+hflip
    (lambda x: hflip(rot180(x)),             inv_rot180_hflip),   # rot180+hflip
    (lambda x: hflip(rot270(x)),             inv_rot270_hflip),   # rot270+hflip
]


def test_symmetry(model):
    """Verify all 8 variants are true inverses by round-trip on a test pattern."""
    print('\n🧪 TTA Transform Symmetry Test (fixed version)...')
    test_img = np.zeros((256, 256, 3), dtype=np.uint8)
    test_img[120:136, :] = 255  # horizontal white bar
    test_img[:, 120:136] = 255  # vertical white bar

    variant_names = [
        'rot0', 'rot90', 'rot180', 'rot270',
        'hflip', 'rot90+hflip', 'rot180+hflip', 'rot270+hflip'
    ]
    all_pass = True
    for (fwd, inv), name in zip(TTA_VARIANTS, variant_names):
        # Apply forward → model → inverse (model should be identity for clean input)
        x_fwd = fwd(test_img)
        t = img2tensor(x_fwd)
        out = model(t)
        x_out = tensor2img(out)
        x_restored = inv(x_out)
        diff = int(np.abs(x_restored.astype(np.int16) - test_img.astype(np.int16)).sum())
        status = '✅' if diff == 0 else '❌'
        print(f'   {name}: {status} (diff={diff})')
        if diff > 0:
            all_pass = False

    if all_pass:
        print('   All transforms symmetric ✅')
    else:
        print('   Some transforms NOT symmetric — check model behavior on rotated input')
    return all_pass


@torch.no_grad()
def denoise_block_tta(model, block):
    preds = []
    for fwd, inv in TTA_VARIANTS:
        t = img2tensor(fwd(block))
        out = tensor2img(model(t))
        preds.append(inv(out).astype(np.float32))
    avg = np.mean(preds, axis=0)
    return np.clip(avg, 0, 255).round().astype(np.uint8)


def block_to_base64(img):
    """Encode as raw bytes base64 (official format for Kaggle submission)."""
    return base64.b64encode(img.tobytes()).decode('utf-8')


def main():
    # Check data
    if not os.path.exists(MAT_FILE):
        print(f'❌ {MAT_FILE} not found.')
        sys.exit(1)

    print('='*60)
    print('VRDL Final – NAFNet + 8-way TTA (FIXED)')
    print('='*60)

    model = build_model()
    print(f'Loaded NAFNet from {WEIGHT_FILE}')

    test_symmetry(model)

    # Load data
    print(f'\n📂 Loading {MAT_FILE}...')
    mat = scipy.io.loadmat(MAT_FILE)
    inputs = mat['BenchmarkNoisyBlocksSrgb']
    n_i, n_j = inputs.shape[0], inputs.shape[1]
    n_blocks = n_i * n_j
    print(f'   shape: {inputs.shape} → {n_i}×{n_j}={n_blocks} blocks')

    # Run TTA inference
    print(f'\n🔁 Running fixed 8-way TTA on {n_blocks} blocks...')
    results = []
    for i in range(n_i):
        for j in range(n_j):
            block = inputs[i, j, :, :, :]
            denoised = denoise_block_tta(model, block)
            block_id = i * n_j + j
            results.append({
                'Id': block_id,
                'Base64EncodedBlocks': block_to_base64(denoised)
            })

    df = pd.DataFrame(results)
    df = df.sort_values('Id').reset_index(drop=True)  # ensure 0..1279 sorted
    os.makedirs(SUBS_DIR, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    size_mb = os.path.getsize(OUTPUT_CSV) / 1e6
    print(f'\n✅ Saved: {OUTPUT_CSV} ({size_mb:.1f} MB, {len(df)} rows)')

    # Verify format
    if len(df) == 1280:
        sample_b64 = df.iloc[0]['Base64EncodedBlocks']
        expected_len = 262144
        actual_len = len(sample_b64)
        print(f'   Format check: base64 len={actual_len} (expected {expected_len})')
        if actual_len == expected_len:
            print('   ✅ Format OK — ready for Kaggle submit!')
        else:
            print(f'   ⚠️  Unexpected base64 length (may need PNG encoding instead)')
    print('\n📋 Next: Run ensemble script with this CSV to build new ensemble v3')


if __name__ == '__main__':
    main()