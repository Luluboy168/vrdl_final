#!/usr/bin/env python3
"""
Run TTA inference with the NEW fine-tuned NAFTta model (net_g_latest.pth).
Creates SubmitSrgb_naftta_new_tta8.csv, then blend with RFT-TTA8 at alpha=0.745.
"""
import os, sys, base64, cv2, time
import numpy as np
import pandas as pd
import scipy.io
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[DEVICE] {DEVICE}')

PROJECT_DIR = '/home/luluboy/projects/vrdl_final'
CODE_DIR    = os.path.join(PROJECT_DIR, 'code')
NAFNET_DIR  = os.path.join(CODE_DIR, 'NAFNet')
DATA_DIR    = os.path.join(PROJECT_DIR, 'data')
SUBS_DIR    = os.path.join(PROJECT_DIR, 'submissions')

# NEW fine-tuned NAFTta model (latest fine-tune completed 2026-05-25 11:42)
WEIGHT_FILE = '/home/luluboy/projects/vrdl_final/checkpoints/nafnet_ft_final.pth'
MAT_FILE    = os.path.join(DATA_DIR, 'BenchmarkNoisyBlocksSrgb.mat')
OUTPUT_CSV  = os.path.join(SUBS_DIR, 'SubmitSrgb_naftta_new_tta8.csv')

print(f"Weight file: {WEIGHT_FILE}")
assert os.path.exists(WEIGHT_FILE), f"Weight file not found: {WEIGHT_FILE}"
print(f"Mat file: {MAT_FILE}")
assert os.path.exists(MAT_FILE), f"Mat file not found: {MAT_FILE}"


# ── Model ────────────────────────────────────────────────────────────────────

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
    print(f"Loaded state_dict type: {type(sd)}, keys: {list(sd.keys()) if isinstance(sd, dict) else 'N/A'}")
    if isinstance(sd, dict) and 'params' in sd:
        sd = sd['params']
    # Handle 'module.' prefix
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.to(DEVICE).eval()
    return model


# ── TTA transforms ─────────────────────────────────────────────────────────────

def rot90(img):      return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
def rot180(img):     return cv2.rotate(img, cv2.ROTATE_180)
def rot270(img):     return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
def hflip(img):      return cv2.flip(img, 1)

def inv_rot0(img):   return img
def inv_rot90(img):  return rot270(img)
def inv_rot180(img): return rot180(img)
def inv_rot270(img): return rot90(img)
def inv_hflip(img):  return hflip(img)

def inv_rot0_hflip(img):   return hflip(img)
def inv_rot90_hflip(img):  return rot270(hflip(img))
def inv_rot180_hflip(img): return rot180(hflip(img))
def inv_rot270_hflip(img): return rot90(hflip(img))

# 8 TTA variants: (fwd_transform, correct_inverse_transform)
TTA_VARIANTS = [
    (lambda x: x,                        inv_rot0),        # rot0
    (rot90,                              inv_rot90),       # rot90
    (rot180,                             inv_rot180),      # rot180
    (rot270,                             inv_rot270),      # rot270
    (hflip,                              inv_hflip),      # hflip
    (lambda x: hflip(rot90(x)),          inv_rot90_hflip),  # rot90+hflip
    (lambda x: hflip(rot180(x)),         inv_rot180_hflip), # rot180+hflip
    (lambda x: hflip(rot270(x)),         inv_rot270_hflip), # rot270+hflip
]

def apply_tta(model, img):
    """Apply 8-way TTA and average the results (geometric mean → arithmetic for uint8)."""
    preds = []
    for fwd, inv in TTA_VARIANTS:
        t = fwd(img)
        t_t = img2tensor(t)
        with torch.no_grad():
            out = model(t_t)
        out_img = tensor2img(out)
        out_img = inv(out_img)
        preds.append(out_img.astype(np.float32))
    # Arithmetic mean of float32, then clip to uint8
    avg = np.mean(preds, axis=0).clip(0, 255).astype(np.uint8)
    return avg


# ── Main ───────────────────────────────────────────────────────────────────────

print("Building model...")
model = build_model()
print("Model ready.")

print("Loading noisy blocks...")
data = scipy.io.loadmat(MAT_FILE)
noisy = data['BenchmarkNoisyBlocksSrgb']  # (40, 32, 256, 256, 3)
print(f"Noisy blocks shape: {noisy.shape}")  # expect (40, 32, 256, 256, 3)

# Run TTA inference for all 1280 blocks
print("Running TTA inference...")
blocks = []
start = time.time()

for i in range(40):
    for j in range(32):
        k = i * 32 + j
        noisy_block = noisy[i, j]  # (256, 256, 3), uint8
        denoised = apply_tta(model, noisy_block)
        blocks.append(denoised)
        if (k + 1) % 128 == 0:
            elapsed = time.time() - start
            eta = elapsed / (k + 1) * (1280 - k - 1)
            print(f"  [{k+1}/1280]  ETA: {eta:.0f}s")

print(f"Inference done in {time.time()-start:.0f}s")

# Save TTA outputs
print("Saving NAFTta TTA CSV...")
tta_blocks = []
for block in blocks:
    tta_blocks.append(base64.b64encode(block.tobytes()).decode('utf-8'))

df = pd.DataFrame({'ID': np.arange(1280), 'BLOCK': tta_blocks})
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved: {OUTPUT_CSV}")
assert len(df['BLOCK'].iloc[0]) == 262144, "Bad base64 length!"

# Validate: check a few blocks
print(f"CSV rows: {len(df)}, columns: {list(df.columns)}")

# Now blend with RFT-TTA8 at alpha=0.745
RFT8_FILE = os.path.join(SUBS_DIR, 'SubmitSrgb_restormer_ft_tta8.csv')
print(f"\nBlending with RFT-TTA8 at alpha=0.745...")
assert os.path.exists(RFT8_FILE), f"RFT8 file not found: {RFT8_FILE}"

df_rft = pd.read_csv(RFT8_FILE)
alpha = 0.745

blended_blocks = []
for i in range(1280):
    b_new = np.frombuffer(base64.b64decode(df['BLOCK'].iloc[i]), dtype=np.uint8).reshape(256,256,3)
    b_rft = np.frombuffer(base64.b64decode(df_rft['BLOCK'].iloc[i]), dtype=np.uint8).reshape(256,256,3)
    blended = (alpha * b_new.astype(np.float32) + (1-alpha) * b_rft.astype(np.float32)).clip(0,255).astype(np.uint8)
    blended_blocks.append(base64.b64encode(blended.tobytes()).decode('utf-8'))

df_blend = pd.DataFrame({'ID': np.arange(1280), 'BLOCK': blended_blocks})
BLEND_FILE = os.path.join(SUBS_DIR, 'SubmitSrgb_2model_naftta_new_fttta_0745.csv')
df_blend.to_csv(BLEND_FILE, index=False)
print(f"Saved blend: {BLEND_FILE}")
assert len(df_blend['BLOCK'].iloc[0]) == 262144, "Bad blend base64 length!"

print("\n✅ Done! Blend CSV ready for submission.")
print(f"Blend file: {BLEND_FILE}")