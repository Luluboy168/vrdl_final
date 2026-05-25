#!/usr/bin/env python3
"""
infer_restormer_ft_epoch5.py
用 fine-tuned Restormer epoch 5 對 1280 個測試 blocks 推論，
產出 SubmitSrgb_restormer_ft_epoch5.csv
"""
import os, sys, torch, scipy.io, numpy as np, pandas as pd, base64, cv2

PROJECT_DIR    = '/home/luluboy/projects/vrdl_final'
CODE_DIR       = os.path.join(PROJECT_DIR, 'code')
RESTORMER_DIR  = os.path.join(CODE_DIR, 'Restormer')
DATA_DIR       = os.path.join(PROJECT_DIR, 'data')
CKPT_DIR       = os.path.join(PROJECT_DIR, 'checkpoints')
SUBS_DIR       = os.path.join(PROJECT_DIR, 'submissions')

MAT_FILE      = os.path.join(DATA_DIR, 'BenchmarkNoisyBlocksSrgb.mat')
WEIGHT_FILE   = os.path.join(CKPT_DIR, 'restormer_ft_epoch5.pth')
OUTPUT_CSV    = os.path.join(SUBS_DIR, 'SubmitSrgb_restormer_ft_epoch5.csv')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[DEVICE] Using: {DEVICE}')

def build_model():
    sys.path.insert(0, os.path.join(RESTORMER_DIR, 'basicsr/models/archs'))
    from restormer_arch import Restormer
    model = Restormer(
        inp_channels=3, out_channels=3, dim=48,
        num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type='BiasFree', dual_pixel_task=False,
    )
    print(f'Loading epoch 5 weights: {WEIGHT_FILE}')
    ckpt = torch.load(WEIGHT_FILE, map_location='cpu')
    state_dict = ckpt.get('params', ckpt.get('state_dict', ckpt))
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

def img2tensor(img):
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).float()

def tensor2img(tensor):
    out = tensor.detach().cpu().clamp(0, 1).mul(255).round().byte()
    return out.permute(1, 2, 0).numpy()

def raw_base64(img):
    return base64.b64encode(img.tobytes()).decode('utf-8')

def main():
    print(f'Reading {MAT_FILE} ...')
    mat = scipy.io.loadmat(MAT_FILE)
    keys = [k for k in mat.keys() if not k.startswith('__')]
    noisy_blocks = mat[keys[0]]
    print(f'Shape: {noisy_blocks.shape}')

    model = build_model()
    rows, cols = noisy_blocks.shape[0], noisy_blocks.shape[1]
    total = rows * cols

    # TTA: 8-way (original + 4 rotations + flip variants)
    def apply_tta(model, img_t):
        preds = []
        # Original
        with torch.no_grad():
            p = model(img_t)
            preds.append(p)
        # Rotate 90
        img_r90 = torch.rot90(img_t, 1, [2, 3])
        with torch.no_grad():
            p = torch.rot90(model(img_r90), -1, [2, 3])
            preds.append(p)
        # Rotate 180
        img_r180 = torch.rot90(img_t, 2, [2, 3])
        with torch.no_grad():
            p = torch.rot90(model(img_r180), -2, [2, 3])
            preds.append(p)
        # Rotate 270
        img_r270 = torch.rot90(img_t, 3, [2, 3])
        with torch.no_grad():
            p = torch.rot90(model(img_r270), -3, [2, 3])
            preds.append(p)
        # Flip H
        img_h = torch.flip(img_t, [3])
        with torch.no_grad():
            p = torch.flip(model(img_h), [3])
            preds.append(p)
        # Flip V
        img_v = torch.flip(img_t, [2])
        with torch.no_grad():
            p = torch.flip(model(img_v), [2])
            preds.append(p)
        # Flip both
        img_hv = torch.flip(img_t, [2, 3])
        with torch.no_grad():
            p = torch.flip(model(img_hv), [2, 3])
            preds.append(p)
        # Transpose
        img_t2 = img_t.transpose(2, 3)
        with torch.no_grad():
            p = model(img_t2).transpose(2, 3)
            preds.append(p)
        avg = torch.mean(torch.stack(preds), dim=0)
        return avg

    print(f'Running Restormer-FT Epoch5 + 8-way TTA inference ({total} blocks)...')
    records = []
    for i in range(rows):
        for j in range(cols):
            block_id = i * cols + j
            img = noisy_blocks[i, j]
            img_t = img2tensor(img).unsqueeze(0).to(DEVICE)
            out_t = apply_tta(model, img_t)
            out_img = tensor2img(out_t[0])
            out_img = np.clip(out_img, 0, 255).astype(np.uint8)
            b64 = raw_base64(out_img)
            records.append({'ID': block_id, 'BLOCK': b64})
            if (block_id + 1) % 200 == 0 or block_id == total - 1:
                print(f'  [{block_id+1}/{total}] done. Peak: {torch.cuda.max_memory_allocated()/1e9:.3f}GB')

    print(f'Writing {OUTPUT_CSV} ...')
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    sz = os.path.getsize(OUTPUT_CSV) / 1e6
    first_b64 = df['BLOCK'].iloc[0]
    assert len(first_b64) == 262144, f'Wrong block size: {len(first_b64)}'
    print(f'✅ Done: {OUTPUT_CSV} ({sz:.1f} MB, {len(records)} rows)')
    print(f'EXIT_CODE:0')

if __name__ == '__main__':
    main()