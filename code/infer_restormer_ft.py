#!/usr/bin/env python3
"""
infer_restormer_ft.py
用 fine-tuned Restormer (restormer_ft_final.pth) 對 1280 個測試 blocks 推論，
產出 SubmitSrgb_restormer_ft.csv（raw bytes base64，262144 chars/block）
"""
import os, sys, torch, scipy.io, numpy as np, pandas as pd, base64, cv2

PROJECT_DIR    = '/home/luluboy/projects/vrdl_final'
CODE_DIR       = os.path.join(PROJECT_DIR, 'code')
RESTORMER_DIR  = os.path.join(CODE_DIR, 'Restormer')
DATA_DIR       = os.path.join(PROJECT_DIR, 'data')
CKPT_DIR       = os.path.join(PROJECT_DIR, 'checkpoints')
SUBS_DIR       = os.path.join(PROJECT_DIR, 'submissions')

MAT_FILE      = os.path.join(DATA_DIR, 'BenchmarkNoisyBlocksSrgb.mat')
WEIGHT_FILE    = os.path.join(CKPT_DIR, 'restormer_ft_final.pth')
OUTPUT_CSV     = os.path.join(SUBS_DIR, 'SubmitSrgb_restormer_ft.csv')

os.makedirs(SUBS_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

# ===== Build model =====
def build_model():
    sys.path.insert(0, os.path.join(RESTORMER_DIR, 'basicsr/models/archs'))
    from restormer_arch import Restormer
    model = Restormer(
        inp_channels=3, out_channels=3, dim=48,
        num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type='BiasFree', dual_pixel_task=False,
    )
    print(f'📦 載入 fine-tuned 權重: {WEIGHT_FILE}')
    ckpt = torch.load(WEIGHT_FILE, map_location='cpu')
    state_dict = ckpt.get('params', ckpt.get('state_dict', ckpt))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f'Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}')
    model.to(DEVICE)
    model.eval()
    return model

# HWC uint8 [0,255] → CHW float32 [0,1] tensor
def img2tensor(img):
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).float()

# CHW float32 [0,1] tensor → HWC uint8 [0,255]
def tensor2img(tensor):
    out = tensor.detach().cpu().clamp(0, 1).mul(255).round().byte()
    return out.permute(1, 2, 0).numpy()

# Raw bytes base64 (official Kaggle format, 262144 chars per block)
def raw_base64(img):
    return base64.b64encode(img.tobytes()).decode('utf-8')

def main():
    print(f'📂 讀取 {MAT_FILE} ...')
    mat = scipy.io.loadmat(MAT_FILE)
    keys = [k for k in mat.keys() if not k.startswith('__')]
    noisy_blocks = mat[keys[0]]
    print(f'   shape: {noisy_blocks.shape}')

    model = build_model()

    rows, cols = noisy_blocks.shape[0], noisy_blocks.shape[1]
    total = rows * cols
    print(f'🔄 開始 fine-tuned Restormer 推論 ({total} blocks)...')

    records = []
    for i in range(rows):
        for j in range(cols):
            block_id = i * cols + j      # 0-based (0..1279)
            img = noisy_blocks[i, j]      # (256,256,3), uint8
            img_t = img2tensor(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out_t = model(img_t)
            out_img = tensor2img(out_t[0])
            out_img = np.clip(out_img, 0, 255).astype(np.uint8)
            b64 = raw_base64(out_img)
            records.append({'ID': block_id, 'BLOCK': b64})

            if (block_id + 1) % 200 == 0 or block_id == total - 1:
                print(f'   [{block_id+1}/{total}] done')

    print(f'💾 寫入 {OUTPUT_CSV} ...')
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    sz = os.path.getsize(OUTPUT_CSV) / 1e6
    print(f'✅ CSV ({sz:.1f} MB), {len(records)} rows')

    # Quick format check
    first_b64 = df['BLOCK'].iloc[0]
    print(f'   First BLOCK length: {len(first_b64)} (expected 262144)')
    assert len(first_b64) == 262144, f'Wrong block size: {len(first_b64)}'
    print('✅ Format check PASSED')

if __name__ == '__main__':
    main()