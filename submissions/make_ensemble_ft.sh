#!/bin/bash
# Ensemble NAFNet baseline + Restormer-FT with different alpha values
PYTHON=/home/luluboy/miniconda3/bin/python3
NAFNET=/home/luluboy/projects/vrdl_final/submissions/SubmitSrgb_baseline_official.csv
RESTORMER_FT=/home/luluboy/projects/vrdl_final/submissions/SubmitSrgb_restormer_ft.csv
OUTDIR=/home/luluboy/projects/vrdl_final/submissions

for alpha in 0.60 0.65 0.72; do
  echo "Generating alpha=$alpha..."
  $PYTHON - <<PY
import pandas as pd, base64, numpy as np

alpha = $alpha
nafnet = pd.read_csv('$NAFNET')
restormer_ft = pd.read_csv('$RESTORMER_FT')

# Ensure same ID order
nafnet = nafnet.sort_values('ID').reset_index(drop=True)
restormer_ft = restormer_ft.sort_values('ID').reset_index(drop=True)

def b64_to_img(b64):
    return np.frombuffer(base64.b64decode(b64), dtype=np.uint8).reshape(256, 256, 3).copy()

def img_to_b64(img):
    return base64.b64encode(img.tobytes()).decode('utf-8')

new_blocks = []
for i in range(len(nafnet)):
    img_n = b64_to_img(nafnet.loc[i, 'BLOCK'])
    img_r = b64_to_img(restormer_ft.loc[i, 'BLOCK'])
    # Blend in uint8 space (clip to 0-255)
    blended = np.clip(np.round(alpha * img_n.astype(np.float32) + (1 - alpha) * img_r.astype(np.float32)), 0, 255).astype(np.uint8)
    new_blocks.append(img_to_b64(blended))

result = pd.DataFrame({'ID': nafnet['ID'], 'BLOCK': new_blocks})
fname = f'$OUTDIR/SubmitSrgb_nafnet_ft_alpha_{str(alpha).replace(".", "")}.csv'
result.to_csv(fname, index=False)
print(f'Saved {fname}, size={len(result)}, first BLOCK len={len(new_blocks[0])}')
PY
done
echo "Done!"
