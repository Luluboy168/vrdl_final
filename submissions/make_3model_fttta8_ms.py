"""
3-model ensemble: NAFTta + RFT-TTA8 + RFT-MS-TTA
Weights: NAFTta=0.74, RFT-TTA8=0.15, RFT-MS-TTA=0.11
Motivation: fttta (NAFTta+RFT-TTA8) = 40.4852, msfttta (NAFTta+RFT-MS-TTA) = 40.4757
           → both Restormer variants have complementary info; blend them with NAFTta.
"""
import os, time, base64
import pandas as pd
import numpy as np

SUB   = "/home/luluboy/projects/vrdl_final/submissions"
PY36  = "/home/luluboy/miniconda3/bin/python3"

NAFTA  = os.path.join(SUB, "SubmitSrgb_nafnet_tta8_fixed.csv")     # 40.3675
RFT8   = os.path.join(SUB, "SubmitSrgb_restormer_ft_tta8.csv")      # 40.1344
RFMS   = os.path.join(SUB, "SubmitSrgb_restormer_ft_ms_tta.csv")    # 40.0371
OUT    = os.path.join(SUB, "SubmitSrgb_3model_fttta8_ms_0.74_0.15_0.11.csv")

if os.path.exists(OUT):
    print(f"[SKIP] Already exists: {OUT}")
else:
    print("Loading CSVs...")
    t0 = time.time()
    df_n = pd.read_csv(NAFTA).sort_values('ID').reset_index(drop=True)
    df_r = pd.read_csv(RFT8).sort_values('ID').reset_index(drop=True)
    df_m = pd.read_csv(RFMS).sort_values('ID').reset_index(drop=True)

    for name, df in [("NAFTta", df_n), ("RFT-TTA8", df_r), ("RFT-MS-TTA", df_m)]:
        if df['ID'].min() == 1:
            df['ID'] = df['ID'] - 1
        print(f"  {name}: {len(df)} rows, ID range [{df['ID'].min()}, {df['ID'].max()}]")

    alpha, beta, gamma = 0.74, 0.15, 0.11
    assert abs(alpha+beta+gamma - 1.0) < 1e-9

    print(f"\nBlending: NAFTta={alpha} + RFT-TTA8={beta} + RFT-MS-TTA={gamma}")
    blocks = []
    for i in range(len(df_n)):
        n = np.frombuffer(base64.b64decode(df_n.loc[i, 'BLOCK']), np.uint8).reshape(256,256,3).astype(np.float32)
        r = np.frombuffer(base64.b64decode(df_r.loc[i, 'BLOCK']), np.uint8).reshape(256,256,3).astype(np.float32)
        m = np.frombuffer(base64.b64decode(df_m.loc[i, 'BLOCK']), np.uint8).reshape(256,256,3).astype(np.float32)
        fused = np.clip(alpha*n + beta*r + gamma*m, 0, 255).round().astype(np.uint8)
        blocks.append(base64.b64encode(fused.tobytes()).decode('utf-8'))

    result = pd.DataFrame({'ID': df_n['ID'].values, 'BLOCK': blocks})
    result.to_csv(OUT, index=False)
    print(f"\nSaved: {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB, {time.time()-t0:.1f}s)")
    print(f"First BLOCK len: {len(blocks[0])}")

print("\nDone!")