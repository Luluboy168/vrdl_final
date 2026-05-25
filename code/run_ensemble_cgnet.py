import os, sys, base64, numpy as np, pandas as pd

PROJECT = "/home/luluboy/projects/vrdl_final"
NAF_CSV = PROJECT + "/submissions/SubmitSrgb_nafnet_tta8_fixed.csv"
CGNET_CSV = PROJECT + "/submissions/SubmitSrgb_cgnet_tta_fixed.csv"
OUT_DIR = PROJECT + "/submissions"
ALPHAS = [0.70, 0.75]

print("Loading CSVs...")
df_naf = pd.read_csv(NAF_CSV)
df_cg = pd.read_csv(CGNET_CSV)
print(f"Loaded NAF={len(df_naf)}, CGNet={len(df_cg)}")

for alpha in ALPHAS:
    out_name = f"SubmitSrgb_2model_naftta_cgnet_alpha{int(alpha*100)}.csv"
    out_path = f"{OUT_DIR}/{out_name}"
    if os.path.exists(out_path):
        print(f"Skip {out_name} (exists)")
        continue
    print(f"Generating alpha={alpha}...")
    rows = []
    for i in range(1280):
        img_n = np.frombuffer(base64.b64decode(df_naf.iloc[i]["BLOCK"]), np.uint8).reshape(256,256,3).astype(np.float64)
        img_c = np.frombuffer(base64.b64decode(df_cg.iloc[i]["BLOCK"]), np.uint8).reshape(256,256,3).astype(np.float64)
        img_ens = (alpha*img_n + (1-alpha)*img_c).clip(0,255).astype(np.uint8)
        rows.append({"ID": i, "BLOCK": base64.b64encode(img_ens.tobytes()).decode("utf-8")})
        if (i+1) % 256 == 0:
            print(f"  {i+1}/1280")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved {out_name} ({os.path.getsize(out_path)/1e6:.1f} MB)")

print("Done!")