#!/usr/bin/env python3
"""Generate ensemble CSVs with weighted average of NAFNet+TTA and Restormer+TTA."""

import base64
import csv
import sys
import numpy as np

NAFNET_TTA = '/home/luluboy/projects/vrdl_final/submissions/SubmitSrgb_nafnet_tta8_fixed.csv'
RESTORMER_TTA = '/home/luluboy/projects/vrdl_final/submissions/SubmitSrgb_restormer_tta8.csv'
OUT_DIR = '/home/luluboy/projects/vrdl_final/submissions/'

def load_csv(path):
    """Load CSV: returns dict of {block_id: np.array(uint8, 196608)}. Decodes base64, reshapes to (512, 512)."""
    import csv
    csv.field_size_limit(2**20 * 512)  # increase field size limit
    blocks = {}
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            block_id = row[0]
            b64_str = row[1]
            decoded = base64.b64decode(b64_str)
            arr = np.frombuffer(decoded, dtype=np.uint8)
            blocks[block_id] = arr
    return blocks

def save_csv(blocks, out_path):
    """Save blocks dict to CSV in official format."""
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'BLOCK'])
        for block_id in sorted(blocks.keys(), key=lambda x: int(x)):
            arr = blocks[block_id]
            b64_str = base64.b64encode(arr.tobytes()).decode('ascii')
            writer.writerow([block_id, b64_str])

def main():
    alphas = [0.65, 0.75, 0.68]
    
    print("Loading NAFNet+TTA...")
    nafnet_blocks = load_csv(NAFNET_TTA)
    print(f"  Loaded {len(nafnet_blocks)} blocks")
    
    print("Loading Restormer+TTA...")
    restormer_blocks = load_csv(RESTORMER_TTA)
    print(f"  Loaded {len(restormer_blocks)} blocks")
    
    for alpha in alphas:
        print(f"\nGenerating ensemble alpha={alpha}...")
        beta = 1.0 - alpha
        
        # Create ensemble blocks
        ensemble_blocks = {}
        common_ids = set(nafnet_blocks.keys()) & set(restormer_blocks.keys())
        print(f"  Common block IDs: {len(common_ids)}")
        
        for bid in common_ids:
            n_arr = nafnet_blocks[bid].astype(np.float32)
            r_arr = restormer_blocks[bid].astype(np.float32)
            blended = alpha * n_arr + beta * r_arr
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            ensemble_blocks[bid] = blended
        
        # Validate
        sample_bid = sorted(common_ids)[0]
        print(f"  Sample block {sample_bid}: shape={ensemble_blocks[sample_bid].shape}, dtype={ensemble_blocks[sample_bid].dtype}")
        print(f"  Sample range: [{ensemble_blocks[sample_bid].min()}, {ensemble_blocks[sample_bid].max()}]")
        
        # Save
        out_path = f"{OUT_DIR}SubmitSrgb_ensemble_alpha_{int(alpha*100)}.csv"
        save_csv(ensemble_blocks, out_path)
        
        # Validate saved file
        with open(out_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            first_row = next(reader)
            b64_len = len(first_row[1])
            print(f"  Output: {out_path}")
            print(f"  First b64 length: {b64_len} (expected 262144)")
            line_count = 1 + sum(1 for _ in reader)
            print(f"  Total lines: {line_count} (expected 1281)")
        
        print(f"  Done: alpha={alpha}")

    print("\nAll done!")

if __name__ == '__main__':
    main()