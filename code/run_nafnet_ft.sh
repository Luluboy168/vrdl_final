#!/bin/bash
# NAFNet Fine-tuning launcher - fix path issues from previous attempt
LOG="/home/luluboy/projects/vrdl_final/code/NAFNet/experiments/NAFNet-SIDD-finetune-width64-8k/train_ft.log"
EXP_DIR="/home/luluboy/projects/vrdl_final/code/NAFNet/experiments/NAFNet-SIDD-finetune-width64-8k"
YAML_ORIG="/home/luluboy/projects/vrdl_final/code/NAFNet/options/train/SIDD/NAFNet-width64-finetune-8k.yml"

echo "[$(date)] Starting NAFNet fine-tuning..." >> "$LOG"

cd /home/luluboy/projects/vrdl_final/code/NAFNet || { echo "NAFNet dir not found!"; exit 1; }

export PYTHONPATH="$(pwd):$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# Launch training - it will auto-resume from latest state if training_states/ exists
/home/luluboy/miniconda3/bin/python basicsr/train.py \
    -opt "$YAML_ORIG" \
    2>&1 | tee -a "$LOG"