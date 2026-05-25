#!/bin/bash
# NAFNet Fine-tuning launcher
cd /home/luluboy/projects/vrdl_final/code/NAFNet

export PYTHONPATH=/home/luluboy/projects/vrdl_final/code/NAFNet:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOGFILE="experiments/NAFNet-SIDD-finetune-width64-8k/train_ft.log"
mkdir -p "experiments/NAFNet-SIDD-finetune-width64-8k"

echo "[$(date)] Starting NAFNet fine-tuning..." >> "$LOGFILE"

/home/luluboy/miniconda3/bin/python basicsr/train.py \
  -opt options/train/SIDD/NAFNet-width64-finetune-8k.yml \
  2>&1 | tee -a "$LOGFILE"

echo "[$(date)] NAFNet fine-tuning finished!" >> "$LOGFILE"