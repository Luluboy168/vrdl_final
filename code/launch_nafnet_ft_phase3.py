#!/usr/bin/env python3
"""Phase 3 NAFNet Fine-tuning: 5 more epochs from latest NAFNet ft checkpoint."""
import yaml, subprocess, sys, os, shutil

# Use latest NAFNet finetuned checkpoint as pretrain
EXP_DIR     = '/home/luluboy/projects/vrdl_final/code/NAFNet/experiments/NAFNet-SIDD-finetune-width64-8k'
PRETRAINED  = '/home/luluboy/projects/vrdl_final/weights/NAFNet-SIDD-width64.pth'
YAML_ORIG   = '/home/luluboy/projects/vrdl_final/code/NAFNet/options/train/SIDD/NAFNet-width64-finetune-8k.yml'

os.makedirs(EXP_DIR, exist_ok=True)

with open(YAML_ORIG) as f:
    config = yaml.safe_load(f)

# Phase 3: 5 more epochs → ~10000 iterations (160*10 patches per epoch)
config['train']['total_iter']        = 10000
config['train']['warmup_iter']       = 0
config['train']['optim_g']['lr']     = 2.0e-5   # lower LR for continuation
config['logger']['save_checkpoint_freq'] = 2000
config['val']['val_freq']           = 0        # skip val to avoid OOM
config['path']['pretrain_network_g'] = PRETRAINED
config['path']['resume_state']       = None     # fresh start from pretrained

YAML_PH3 = f'{EXP_DIR}/train_phase3.yml'
with open(YAML_PH3, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"[Phase 3 NAFNet] Config: {YAML_PH3}")
print(f"[Phase 3 NAFNet] Pretrain: {PRETRAINED}")
print(f"[Phase 3 NAFNet] Total iter: {config['train']['total_iter']}, LR: {config['train']['optim_g']['lr']}")

env = os.environ.copy()
env['PYTHONPATH'] = '/home/luluboy/projects/vrdl_final/code/NAFNet:' + env.get('PYTHONPATH', '')
env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

cmd = [
    '/home/luluboy/miniconda3/bin/python', 'basicsr/train.py',
    '-opt', YAML_PH3
]

log_file = f'{EXP_DIR}/train_phase3.log'
with open(log_file, 'a') as lf:
    lf.write(f"\n\n=== NAFNet Phase3 start: {subprocess.list2cmdline(cmd)} ===\n\n")

proc = subprocess.Popen(
    cmd,
    cwd='/home/luluboy/projects/vrdl_final/code/NAFNet',
    env=env,
    stdout=open(log_file, 'a'),
    stderr=subprocess.STDOUT
)
print(f"[Phase 3 NAFNet] PID {proc.pid}, log: {log_file}")