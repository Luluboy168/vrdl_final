#!/usr/bin/env python3
"""Launch NAFNet fine-tuning with modified settings for first checkpoint."""
import yaml, subprocess, sys, os

YAML_ORIG = '/home/luluboy/projects/vrdl_final/code/NAFNet/options/train/SIDD/NAFNet-width64-finetune-8k.yml'
EXP_DIR = '/home/luluboy/projects/vrdl_final/code/NAFNet/experiments/NAFNet-SIDD-finetune-width64-8k'
os.makedirs(EXP_DIR, exist_ok=True)

with open(YAML_ORIG) as f:
    config = yaml.safe_load(f)

# Modify settings for faster first checkpoint
config['train']['total_iter'] = 10000
config['logger']['save_checkpoint_freq'] = 2000
config['val']['val_freq'] = 0  # skip validation during training

# Write temp YAML
YAML_TMP = f'{EXP_DIR}/train_ft_cron.yml'
with open(YAML_TMP, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"Launching NAFNet fine-tuning with {config['train']['total_iter']} iters, checkpoint every {config['logger']['save_checkpoint_freq']}")
print(f"Config written to {YAML_TMP}")

cmd = [
    '/home/luluboy/miniconda3/bin/python', 'basicsr/train.py',
    '-opt', YAML_TMP
]

env = os.environ.copy()
env['PYTHONPATH'] = '/home/luluboy/projects/vrdl_final/code/NAFNet:' + env.get('PYTHONPATH', '')
env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

log_file = f'{EXP_DIR}/train_ft.log'
print(f"Logging to {log_file}")

with open(log_file, 'a') as lf:
    lf.write(f"\n\n=== NAFNet Fine-tune Restart: {subprocess.list2cmdline(cmd)} ===\n\n")

proc = subprocess.Popen(cmd, cwd='/home/luluboy/projects/vrdl_final/code/NAFNet', env=env, stdout=open(log_file, 'a'), stderr=subprocess.STDOUT)
print(f"Training started with PID {proc.pid}. Will save first checkpoint at iter 2000 (~20min from now)")
print(f"Monitor with: tail -f {log_file}")