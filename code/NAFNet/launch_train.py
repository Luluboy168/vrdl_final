#!/usr/bin/env python
"""Launch NAFNet training with proper module resolution."""
import sys
# Remove any installed basicsr from site-packages to avoid shadowing local version
sys.path = [p for p in sys.path if 'site-packages/basicsr' not in p and 'site-packages\\basicsr' not in p]
sys.path.insert(0, '/home/luluboy/projects/vrdl_final/code/NAFNet')

import os
os.chdir('/home/luluboy/projects/vrdl_final/code/NAFNet')

# Now run the actual basicsr train
from basicsr.train import train
train()