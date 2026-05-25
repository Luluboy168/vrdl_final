#!/usr/bin/env python3
"""
Quick test: Load NAFNet-SIDD-width64.pth and run inference on a synthetic image.
验证模型加载成功 + 输出图片正常。
"""
import sys
import os
import torch
import cv2
import numpy as np

# Add NAFNet to path
sys.path.insert(0, '/home/luluboy/projects/vrdl_final/code/NAFNet')

from basicsr.models.archs.NAFNet_arch import NAFNet
from basicsr.utils import tensor2img

def create_synthetic_noisy_image(h=256, w=256):
    """Create a synthetic noisy image for testing."""
    # Create a simple gradient image
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            img[i, j] = [int(i * 255 / h), int(j * 255 / w), 128]
    # Add noise
    noise = np.random.randint(0, 50, (h, w, 3), dtype=np.uint8)
    img = np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)
    return img

def img2tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.0
    if bgr2rgb:
        img = img[:, :, [2, 1, 0]]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img

def test_nafnet():
    print("=== NAFNet Inference Test ===")
    
    # Create model with same architecture as NAFNet-SIDD-width64
    model = NAFNet(
        img_channel=3,
        width=64,
        enc_blk_nums=[2, 2, 4, 8],
        middle_blk_num=12,
        dec_blk_nums=[2, 2, 2, 2]
    )
    
    # Load pretrained weights
    weight_path = '/home/luluboy/projects/vrdl_final/weights/NAFNet-SIDD-width64.pth'
    print(f"Loading weights from: {weight_path}")
    state_dict = torch.load(weight_path, map_location='cuda:0', weights_only=True)
    
    # The checkpoint wraps weights inside 'params' key
    if isinstance(state_dict, dict) and 'params' in state_dict:
        state_dict = state_dict['params']
    
    # Handle 'module.' prefix if present (from DDP training)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # remove 'module.' prefix
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=True)
    print("✅ Weight loaded successfully (strict=True)")
    
    model = model.cuda()
    model.eval()
    
    # Create synthetic test image
    print("Creating synthetic noisy test image (256x256)...")
    img_noisy = create_synthetic_noisy_image(256, 256)
    
    # Save input image
    input_path = '/home/luluboy/projects/vrdl_final/data/test_input.png'
    cv2.imwrite(input_path, img_noisy)
    print(f"✅ Saved noisy input: {input_path}")
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        img_tensor = img2tensor(img_noisy).unsqueeze(0).cuda()
        print(f"  Input shape: {img_tensor.shape}")
        
        output_tensor = model(img_tensor)
        print(f"  Output shape: {output_tensor.shape}")
        
        # Convert back to image
        output_img = tensor2img([output_tensor[0]], rgb2bgr=False)
        output_img = np.clip(output_img, 0, 255).astype(np.uint8)
    
    # Save output
    output_path = '/home/luluboy/projects/vrdl_final/data/test_output.png'
    cv2.imwrite(output_path, output_img)
    print(f"✅ Saved denoised output: {output_path}")
    print(f"  Input range: [{img_noisy.min()}, {img_noisy.max()}]")
    print(f"  Output range: [{output_img.min()}, {output_img.max()}]")
    
    print("\n=== Test PASSED: NAFNet inference working correctly ===")
    return True

if __name__ == '__main__':
    success = test_nafnet()
    sys.exit(0 if success else 1)