import os
import shutil
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

root_dir = f'./datasets/iOCT/test'
output_base = f'./datasets/iOCT/yolo'

# ============ Create test dataset ============
target_folders = ['images', 'masks']
for target_sub in target_folders:
    os.makedirs(os.path.join(output_base, target_sub, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_base, target_sub, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_base, target_sub, 'test'), exist_ok=True)

start_idx = 0
seq_folders = [f for f in os.listdir(os.path.join(root_dir, 'JPEGImages'))
                if os.path.isdir(os.path.join(root_dir, 'JPEGImages', f))]
for seq_folder in seq_folders:
    source_image_folder = os.path.join(root_dir, 'JPEGImages', seq_folder)
    source_mask_folder = os.path.join(root_dir, 'Annotations', seq_folder)
    if os.path.isdir(source_mask_folder):
        fnames = os.listdir(source_mask_folder)
        for fname in fnames:
            dst_name = f"{start_idx:04d}.png"

            src_image = os.path.join(source_image_folder, fname)
            dst_image = os.path.join(output_base, 'images', 'test', dst_name)
            shutil.copy(src_image, dst_image)

            src_mask = os.path.join(source_mask_folder, fname)
            dst_mask = os.path.join(output_base, 'masks', 'test', dst_name)
            img = Image.open(src_mask)
            img = np.array(img).astype(np.uint8)
            out = Image.fromarray(img, mode="L")
            out.save(dst_mask)
            # shutil.copy(src_mask, dst_mask)

            start_idx += 1
