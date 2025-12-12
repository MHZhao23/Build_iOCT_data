import os
import shutil
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2

fs_img_root = "./datasets/Synthetic_iOCT/train/JPEGImages"
fs_mask_root = "./datasets/Synthetic_iOCT/train/Annotations"

yolo_img_train_root = "./datasets/Synthetic_iOCT/yolo/images/train"
yolo_mask_train_root = "./datasets/Synthetic_iOCT/yolo/masks/train"
yolo_img_val_root = "./datasets/Synthetic_iOCT/yolo/images/val"
yolo_mask_val_root = "./datasets/Synthetic_iOCT/yolo/masks/val"
os.makedirs(yolo_img_train_root, exist_ok=True)
os.makedirs(yolo_mask_train_root, exist_ok=True)
os.makedirs(yolo_img_val_root, exist_ok=True)
os.makedirs(yolo_mask_val_root, exist_ok=True)

image_idx = 0
seqs = sorted(os.listdir(fs_img_root))
val_seqs = ['seq_21', 'seq_22', 'seq_23', 'seq_24']
for seq in tqdm(seqs):
    fs_names = os.listdir(os.path.join(fs_img_root, seq))
    for fs_name in fs_names:
        src = os.path.join(fs_img_root, seq, fs_name)
        dst = os.path.join(yolo_img_val_root, f"{image_idx:04d}.png") if seq in val_seqs \
            else os.path.join(yolo_img_train_root, f"{image_idx:04d}.png")
        shutil.copy(src, dst)

        src = os.path.join(fs_mask_root, seq, fs_name)
        dst = os.path.join(yolo_mask_val_root, f"{image_idx:04d}.png") if seq in val_seqs \
            else os.path.join(yolo_mask_train_root, f"{image_idx:04d}.png")
        img = Image.open(src)
        img = np.array(img).astype(np.uint8)
        out = Image.fromarray(img, mode="L")
        out.save(dst)
        # print(cv2.imread(dst, cv2.IMREAD_GRAYSCALE).shape, np.unique(cv2.imread(dst, cv2.IMREAD_GRAYSCALE)))
        
        image_idx += 1
