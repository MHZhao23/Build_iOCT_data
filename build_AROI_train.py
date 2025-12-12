import os
import random
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from tqdm import tqdm
import cv2


# ---------------- get Synthetic_iOCT ----------------
def save_ann_png(path, mask):
    """Save a mask as a PNG file with the given palette."""
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    label_colors = {
        0: (0, 0, 0),        # background
        1: (220, 50, 47),    # retina (red)
        2: (60, 201, 230),   # tool (cyan)
        3: (239, 247, 5),    # artifact (yellow)
    }

    palette = []
    for i in range(4):
        if i in label_colors:
            palette.extend(label_colors[i])
        else:
            palette.extend((0, 0, 0))  # pad unused indices with black

    output_mask = Image.fromarray(mask)
    output_mask.putpalette(palette)
    output_mask.save(path)

# Paths
oct_root = "./datasets/AROI - online/24 patient"
output_img_root = "./datasets/AROI_OCT/train/JPEGImages"
output_mask_root = "./datasets/AROI_OCT/train/Annotations"
    
all_image_paths = []
all_mask_paths = []

# Process all patients
start_idx = 0
img_size = 512
patients = sorted(os.listdir(oct_root))
np.random.seed(42)
for patient in tqdm(patients):
    labelled_oct_dir = os.path.join(oct_root, patient, "raw/labeled")
    oct_mask_dir = os.path.join(oct_root, patient, "mask/number")

    p_i = patient[7:]
    output_img_dir = os.path.join(output_img_root, f"seq_{p_i}")
    output_mask_dir = os.path.join(output_mask_root, f"seq_{p_i}")
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    labelled_oct_images = sorted([os.path.join(labelled_oct_dir, f) for f in os.listdir(labelled_oct_dir) if f.endswith(".png")])
    oct_mask_names = sorted([f for f in os.listdir(oct_mask_dir) if f.endswith(".png")])
    assert len(labelled_oct_images) == len(oct_mask_names)
    num_oct_frames = len(oct_mask_names) 

    for i in range(num_oct_frames):
        oct_name = os.path.basename(labelled_oct_images[i])
        oct_image = Image.open(labelled_oct_images[i]).convert("L")
        oct_mask = Image.open(os.path.join(oct_mask_dir, oct_name)).convert("L") if oct_name in oct_mask_names else None

        # ---------------- Augmentation ----------------
        brightness_oct = round(np.random.uniform(0.5, 1.2), 2)
        oct_image = ImageEnhance.Brightness(oct_image).enhance(brightness_oct)
        oct_image_array = np.array(oct_image.resize((img_size, img_size), Image.NEAREST))
        oct_mask_array = np.array(oct_mask.resize((img_size, img_size), Image.NEAREST)) if oct_mask else None

        oct_mask_array[oct_mask_array == 4] = 0
        oct_mask_array[oct_mask_array != 0] = 1

        out_name = oct_name.split('_')[1][3:]
        oct_image = Image.fromarray(oct_image_array)
        oct_image.save(os.path.join(output_img_dir, out_name))
        save_ann_png(os.path.join(output_mask_dir, out_name), oct_mask_array)

