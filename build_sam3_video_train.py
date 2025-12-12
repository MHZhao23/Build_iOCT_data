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

# Build tool image/mask lists
def collect_tool_paths(group_folders):
    img_paths, mask_paths = [], []
    for name in group_folders:
        img_dir = os.path.join(tools_root, "images", name)
        mask_dir = os.path.join(tools_root, "masks", name)
        imgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")])
        masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")])
        img_paths.extend(imgs)
        mask_paths.extend(masks)
    return img_paths, mask_paths

# Paths
oct_root = "./datasets/AROI - online/24 patient"
tools_root = "./datasets/tools"
train_img_root = "./datasets/SAM3_iOCT/train/JPEGImages"
train_mask_root = "./datasets/SAM3_iOCT/train/Annotations"
val_img_root = "./datasets/SAM3_iOCT/val/JPEGImages"
val_mask_root = "./datasets/SAM3_iOCT/val/Annotations"

# Sort paths
tool_groups = {
    "Forceps_23": ["Forceps_23"],
    "Scissors_25": [f"Scissors_25_{ch}" for ch in "ab"],
    "Cutter_25_1": [f"Cutter_25_1{ch}" for ch in "ab"],
    "Cutter_25_2": [f"Cutter_25_2{ch}" for ch in "abcdefghijk"],
}
tool_sequences = {group_name: collect_tool_paths(group_folders) for group_name, group_folders in tool_groups.items()}
    
all_image_paths = []
all_mask_paths = []
split_idx = [0]
for group_name, (img_paths, mask_paths) in tool_sequences.items():
    split_idx.append(split_idx[-1] + len(img_paths))
    all_image_paths.extend(img_paths)
    all_mask_paths.extend(mask_paths)
    assert len(all_image_paths) == len(all_mask_paths)

# Process all patients
start_idx = 0
img_size = 512
patients = sorted(os.listdir(oct_root))
np.random.seed(42)
for patient in tqdm(patients):
    all_oct_dir = os.path.join(oct_root, patient, "raw/ALL")
    labelled_oct_dir = os.path.join(oct_root, patient, "raw/labeled")
    oct_mask_dir = os.path.join(oct_root, patient, "mask/number")

    p_i = patient[7:]
    if int(p_i) > 20:
        output_img_dir = os.path.join(val_img_root, f"seq_{p_i}")
        output_mask_dir = os.path.join(val_mask_root, f"seq_{p_i}")
    else:
        output_img_dir = os.path.join(train_img_root, f"seq_{p_i}")
        output_mask_dir = os.path.join(train_mask_root, f"seq_{p_i}")
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    all_oct_images = sorted([os.path.join(all_oct_dir, f) for f in os.listdir(all_oct_dir) if f.endswith(".png")])
    labelled_oct_images = sorted([os.path.join(labelled_oct_dir, f) for f in os.listdir(labelled_oct_dir) if f.endswith(".png")])
    oct_mask_names = sorted([f for f in os.listdir(oct_mask_dir) if f.endswith(".png")])
    assert len(labelled_oct_images) == len(oct_mask_names)
    num_oct_frames = len(all_oct_images) 

    end_idx = start_idx + num_oct_frames
    for j in range(len(split_idx)-1):
        if (start_idx >= split_idx[j]) and (start_idx < split_idx[j+1]):
            if (end_idx <= split_idx[j+1]):
                break
            elif ((end_idx - split_idx[j+1]) < 50) or (j == (len(split_idx) - 2)):
                end_idx = split_idx[j+1]
                start_idx = end_idx - num_oct_frames
                break
            else:
                start_idx = split_idx[j+1]
                end_idx = start_idx + num_oct_frames
                break
        else:
            if j == (len(split_idx)-2):
                end_idx = split_idx[j+1]
                start_idx = end_idx - num_oct_frames

    tool_images = all_image_paths[start_idx:end_idx]
    tool_masks = all_mask_paths[start_idx:end_idx]
    assert len(tool_masks) == len(tool_images)
    start_idx += num_oct_frames 

    # Generate synthetic frames
    # brightness_oct = np.random.random_integers(4, 11) * 0.1
    # brightness_tool = np.random.random_integers(4, 10) * 0.1
    p_flip = np.random.random()
    for i in range(num_oct_frames):
        oct_name = os.path.basename(all_oct_images[i])
        oct_image = Image.open(all_oct_images[i]).convert("L")
        oct_mask = Image.open(os.path.join(oct_mask_dir, oct_name)).convert("L") if oct_name in oct_mask_names else None
        tool_image = Image.open(tool_images[i]).convert("L")
        tool_mask = Image.open(tool_masks[i]).convert("P")

        # ---------------- Augmentation ----------------
        # Brightness
        # brightness_oct_i = brightness_oct + round(np.random.random(), 2) * 0.1
        # brightness_tool_i = brightness_tool + round(np.random.random(), 2) * 0.1
        brightness_oct = round(np.random.uniform(0.5, 1.2), 2)
        brightness_tool = round(np.random.uniform(0.5, 1.1), 2)
        oct_image = ImageEnhance.Brightness(oct_image).enhance(brightness_oct)
        tool_image = ImageEnhance.Brightness(tool_image).enhance(brightness_tool)

        # Crop tools
        tool_mask_array = np.array(tool_mask)
        translation = int((250 / num_oct_frames) * i + 50) + np.random.randint(-3, 3) # np.random.randint(50, 150)
        mask = (tool_mask_array == 2) | (tool_mask_array == 3)
        x1, x2 = np.where(mask != 0)[1].min().item(), np.where(mask != 0)[1].max().item()
        crop_box = (x1 - translation, 0, x2 + (300 - translation), 295)
        tool_image = tool_image.crop(crop_box)
        tool_mask = tool_mask.crop(crop_box)

        if p_flip > 0.5:
            tool_image = tool_image.transpose(Image.FLIP_LEFT_RIGHT)
            tool_mask = tool_mask.transpose(Image.FLIP_LEFT_RIGHT)

        oct_image_array = np.array(oct_image.resize((img_size, img_size), Image.NEAREST))
        oct_mask_array = np.array(oct_mask.resize((img_size, img_size), Image.NEAREST)) if oct_mask else None
        tool_image_array = np.array(tool_image.resize((img_size, img_size), Image.NEAREST))
        tool_mask_array = np.array(tool_mask.resize((img_size, img_size), Image.NEAREST))

        # ---------------- image paste ----------------
        # remove small connected components for class == 3
        mask = (tool_mask_array == 3).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] < 50:   # small artifact
                tool_mask_array[labels == lbl] = 0

        mask = (tool_mask_array == 2) | (tool_mask_array == 3)
        translation = 10 + np.random.randint(-3, 3)
        x1, x2 = np.where(mask != 0)[1].min().item() - translation, np.where(mask != 0)[1].max().item() + translation
        x1 = 0 if x1 < 20 else x1
        x2 = img_size if abs(img_size - x2) < 20 else x2
        oct_image_array[:, x1:x2] = 0
        oct_image_array[mask] = tool_image_array[mask]

        # ---------------- mask paste ----------------
        if oct_mask:
            oct_mask_array[oct_mask_array == 4] = 0
            oct_mask_array[oct_mask_array != 0] = 1
            oct_mask_array[:, x1:x2] = 0
            oct_mask_array[mask] = tool_mask_array[mask]

        out_name = oct_name.split('_')[1][3:]
        oct_image = Image.fromarray(oct_image_array)
        if oct_mask: 
            oct_image.save(os.path.join(output_img_dir, out_name))
            save_ann_png(os.path.join(output_mask_dir, out_name), oct_mask_array)

