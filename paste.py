import os
import random
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from tqdm import tqdm
import cv2

# # ---------------- get support set ----------------
# def save_ann_png(path, mask):
#     """Save a mask as a PNG file with the given palette."""
#     # assert mask.dtype == np.uint8
#     # assert mask.ndim == 2
#     label_colors = {
#         0: (0, 0, 0),        # background
#         1: (220, 50, 47),    # retina (red)
#         2: (60, 201, 230),   # tool (cyan)
#         3: (239, 247, 5),    # artifact (yellow)
#     }

#     palette = []
#     for i in range(4):
#         if i in label_colors:
#             palette.extend(label_colors[i])
#         else:
#             palette.extend((0, 0, 0))  # pad unused indices with black

#     # output_mask = Image.fromarray(mask)
#     mask.putpalette(palette)
#     mask.save(path)

# mask_dir = "./SUP/masks"
# anno_dir = "./SUP/Annotations"
# mask_names = os.listdir(mask_dir)
# for img_name in mask_names:
#     ann_path = os.path.join(anno_dir, img_name)
#     mask = Image.open(os.path.join(mask_dir, img_name))
#     save_ann_png(ann_path, mask)

# # ---------------- get AROI ----------------
# def save_ann_png(path, mask):
#     """Save a mask as a PNG file with the given palette."""
#     assert mask.dtype == np.uint8
#     assert mask.ndim == 2
#     label_colors = {
#         0: (0, 0, 0),        # background
#         1: (220, 50, 47),    # retina (red)
#         2: (60, 201, 230),   # tool (cyan)
#         3: (239, 247, 5),    # artifact (yellow)
#     }

#     palette = []
#     for i in range(4):
#         if i in label_colors:
#             palette.extend(label_colors[i])
#         else:
#             palette.extend((0, 0, 0))  # pad unused indices with black

#     output_mask = Image.fromarray(mask)
#     output_mask.putpalette(palette)
#     output_mask.save(path)

# # Paths
# oct_root = "./AROI - online/24 patient"
# output_img_root = "./AROI/train/JPEGImages"
# output_all_img_root = "./AROI/train/JPEGImagesAll"
# output_mask_root = "./AROI/train/Annotations"

# # Process all patients
# img_size = 512
# patients = sorted(os.listdir(oct_root))
# for patient in tqdm(patients):
#     all_oct_dir = os.path.join(oct_root, patient, "raw/ALL")
#     labelled_oct_dir = os.path.join(oct_root, patient, "raw/labeled")
#     oct_mask_dir = os.path.join(oct_root, patient, "mask/number")

#     p_i = patient[7:]
#     output_img_dir = os.path.join(output_img_root, f"seq_{p_i}")
#     output_mask_dir = os.path.join(output_mask_root, f"seq_{p_i}")
#     output_all_img_dir = os.path.join(output_all_img_root, f"seq_{p_i}")
#     os.makedirs(output_img_dir, exist_ok=True)
#     os.makedirs(output_mask_dir, exist_ok=True)
#     os.makedirs(output_all_img_dir, exist_ok=True)

#     all_oct_images = sorted([os.path.join(all_oct_dir, f) for f in os.listdir(all_oct_dir) if f.endswith(".png")])
#     labelled_oct_images = sorted([os.path.join(labelled_oct_dir, f) for f in os.listdir(labelled_oct_dir) if f.endswith(".png")])
#     oct_mask_names = sorted([f for f in os.listdir(oct_mask_dir) if f.endswith(".png")])
#     assert len(labelled_oct_images) == len(oct_mask_names)
#     num_oct_frames = len(all_oct_images) 

#     for i in range(num_oct_frames):
#         oct_name = os.path.basename(all_oct_images[i])
#         oct_image = Image.open(all_oct_images[i]).convert("L")
#         oct_mask = Image.open(os.path.join(oct_mask_dir, oct_name)).convert("L") if oct_name in oct_mask_names else None
#         out_name = oct_name.split('_')[1][3:]

#         brightness_oct = round(np.random.uniform(0.5, 1.2), 2)
#         oct_image = ImageEnhance.Brightness(oct_image).enhance(brightness_oct)
#         oct_image = oct_image.resize((img_size, img_size), Image.NEAREST)
#         oct_image.save(os.path.join(output_all_img_dir, out_name))

#         if oct_mask:
#             oct_mask_array = np.array(oct_mask.resize((img_size, img_size), Image.NEAREST))
#             oct_mask_array[oct_mask_array == 4] = 0
#             oct_mask_array[oct_mask_array != 0] = 1

#             oct_image.save(os.path.join(output_img_dir, out_name))
#             save_ann_png(os.path.join(output_mask_dir, out_name), oct_mask_array)


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

def augmentation(oct_image, tool_image, oct_mask, tool_mask, s='l'):

    # ---------------- Augmentation ----------------
    # Brightness
    # brightness_oct_i = brightness_oct + round(np.random.random(), 2) * 0.1
    # brightness_tool_i = brightness_tool + round(np.random.random(), 2) * 0.1
    brightness_oct = round(np.random.uniform(0.5, 1.2), 2)
    brightness_tool = round(np.random.uniform(0.5, 1.1), 2)
    oct_image = ImageEnhance.Brightness(oct_image).enhance(brightness_oct)
    tool_image = ImageEnhance.Brightness(tool_image).enhance(brightness_tool)

    # Crop and resize tools
    tool_mask_array = np.array(tool_mask)
    mask = (tool_mask_array == 2) | (tool_mask_array == 3)
    x1, x2 = np.where(mask != 0)[1].min().item(), np.where(mask != 0)[1].max().item()
    crop_box = (x1, 0, x2 + 300, 295)
    tool_image = tool_image.crop(crop_box)
    tool_mask = tool_mask.crop(crop_box)
    translation = np.random.randint(0, 50)
    tool_image = ImageOps.expand(tool_image.resize((img_size, img_size), Image.NEAREST), border=(0, 0, translation, translation), fill=0)
    tool_mask = ImageOps.expand(tool_mask.resize((img_size, img_size), Image.NEAREST), border=(0, 0, translation, translation), fill=0)

    oct_image_array = np.array(oct_image.resize((img_size, img_size), Image.NEAREST))
    oct_mask_array = np.array(oct_mask.resize((img_size, img_size), Image.NEAREST))
    tool_image_array = np.array(tool_image.resize((img_size, img_size), Image.NEAREST))
    tool_mask_array = np.array(tool_mask.resize((img_size, img_size), Image.NEAREST))

    # remove small artifact
    mask = (tool_mask_array == 3).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] < 300:
            tool_mask_array[labels == lbl] = 0

    # translation
    mask = (tool_mask_array == 2) | (tool_mask_array == 3)
    x_min, x_max = np.where(mask != 0)[1].min().item(), np.where(mask != 0)[1].max().item()
    translation = int(((img_size - (x_max - x_min)) / num_oct_frames) * i) + np.random.randint(-3, 3)
    if s == 'l':
        left_pad = max(translation, 0)
        right_pad = max(img_size - (x_max - x_min) - left_pad, 0)
        if (left_pad + right_pad + (x_max - x_min)) > img_size:
            left_pad = img_size - right_pad - (x_max - x_min)
    if s == 'r':
        right_pad = max(translation, 0)
        left_pad = max(img_size - (x_max - x_min) - right_pad, 0)
        if (left_pad + right_pad + (x_max - x_min)) > img_size:
            right_pad = img_size - left_pad - (x_max - x_min)
    assert (left_pad + right_pad + (x_max - x_min)) == img_size
    tool_image_array = tool_image_array[:, x_min:x_max]
    tool_image_array = np.pad(tool_image_array, pad_width=((0, 0), (left_pad, right_pad)), mode='constant', constant_values=0)    # (top, bottom), (left, right)
    
    tool_mask_array = tool_mask_array[:, x_min:x_max]
    tool_mask_array = np.pad(tool_mask_array, pad_width=((0, 0), (left_pad, right_pad)), mode='constant', constant_values=0)    # (top, bottom), (left, right)

    return oct_image_array, oct_mask_array, tool_image_array, tool_mask_array

# Paths
oct_root = "./datasets/AROI - online/24 patient"
tools_root = "./datasets/tools"
output_img_root = "./datasets/Synthetic_iOCT/train/JPEGImages"
output_mask_root = "./datasets/Synthetic_iOCT/train/Annotations"

fs_img_root = "./datasets/Synthetic_iOCT/train/images"
fs_mask_root = "./datasets/Synthetic_iOCT/train/masks"
os.makedirs(fs_img_root, exist_ok=True)
os.makedirs(fs_mask_root, exist_ok=True)

yolo_img_train_root = "./datasets/Synthetic_iOCT/yolo/images/train"
yolo_mask_train_root = "./datasets/Synthetic_iOCT/yolo/masks/train"
yolo_img_val_root = "./datasets/Synthetic_iOCT/yolo/images/val"
yolo_mask_val_root = "./datasets/Synthetic_iOCT/yolo/masks/val"
os.makedirs(yolo_img_train_root, exist_ok=True)
os.makedirs(yolo_mask_train_root, exist_ok=True)
os.makedirs(yolo_img_val_root, exist_ok=True)
os.makedirs(yolo_mask_val_root, exist_ok=True)

# Sort paths
tool_groups = {
    "Cutter_25_1": [f"Cutter_25_1{ch}" for ch in "ab"],
    "Cutter_25_2": [f"Cutter_25_2{ch}" for ch in "ab"],
    "Forceps_23": ["Forceps_23"],
    "Scissors_25": ["Scissors_25_a"],
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
image_idx = 0
start_idx = 0
img_size = 512
patients = sorted(os.listdir(oct_root))
for patient in tqdm(patients):
    labelled_oct_dir = os.path.join(oct_root, patient, "raw/labeled")
    oct_mask_dir = os.path.join(oct_root, patient, "mask/number")

    p_i = int(patient[7:])
    output_img_dir = os.path.join(output_img_root, f"seq_{p_i}")
    output_mask_dir = os.path.join(output_mask_root, f"seq_{p_i}")
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    output_img_dir_F = os.path.join(output_img_root, f"seq_{p_i + len(patients)}")
    output_mask_dir_F = os.path.join(output_mask_root, f"seq_{p_i + len(patients)}")
    os.makedirs(output_img_dir_F, exist_ok=True)
    os.makedirs(output_mask_dir_F, exist_ok=True)

    labelled_oct_images = sorted([os.path.join(labelled_oct_dir, f) for f in os.listdir(labelled_oct_dir) if f.endswith(".png")])
    oct_mask_names = sorted([f for f in os.listdir(oct_mask_dir) if f.endswith(".png")])
    assert len(labelled_oct_images) == len(oct_mask_names)
    num_oct_frames = len(labelled_oct_images) 

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
    np.random.seed(p_i)
    p_flip = np.random.random()
    for i in range(num_oct_frames):
        oct_name = os.path.basename(labelled_oct_images[i])
        oct_image = Image.open(labelled_oct_images[i]).convert("L")
        oct_mask = Image.open(os.path.join(oct_mask_dir, oct_name)).convert("L")
        tool_image = Image.open(tool_images[i]).convert("L")
        tool_mask = Image.open(tool_masks[i]).convert("P")

        oct_image_array, oct_mask_array, tool_image_array, tool_mask_array = augmentation(oct_image, tool_image, oct_mask, tool_mask, s = 'l')

        # ---------------- image paste ----------------
        mask = (tool_mask_array == 2) | (tool_mask_array == 3)
        translation = 10 + np.random.randint(-3, 3)
        x1, x2 = np.where(mask != 0)[1].min().item() - translation, np.where(mask != 0)[1].max().item() + translation
        x1 = 0 if x1 < 20 else x1
        x2 = img_size if abs(img_size - x2) < 20 else x2
        oct_image_array[:, x1:x2] = 0
        oct_image_array[mask] = tool_image_array[mask]

        # ---------------- mask paste ----------------
        oct_mask_array[oct_mask_array == 4] = 0
        oct_mask_array[oct_mask_array != 0] = 1
        oct_mask_array[:, x1:x2] = 0
        oct_mask_array[mask] = tool_mask_array[mask]

        # # for SAM2 dataset
        # out_name = oct_name.split('_')[1][3:]
        # oct_image_new = Image.fromarray(oct_image_array)
        # oct_image_new.save(os.path.join(output_img_dir, out_name))
        # save_ann_png(os.path.join(output_mask_dir, out_name), oct_mask_array)
        
        # # for YOLO dataset
        # yolo_img_name = f"{image_idx:04d}.png"
        # yolo_mask_name = f"{image_idx:04d}.png"
        # oct_image_new = Image.fromarray(oct_image_array)
        # oct_mask_new = Image.fromarray(oct_mask_array)
        # if image_idx % 10 == 0:
        #     oct_image_new.save(os.path.join(yolo_img_val_root, yolo_img_name))
        #     oct_mask_new.save(os.path.join(yolo_mask_val_root, yolo_mask_name))
        # else:
        #     oct_image_new.save(os.path.join(yolo_img_train_root, yolo_img_name))
        #     oct_mask_new.save(os.path.join(yolo_mask_train_root, yolo_mask_name))

        # for few shot learning dataset
        img_name = f"{image_idx:04d}.png"
        mask_name = f"{image_idx:04d}.png"
        oct_image_new = Image.fromarray(oct_image_array)
        oct_mask_new = Image.fromarray(oct_mask_array)
        oct_image_new.save(os.path.join(fs_img_root, img_name))
        oct_mask_new.save(os.path.join(fs_mask_root, mask_name))
        image_idx += 1


        tool_image_F = tool_image.transpose(Image.FLIP_LEFT_RIGHT)
        tool_mask_F = tool_mask.transpose(Image.FLIP_LEFT_RIGHT)
        oct_image_array, oct_mask_array, tool_image_array, tool_mask_array = augmentation(oct_image, tool_image_F, oct_mask, tool_mask_F, s = 'r')

        # ---------------- image paste ----------------
        mask = (tool_mask_array == 2) | (tool_mask_array == 3)
        translation = 10 + np.random.randint(-3, 3)
        x1, x2 = np.where(mask != 0)[1].min().item() - translation, np.where(mask != 0)[1].max().item() + translation
        x1 = 0 if x1 < 20 else x1
        x2 = img_size if abs(img_size - x2) < 20 else x2
        oct_image_array[:, x1:x2] = 0
        oct_image_array[mask] = tool_image_array[mask]

        # ---------------- mask paste ----------------
        oct_mask_array[oct_mask_array == 4] = 0
        oct_mask_array[oct_mask_array != 0] = 1
        oct_mask_array[:, x1:x2] = 0
        oct_mask_array[mask] = tool_mask_array[mask]

        # # for SAM2 dataset
        # out_name = oct_name.split('_')[1][3:]
        # oct_image_new = Image.fromarray(oct_image_array)
        # oct_image_new.save(os.path.join(output_img_dir_F, out_name))
        # save_ann_png(os.path.join(output_mask_dir_F, out_name), oct_mask_array)
        
        # # for YOLO dataset
        # yolo_img_name = f"{image_idx:04d}.png"
        # yolo_mask_name = f"{image_idx:04d}.png"
        # oct_image_new = Image.fromarray(oct_image_array)
        # oct_mask_new = Image.fromarray(oct_mask_array)
        # if image_idx % 10 == 0:
        #     oct_image_new.save(os.path.join(yolo_img_val_root, yolo_img_name))
        #     oct_mask_new.save(os.path.join(yolo_mask_val_root, yolo_mask_name))
        # else:
        #     oct_image_new.save(os.path.join(yolo_img_train_root, yolo_img_name))
        #     oct_mask_new.save(os.path.join(yolo_mask_train_root, yolo_mask_name))
        
        # for few shot learning dataset
        img_name = f"{image_idx:04d}.png"
        mask_name = f"{image_idx:04d}.png"
        oct_image_new = Image.fromarray(oct_image_array)
        oct_mask_new = Image.fromarray(oct_mask_array)
        oct_image_new.save(os.path.join(fs_img_root, img_name))
        oct_mask_new.save(os.path.join(fs_mask_root, mask_name))
        image_idx += 1
