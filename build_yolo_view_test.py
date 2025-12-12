import os
import shutil
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

for view in ['horizontal', 'vertical']:
    root_dir = f'./datasets/test_data/{view}_data'
    output_base = f'./datasets/iOCT_FAMNet/yolo/{view}'

    # ============ Create test dataset ============
    source_folders = ['image', 'mask']
    target_folders = ['images', 'masks']
    for target_sub in target_folders:
        os.makedirs(os.path.join(output_base, target_sub, 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_base, target_sub, 'val'), exist_ok=True)
        os.makedirs(os.path.join(output_base, target_sub, 'test'), exist_ok=True)

    start_idx = 0
    video_folders = [f for f in os.listdir(root_dir)
                    if os.path.isdir(os.path.join(root_dir, f)) and (f.startswith('OS') or f.startswith('OD'))]
    for video_folder in video_folders:
        video_path = os.path.join(root_dir, video_folder)
        if os.path.isdir(video_path):
            fnames = os.listdir(os.path.join(video_path, 'image'))
            for fname in fnames:
                for (source_sub, target_sub) in zip(source_folders, target_folders):
                    source_path = os.path.join(video_path, source_sub)
                    print(source_path, len(os.listdir(source_path)))
                    src = os.path.join(source_path, fname)
                    dst_name = f"{start_idx:04d}.png"
                    dst = os.path.join(output_base, target_sub, 'test', dst_name)
                    shutil.copy(src, dst)
                start_idx += 1

    # ============ Visualize test dataset ============
    image_dir = output_base + '/images/test'
    mask_dir = output_base + '/masks/test'

    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    num_images = len(image_files)
    num_masks = len(mask_files)
    print(f"Number of images: {num_images}, Number of masks: {num_masks}")

    assert num_images == num_masks, "Mismatch between number of images and masks."

    indices = np.linspace(0, num_images - 1, 120, dtype=int)

    fig, axs = plt.subplots(10, 12, figsize=(15, 10))
    axs = axs.flatten()
    label_map = {
        "Tissue": 1,
        "Tool": 2,
        "Artifact": 3,
    }

    for i, idx in enumerate(indices):
        image = Image.open(os.path.join(image_dir, image_files[idx]))
        mask = Image.open(os.path.join(mask_dir, mask_files[idx]))

        cmap = plt.get_cmap("tab10")
        image_np, mask_np = np.array(image), np.array(mask)
        colored_mask = np.zeros((*mask.size[::-1], 3), dtype=np.float32)
        for _, label_id in label_map.items():
            tmp = (mask_np == label_id)
            if tmp.any():
                colored_mask[tmp] = cmap(label_id / 3)[:3]
        colored_mask = (colored_mask * 255).astype(np.uint8)
        alpha = 0.4
        colored_mask = ((1 - alpha) * image_np + alpha * colored_mask).astype(np.uint8)
        # Image.fromarray(colored_mask).save(os.path.join(plot_path, f"{filename}.png"))

        axs[i].imshow(colored_mask)
        axs[i].axis('off')
        axs[i].set_title(mask_files[idx], fontsize=8)

    plt.tight_layout()
    plt.show()
    plt.close()
