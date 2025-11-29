import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
import matplotlib.pyplot as plt

def save_ann_png(path, mask):
    """Save a mask as a PNG file with the given palette."""
    # assert mask.dtype == np.uint8
    # assert mask.ndim == 2
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

    # output_mask = Image.fromarray(mask)
    mask.putpalette(palette)
    mask.save(path)


# === CONFIG ===
for root_path in ["./videos/horizontal_data"]: # ["./videos/horizontal_data", "./videos/vertical_data"]
# for root_path in ["./datasets/test_data/horizontal_data", "./datasets/test_data/vertical_data"]:
    video_folders = [f for f in os.listdir(root_path)
                    if os.path.isdir(os.path.join(root_path, f)) and (f.startswith('OS') or f.startswith('OD'))]

    for video_folder in video_folders:
        video_path = os.path.join(root_path, video_folder)
        image_path = os.path.join(video_path, "image")
        json_path = os.path.join(video_path, "json")
        mask_path = os.path.join(video_path, "mask")
        filenames = [os.path.splitext(f)[0] for f in os.listdir(image_path)]
        print(video_folder, len(filenames))
        os.makedirs(mask_path, exist_ok=True)

        # plot_path = os.path.join(video_path, "plot")
        # os.makedirs(plot_path, exist_ok=True)

        for filename in filenames:

            # === LOAD JSON ===
            with open(os.path.join(json_path, f"{filename}.json"), "r") as f:
                data = json.load(f)

            image = Image.open(os.path.join(image_path, f"{filename}.png"))
            image_size = image.size  # (width, height)
            label_map = {
                "Tissue": 1,
                "Tool": 2,
                "Artifact": 3,
            }

            # === DRAW MASK ===
            mask = Image.new('L', image_size, 0)
            draw = ImageDraw.Draw(mask)

            for shape in data["shapes"]:
                label = shape["label"]
                polygon = shape["points"]
                polygon = [(int(x), int(y)) for x, y in polygon]
                value = label_map.get(label, 0)
                draw.polygon(polygon, outline=value, fill=value)

            # # === SHOW MASK ===
            # cmap = plt.get_cmap("tab10")
            # image_np, mask_np = np.array(image), np.array(mask)
            # colored_mask = np.zeros((*mask.size[::-1], 3), dtype=np.float32)
            # for label_name, label_id in label_map.items():
            #     tmp = (mask_np == label_id)
            #     if tmp.any():
            #         colored_mask[tmp] = cmap(label_id / 3)[:3]
            # colored_mask = (colored_mask * 255).astype(np.uint8)
            # alpha = 0.4
            # colored_mask = ((1 - alpha) * image_np + alpha * colored_mask).astype(np.uint8)
            # Image.fromarray(colored_mask).save(os.path.join(plot_path, f"{filename}.png"))

            # # === SHOW BOUNDARY ===
            # h, w = mask_np.shape
            # top_boundary_1 = np.full(w, -1)
            # bottom_boundary_1 = np.full(w, -1)
            # for x in range(w):
            #     column = mask_np[:, x]
            #     idxs = np.where(column == 1)[0]
            #     if len(idxs) > 0:
            #         top_boundary_1[x] = idxs[0]
            #         bottom_boundary_1[x] = idxs[-1]

            # plt.figure(figsize=(12, 5))
            # plt.suptitle(f"{np.unique(mask)}")
            # # plt.subplot(1, 3, 1)
            # # plt.imshow(image, cmap='gray')
            # # plt.axis('off')

            # plt.subplot(1, 2, 1)
            # plt.imshow(image, cmap='gray')  # or use cmap='jet' to see layers in color
            # plt.plot(range(w), top_boundary_1, color='red', label='Top of Layer 1', lw=0.5)
            # plt.plot(range(w), bottom_boundary_1, color='blue', label='Bottom of Layer 3', lw=0.5)
            # plt.axis('off')

            # plt.subplot(1, 2, 2)
            # plt.imshow(np.array(mask))
            # plt.axis('off')
            # plt.tight_layout()
            # plt.savefig(os.path.join(plot_path, f"plot_{filename}.png"), dpi=200)
            # # plt.show()
            # plt.close()

            # === SAVE MASK ===
            save_ann_png(os.path.join(mask_path, f"{filename}.png"), mask)
