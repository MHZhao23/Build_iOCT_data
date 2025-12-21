# Convert amos to amos22_3d by amos22_3d.py (following https://github.com/yhygao/CBIM-Medical-Image-Segmentation/tree/main/dataset_conversion)
# Convert amos22_3d to amos22_2d by amos22_2d.py
# Create json file
import os
import json
import numpy as np
from PIL import Image
from pycocotools import mask as mask_util
from pathlib import Path


# CATEGORIES = [
#     {"id": 1, "name": "spleen"},
#     {"id": 2, "name": "right kidney"},
#     {"id": 3, "name": "left kidney"},
#     {"id": 4, "name": "gall bladder"},
#     {"id": 5, "name": "esophagus"},
#     {"id": 6, "name": "liver"},
#     {"id": 7, "name": "stomach"},
#     {"id": 8, "name": "aorta"},
#     {"id": 9, "name": "postcava"},
#     {"id": 10, "name": "pancreas"},
#     {"id": 11, "name": "right adrenal gland"},
#     {"id": 12, "name": "left adrenal gland"},
#     {"id": 13, "name": "duodenum"},
#     {"id": 14, "name": "bladder"},
#     {"id": 15, "name": "prostate/uterus"},
# ]

ORGANS = [
    "spleen",
    "right kidney",
    "left kidney",
    "gall bladder",
    "esophagus",
    "liver",
    "stomach",
    "aorta",
    "postcava",
    "pancreas",
    "right adrenal gland",
    "left adrenal gland",
    "duodenum",
    "bladder",
    "prostate",
]

CATEGORIES = []
cid = 1
for modality in ["CT", "MR"]:
    for organ in ORGANS:
        CATEGORIES.append({"id": cid, "name": f"{modality} {organ}"})
        cid += 1
print(CATEGORIES)

def case_modality(case_id: int):
    return "CT" if case_id < 500 else "MR"

def make_text_prompt(case_id, organ_name):
    return f"{case_modality(case_id)} {organ_name}"

def convert_split(split):
    root_2d = "./other/AMOS/amos22_2d"
    img_dir = Path(root_2d) / "images" / split
    ann_dir = Path(root_2d) / "masks" / split
    out_json = Path(root_2d) / "masks" / f"{split}.json"

    images, annotations = [], []
    img_id, ann_id = 1, 1

    for image_path in sorted(img_dir.glob("*.png")):
        case_id = int(image_path.stem.split("_")[1])
        img = Image.open(image_path)
        w, h = img.size
        mask = np.array(Image.open(ann_dir / image_path.name))

        images.append({
            "id": img_id,
            "file_name": image_path.name,
            "width": w,
            "height": h,
            "is_instance_exhaustive": True,   # ⭐ REQUIRED by SAM3 evaluator
            "is_pixel_exhaustive": True       # (optional but recommended)
        })

        # For each class, create one annotation (semantic mask)
        for cls_id in range(1, len(ORGANS) + 1):
            offset = 0 if case_id < 500 else 15
            bin_mask = (mask == cls_id).astype(np.uint8)
            if bin_mask.sum() == 0:
                continue

            rle = mask_util.encode(np.asfortranarray(bin_mask))
            rle["counts"] = rle["counts"].decode("ascii")

            ys, xs = np.where(bin_mask)
            bbox = [
                float(xs.min()),
                float(ys.min()),
                float(xs.max() - xs.min() + 1),
                float(ys.max() - ys.min() + 1),
            ]

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id + offset,
                "segmentation": rle,
                "area": float(bin_mask.sum()),
                "bbox": bbox,
                "iscrowd": 0,
            })
            ann_id += 1

        img_id += 1

    dataset = {
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES,
    }

    with open(out_json, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"[{split}] saved {out_json}, #images={len(images)}, #annotations={len(annotations)}")


# Convert both train and val
convert_split("train")
convert_split("val")
