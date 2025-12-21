import os
import numpy as np
import SimpleITK as sitk
from PIL import Image
from pathlib import Path

# ---------- ADD PALETTE ----------
LABEL_COLORS = {
    0:  (0, 0, 0),
    1:  (220, 20, 60),
    2:  (65, 105, 225),
    3:  (34, 139, 34),
    4:  (255, 165, 0),
    5:  (138, 43, 226),
    6:  (0, 206, 209),
    7:  (255, 215, 0),
    8:  (199, 21, 133),
    9:  (0, 191, 255),
    10: (154, 205, 50),
    11: (255, 99, 71),
    12: (72, 61, 139),
    13: (0, 128, 128),
    14: (218, 112, 214),
    15: (210, 180, 140),
}

def add_palette(mask: np.ndarray):
    mask = mask.astype(np.uint8)
    out = Image.fromarray(mask)

    palette = []
    for i in range(16):
        palette.extend(LABEL_COLORS.get(i, (0, 0, 0)))
    palette.extend((0, 0, 0))
    # palette.extend([0, 0, 0] * (256 - 16))

    out.putpalette(palette)

    return out

def convert_split_amos(
    split,
    amos_root,
    out_root,
    num_slices_limit=None   # 可选：限制 slice 数
):
    img_dir = Path(amos_root) / f"images{split.capitalize()[:2]}"
    lab_dir = Path(amos_root) / f"labels{split.capitalize()[:2]}"

    out_img = Path(out_root) / "images" / split
    out_lab = Path(out_root) / "masks" / split
    out_img.mkdir(parents=True)
    out_lab.mkdir(parents=True)

    for img_nii in sorted(img_dir.glob("*.nii.gz")):
        case_id = img_nii.stem.split(".")[0]  # XXXX

        lab_nii = lab_dir / img_nii.name
        assert lab_nii.exists(), f"{lab_nii} doesn't exist"

        img = sitk.ReadImage(str(img_nii))
        lab = sitk.ReadImage(str(lab_nii))

        img_vol = sitk.GetArrayFromImage(img)   # (Z, Y, X)
        lab_vol = sitk.GetArrayFromImage(lab)

        assert img_vol.shape == lab_vol.shape

        depth = img_vol.shape[0]
        for z in range(depth):
            if num_slices_limit and z >= num_slices_limit:
                break

            img_slice = img_vol[z]
            lab_slice = lab_vol[z]

            if np.sum(lab_slice) < 1e-5:
                # print(f"[{split}] amos_{case_id} skip slices {z}")
                continue

            if int(case_id) < 500:
                # --- CT windowing ---
                # following FAMNet_main\data\ABD\ABDOMEN_CT\intensity_normalization.py
                img_slice = np.clip(img_slice, -125, 275)
                img_slice = (img_slice - img_slice.min()) / (img_slice.ptp() + 1e-5)
                img_slice = (img_slice * 255).astype(np.uint8)
            else:
                # --- MR windowing ---
                # following FAMNet_main\data\ABD\ABDOMEN_MR\image_normalize.ipynb
                HIST_CUT_TOP = 0.5
                hir = float(np.percentile(img_slice, 100.0 - HIST_CUT_TOP))
                img_slice[img_slice > hir] = hir
                img_slice = img_slice.astype(np.uint8)

            fname = f"amos_{case_id}_{z:03d}.png"

            Image.fromarray(img_slice).save(out_img / fname)
            add_palette(lab_slice).save(out_lab / fname)

        print(f"[{split}] processed amos_{case_id}, slices={depth}, z={z}")


if __name__ == '__main__':
    convert_split_amos("train", "./other/AMOS/amos22_3d", "./other/AMOS/amos22_2d")
    convert_split_amos("val", "./other/AMOS/amos22_3d", "./other/AMOS/amos22_2d")
    # convert_split_amos("test", "./amos22_3d", "./amos22_2d")

    # img = sitk.ReadImage("./other/AMOS/amos22_3d/imagesTr/0001.nii.gz")
    # lab = sitk.ReadImage("./other/AMOS/amos22_3d/labelsTr/0001.nii.gz")

    # print("=== IMAGE ===")
    # print("Size     :", img.GetSize())
    # print("Spacing  :", img.GetSpacing())
    # print("Origin   :", img.GetOrigin())
    # print("Direction:", img.GetDirection())

    # print("\n=== LABEL ===")
    # print("Size     :", lab.GetSize())
    # print("Spacing  :", lab.GetSpacing())
    # print("Origin   :", lab.GetOrigin())
    # print("Direction:", lab.GetDirection())

    # import matplotlib.pyplot as plt
    # # 选一个 label 非零最多的 slice
    # lab_arr = sitk.GetArrayFromImage(lab)
    # z = np.argmax(lab_arr.sum(axis=(1,2)))

    # img_slice = sitk.GetArrayFromImage(img)[z]
    # lab_slice = lab_arr[z]
    
    # img_slice = np.clip(img_slice, -125, 275)
    # img_slice = (img_slice - img_slice.min()) / (img_slice.ptp() + 1e-5)
    # img_slice = (img_slice * 255).astype(np.uint8)

    # plt.figure(figsize=(10,4))

    # plt.subplot(1,2,1)
    # plt.title("Image")
    # plt.imshow(img_slice, cmap="gray")
    # plt.axis("off")

    # plt.subplot(1,2,2)
    # plt.title("Overlay")
    # plt.imshow(img_slice, cmap="gray")
    # plt.imshow(lab_slice, alpha=0.5)
    # plt.axis("off")

    # plt.show()
