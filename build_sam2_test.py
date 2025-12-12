import os
import shutil
import cv2
import sys
from tqdm import tqdm

# for view in ['horizontal', 'vertical']:
# view = 'horizontal' # 'horizontal' 'vertical'
video_root = './videos/videos'
root_dir = f'./videos/horizontal_data'
output_base = f'./datasets/iOCT/test'

# ============ Create test dataset ============
source_folders = ['mask', 'frames']
target_folders = ['Annotations', 'JPEGImages']

video_folders = [f for f in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, f)) and (f.startswith('OS') or f.startswith('OD'))]

for i, video_folder in enumerate(video_folders):
    raw_video_path = os.path.join(video_root, video_folder+".mp4")
    video_capture = cv2.VideoCapture(raw_video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok = video_capture.isOpened()
    if not ok:
        print ('Cannot read video file')
        sys.exit()
    
    video_path = os.path.join(root_dir, video_folder)
    print(video_path, fps, width, height)

    # Copy images
    if os.path.isdir(video_path):
        # for source_folder, target_folder in zip(source_folders, target_folders):
        seq_path = os.path.join(output_base, 'Annotations', f'seq_{i}')
        os.makedirs(seq_path, exist_ok=True)
        source_path = os.path.join(video_path, 'mask')
        if os.path.exists(source_path):
            for fname in os.listdir(source_path):
                src = os.path.join(source_path, fname)
                dst = os.path.join(seq_path, fname)
                shutil.copy(src, dst)
        print(seq_path, len(os.listdir(seq_path)))

    # copy frames
    if width != 1280:
        seq_path = os.path.join(output_base, 'JPEGImages', f'seq_{i}')
        os.makedirs(seq_path, exist_ok=True)
        source_path = os.path.join(video_path, 'frames')
        if os.path.exists(source_path):
            for fname in os.listdir(source_path):
                src = os.path.join(source_path, fname)
                dst = os.path.join(seq_path, fname)
                shutil.copy(src, dst)
        print(seq_path, len(os.listdir(seq_path)), '\n')

    else:
        seq_path = os.path.join(output_base, 'JPEGImages', f'seq_{i}')
        os.makedirs(seq_path, exist_ok=True)
        frame_id = 0
        while True:
            # Read a new frame
            ok, frame = video_capture.read()
            if not ok or (frame_id > int(fname.split('.')[0]) + 5):
                print(seq_path, len(os.listdir(seq_path)), '\n')
                break

            frame_name = f"{frame_id:04d}.png"
            save_dir = os.path.join(seq_path, frame_name)

            # cut upper iOCT (horizontal)
            horizontal_ioct = frame[58:353, 653:1268, :]
            horizontal_ioct[8:10, 8:60, :] = horizontal_ioct[6:8, 8:60, :] # up
            horizontal_ioct[58:60, 8:60, :] = horizontal_ioct[56:58, 8:60, :] # down
            horizontal_ioct[8:60, 8:10, :] = horizontal_ioct[8:60, 6:8, :] # left
            horizontal_ioct[8:60, 58:60, :] = horizontal_ioct[8:60, 56:58, :] # right
            horizontal_ioct[94:199, 605:609, :] = horizontal_ioct[94:199, 609:613, :] # right
            horizontal_ioct[32:36, 8:60, :] = horizontal_ioct[36:40, 8:60, :] # blue
            horizontal_ioct[29:38, 51:56, :] = horizontal_ioct[29:38, 46:51, :]

            horizontal_ioct[151:156, 583:600, :] = horizontal_ioct[151:156, 566:583, :] # up
            horizontal_ioct[135:144, 591:593, :] = horizontal_ioct[135:144, 593:595, :] # up
            horizontal_ioct[136:139, 589:591, :] = horizontal_ioct[136:139, 587:589, :] # up
            cv2.imwrite(save_dir, horizontal_ioct)

            frame_id += 1


# ============ Create prompt dataset ============
gt_root = "./datasets/iOCT/test/Annotations"
prompt_root = "./datasets/iOCT/prompt/Annotations"

image_idx = 0
seqs = sorted(os.listdir(gt_root))
print(seqs)
for seq in tqdm(seqs):
    os.makedirs(os.path.join(prompt_root, seq), exist_ok=True)
    fs_name = os.listdir(os.path.join(gt_root, seq))[0]
    src = os.path.join(gt_root, seq, fs_name)
    dst = os.path.join(prompt_root, seq, fs_name)
    shutil.copy(src, dst)
