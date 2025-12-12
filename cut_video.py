import cv2
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path
import shutil


if __name__ == '__main__' :
    video_root = './videos'

    # ------------------ raw_data ------------------
    filenames = [f for f in os.listdir(video_root) if f.endswith('.mp4')]

    for i, filename in enumerate(filenames):
        folder_name = filename.split(".")[0] + '_test'
        frame_dir = os.path.join('./videos/raw_data', folder_name)
        os.makedirs(frame_dir, exist_ok=True)

        video_capture = cv2.VideoCapture(os.path.join(video_root, filename))
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(fps, width, height)
        ok = video_capture.isOpened()
        if not ok:
            print ('Cannot read video file')
            sys.exit()

        frame_id = 0
        # OS-2025-04-29_132446_test: 9:18 11:30
        # OD-2025-04-29_150558: 22:32 23:24
        # OD-2025-04-29_165526: 00:00 01:04
        # OS-2025-05-20_102739: 00:20 01:04
        # OS-2025-05-20_142838: 01:39 03:08
        # OS-2025-05-27_112744: 01:02 02:22
        # OS-2025-05-27_140240: 01:18 02:22
        # OS-2025-06-10_115259: 00:59 03:51
        mins_start = 0
        sec_start = 59
        mins_end = 3
        sec_end = 51

        creat_test_video = True
        while True:
            if creat_test_video: break

            # Read a new frame
            ok, frame = video_capture.read()
            if not ok or frame_id > (mins_end * 60 + sec_end) * 25:
                print(frame_id, len(os.listdir(frame_dir)))
                break
            
            if frame_id < (mins_start * 60 + sec_start) * 25:
                frame_id += 1
                continue

            frame_name = f"{frame_id:04d}.png"
            save_dir = os.path.join(frame_dir, frame_name)

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


        if creat_test_video:
            img_files = sorted([p for p in os.listdir(frame_dir)])
            video_path = os.path.join('./videos/videos', folder_name + '.mp4')
            testdata_dir = os.path.join('./videos/horizontal_data', folder_name)
            output_dir = os.path.join(testdata_dir, "frames")
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(os.path.join(testdata_dir, "image"), exist_ok=True)
            os.makedirs(os.path.join(testdata_dir, "json"), exist_ok=True)
            os.makedirs(os.path.join(testdata_dir, "mask"), exist_ok=True)
            os.makedirs(os.path.join(testdata_dir, "plot"), exist_ok=True)
            print(folder_name)

            video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"),  fps, (615, 295))

            img_id = 0
            for img_name in img_files:
                src = os.path.join(frame_dir, img_name)
                dst = os.path.join(output_dir, f"{img_id:04d}.png")
                shutil.copy(src, dst)

                frame = cv2.imread(src)
                video.write(frame)

                img_id += 1

            video.release()
            print(f"Video saved to {video_path} with {img_id} frames")
