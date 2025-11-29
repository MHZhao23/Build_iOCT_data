import cv2
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path
import shutil


if __name__ == '__main__' :
    video_root = './videos'

    # ------------------ horizontal_data ------------------
    testdata_root = './videos/horizontal_data'
    filenames = [f for f in os.listdir(video_root) if f.endswith('.mp4')]

    for i, filename in enumerate(filenames):
        folder_name = filename.split(".")[0]
        # folder_name = filename.split("test")[0]
        # folder_idx = filename.split("test")[1].split(".")[0]
        # output_dir = os.path.join(output_root, folder_name)
        testdata_dir = os.path.join(testdata_root, folder_name)
        output_dir = os.path.join(testdata_dir, "frames")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(testdata_dir, "image"), exist_ok=True) # selected image
        os.makedirs(os.path.join(testdata_dir, "json"), exist_ok=True) # json
        os.makedirs(os.path.join(testdata_dir, "mask"), exist_ok=True) # converted mask
        os.makedirs(os.path.join(testdata_dir, "plot"), exist_ok=True) # plot
        print(folder_name)

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
        while True:
            # Read a new frame
            ok, frame = video_capture.read()
            if not ok:
                print(frame_id, len(os.listdir(output_dir)))
                break
            
            if frame_id % 5 != 0:
                frame_id += 1
                continue

            frame_name = f"{frame_id:04d}.png"
            save_dir = os.path.join(output_dir, frame_name)

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


    # ------------------ vertical_data ------------------
    testdata_root = './videos/vertical_data'
    filenames = [f for f in os.listdir(video_root) if f.endswith('.mp4')]

    for i, filename in enumerate(filenames):
        folder_name = filename.split(".")[0]
        testdata_dir = os.path.join(testdata_root, folder_name)
        output_dir = os.path.join(testdata_dir, "frames")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(testdata_dir, "image"), exist_ok=True) # selected image
        os.makedirs(os.path.join(testdata_dir, "json"), exist_ok=True) # json
        os.makedirs(os.path.join(testdata_dir, "mask"), exist_ok=True) # converted mask
        os.makedirs(os.path.join(testdata_dir, "plot"), exist_ok=True) # plot
        print(folder_name)

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
        while True:
            # Read a new frame
            ok, frame = video_capture.read()
            if not ok:
                print(frame_id, len(os.listdir(output_dir)))
                break
            
            if frame_id % 5 != 0:
                frame_id += 1
                continue

            frame_name = f"{frame_id:04d}.png"
            save_dir = os.path.join(output_dir, frame_name)

            vertical_ioct = frame[365:660, 653:1268, :]
            vertical_ioct[8:10, 8:60, :] = vertical_ioct[6:8, 8:60, :] # up
            vertical_ioct[58:60, 8:60, :] = vertical_ioct[56:58, 8:60, :] # down
            vertical_ioct[8:60, 8:10, :] = vertical_ioct[8:60, 6:8, :] # left
            vertical_ioct[8:60, 58:60, :] = vertical_ioct[8:60, 56:58, :] # right
            vertical_ioct[94:199, 605:609, :] = vertical_ioct[94:199, 609:613, :] # right
            vertical_ioct[8:60, 32:35, :] = vertical_ioct[8:60, 35:38, :] # purple
            vertical_ioct[11:17, 31:37, :] = vertical_ioct[5:11, 31:37, :] # purple
            vertical_ioct[151:156, 583:600, :] = vertical_ioct[151:156, 566:583, :] # up
            vertical_ioct[135:144, 591:593, :] = vertical_ioct[135:144, 593:595, :] # up
            vertical_ioct[136:139, 589:591, :] = vertical_ioct[136:139, 587:589, :] # up
            cv2.imwrite(save_dir, vertical_ioct)

            frame_id += 1

