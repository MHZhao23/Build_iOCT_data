import cv2
import os


def fill_in_box_horizontal(frame):
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

    return horizontal_ioct


def fill_in_box_vertical(frame):
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

    return vertical_ioct


if __name__ == "__main__":
    video_root = "D:/Code/work/poct2ioct/FAMNet/FAMNet_oct/data/OCT/video"
    output_root = "./frames"
    video_names = [f for f in os.listdir(video_root) 
                   if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
                   ]
    frames_per_folder = 100

    # === SETUP ===
    for video_name in video_names:
        # video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_path = os.path.join(video_root, video_name)
        output_path = os.path.join(output_root, os.path.splitext(video_name)[0])
        os.makedirs(output_path, exist_ok=True)

        # === EXTRACT FRAMES ===
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        subfolder_index = 1
        frame_index_in_folder = 0

        # Create first subfolder
        subfolder_path = os.path.join(output_path, str(subfolder_index))
        os.makedirs(subfolder_path, exist_ok=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # Crop the frame
            frame_cropped = fill_in_box_horizontal(frame)

            # Save frame
            frame_filename = f"{frame_count+1:06d}.jpg"
            save_path = os.path.join(subfolder_path, frame_filename)
            cv2.imwrite(save_path, frame_cropped)

            # Update counters
            frame_count += 1
            frame_index_in_folder += 1

            # Create new subfolder after 100 frames
            if frame_index_in_folder >= frames_per_folder:
                subfolder_index += 1
                subfolder_path = os.path.join(output_path, str(subfolder_index))
                os.makedirs(subfolder_path, exist_ok=True)
                frame_index_in_folder = 0

        cap.release()
        print(f"✅ Extracted {frame_count} frames into '{output_path}' with subfolders of {frames_per_folder} frames each.")
        break
