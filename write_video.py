import cv2
import os

# === CONFIGURATION ===
image_folder = "./plot/OD-2024-11-26_151419_test_1/1"
output_video = "./plot/OD-2024-11-26_151419_test_1/1_video.mp4"
fps = 30  # frames per second

# === GATHER IMAGES ===
images = [img for img in os.listdir(image_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
images.sort()  # ensure frame order

if not images:
    raise ValueError(f"No images found in {image_folder}")

# === READ FIRST IMAGE TO GET SIZE ===
first_frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = first_frame.shape

# === INITIALIZE VIDEO WRITER ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec for .mp4
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# === WRITE FRAMES ===
for img_name in images:
    img_path = os.path.join(image_folder, img_name)
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"⚠️ Skipping unreadable image: {img_name}")
        continue
    video.write(frame)

video.release()
print(f"✅ Video saved to: {output_video}")
