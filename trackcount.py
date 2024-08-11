from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import time
from sort import *
import argparse


# Inisialisasi argumen
parser = argparse.ArgumentParser(description='Script description')
parser.add_argument('--video_path', type=str, help='Path to input video')
parser.add_argument('--output_video', type=str, help='Path to output video')
parser.add_argument('--model_path', type=str, help='Path to model')
parser.add_argument('--line_coords', type=int, nargs=4, help='Coordinates of the line')
parser.add_argument('--label_count', type=str, help='Label for counted objects')
parser.add_argument('--confidence_threshold', type=float, help='Confidence threshold')
parser.add_argument('--bagi_slice', type=int, help='Coordinates of the line')
# Baca argumen dari baris perintah
args = parser.parse_args()

# Gunakan argumen dalam skrip
video_path = args.video_path
output_video = args.output_video
model_path = args.model_path
line_coords = args.line_coords
label_count = args.label_count
confidence_threshold = args.confidence_threshold
bagi_slice = args.bagi_slice
# Inisialisasi video capture dan video writer
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

# Inisialisasi tracker
tracker = Sort(max_age=60, min_hits=2, iou_threshold=0.3)
model = YOLO(model_path)  # Pastikan ini adalah instance model YOLO

# Inisialisasi counter dan garis penghitung
total_count = 0

# Inisialisasi daftar untuk melacak objek yang telah dihitung
counted_ids = set()

def slice_image(image, slice_height, slice_width, overlap_height_ratio, overlap_width_ratio):
    image_np = np.array(image)
    slices = []
    h, w, _ = image_np.shape
    step_h = int(slice_height * (1 - overlap_height_ratio))
    step_w = int(slice_width * (1 - overlap_width_ratio))
    print(f'Mulai slicing image: {h}x{w}')
    for y in range(0, h, step_h):
        for x in range(0, w, step_w):
            y1 = y
            y2 = min(y + slice_height, h)
            x1 = x
            x2 = min(x + slice_width, w)
            print(f'Slice: x1={x1}, y1={y1}, x2={x2}, y2={y2}')
            slices.append({
                'image': image_np[y1:y2, x1:x2],
                'starting_pixel': (x1, y1)
            })
    print(f'Proses slicing selesai, total slices: {len(slices)}')       
    return slices

def non_max_suppression_fast(boxes, overlapThresh):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    score = boxes[:, 4]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(score)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick]
    
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.perf_counter()

    image = Image.fromarray(frame)
    height, width = image.size

    slice_height = int(height / bagi_slice)
    slice_width = int(width / bagi_slice)
    
    print('Mulai proses slicing frame')
    slice_image_result = slice_image(
        image=image,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )
    print('proses slicing frame selesai')
    bboxes = []

    for i, image_slice in enumerate(slice_image_result):
        window = image_slice['image']
        start_x, start_y = image_slice['starting_pixel']
        
        print(f'Deteksi objek pada slice {i + 1}/{len(slice_image_result)}')
        # Deteksi objek pada slice
        results = model(window, conf=confidence_threshold, verbose=False, iou=0.5)

        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            xyxy = boxes.xyxy.cpu().numpy()

            if xyxy.size == 0:
                continue

            conf = boxes.conf.cpu().numpy()
            class_id = boxes.cls.cpu().numpy()

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]

                # Menyesuaikan koordinat bounding box ke posisi asli pada frame
                x1 += start_x
                y1 += start_y
                x2 += start_x
                y2 += start_y

                bboxes.append([int(x1), int(y1), int(x2), int(y2), conf[i]])

    # Non-Maximum Suppression (NMS)
    if len(bboxes) > 0:
        bboxes = non_max_suppression_fast(np.array(bboxes), overlapThresh=0.3)

    print('Mulai proses kembali ke frame asli')
    # Update tracker
    tracked_objects = tracker.update(bboxes)

    # Gambar bounding boxes dan track IDs
    for obj in tracked_objects:
        if not np.isnan(obj).any():
            x1, y1, x2, y2, track_id = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])

           

            # Hitung pusat objek
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            print('Mulai proses counting')
            # Menghitung objek yang melintasi garis dan berada dalam batas yang ditentukan
            if line_coords[0] < cx < line_coords[2] and line_coords[1] < cy < line_coords[3]:
                if track_id not in counted_ids:  # Periksa apakah objek sudah dihitung
                    total_count += 1  # Tambahkan count jika objek belum dihitung
                    counted_ids.add(track_id)  # Tambahkan track_id ke counted_ids
    print('Proses kembali ke frame asli selesai')               
    # Gambar total count dan FPS
    cv2.putText(frame, f'{label_count}: {total_count}', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    fps = 1 / total_time
    # Gambar garis penghitung
    cv2.line(frame, (line_coords[0], line_coords[1]), (line_coords[2], line_coords[3]), color=(0, 0, 255), thickness=2)

    # Tulis frame ke file output
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f'{label_count}: {total_count}')
