from ultralytics import YOLO
import sys, os
sys.path.append(os.path.abspath("./ByteTrack"))

from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from ByteTrack.yolox.tracking_utils.timer import Timer
import numpy as np
import cv2
from utils.constants import video

# Load YOLOv8 model
model = YOLO("yolov8x.pt")

# ByteTrack setup

class TrackerArgs:
    track_thresh = 0.5
    track_buffer = 30
    match_thresh = 0.8
    aspect_ratio_thresh = 1.6
    min_box_area = 10
    mot20 = False

args = TrackerArgs()
tracker = BYTETracker(args)
timer = Timer()

cap = cv2.VideoCapture(video)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    print("Running YOLO inference...")
    results = model(frame, verbose=False)
    print("YOLO inference completed")

    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        score = float(box.conf[0])
        if score > 0.5:
            detections.append([x1, y1, x2, y2, score])

    if len(detections) == 0:
        cv2.imshow("ByteTrack Tracking", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        continue

    dets = np.array(detections)
    online_targets = tracker.update(dets, [frame.shape[0], frame.shape[1]], [frame.shape[0], frame.shape[1]])

    for t in online_targets:
        tlwh = t.tlwh
        tid = t.track_id
        x1, y1, w, h = tlwh
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {tid}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("ByteTrack Tracking", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
