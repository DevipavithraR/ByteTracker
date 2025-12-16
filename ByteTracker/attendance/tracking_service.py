import sys
import os

# Add ByteTrack folder to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)
BYTE_TRACK_PATH = os.path.join(ROOT_DIR, "ByteTrack")
sys.path.append(BYTE_TRACK_PATH)


import numpy as np
from ultralytics import YOLO
sys.path.append(os.path.abspath("ByteTrack"))
print(os.path.abspath("ByteTrack"))

from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

class TrackingService:
    def __init__(self, face_model, args):
        self.detector = YOLO(face_model)
        self.tracker = BYTETracker(args)

    def detect_and_track(self, frame):
        rects = []
        results = self.detector(frame)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                rects.append([x1, y1, x2, y2, conf])

        dets = np.array(rects) if rects else np.empty((0, 5))
        tracks = self.tracker.update(dets, frame.shape[:2], frame.shape[:2])
        return tracks
