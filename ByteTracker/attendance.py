import cv2
import os
import argparse
from pathlib import Path
from attendance.camera_service import CameraService
from attendance.processing_service import ProcessingService
from attendance.tracking_service import TrackingService
from attendance.recognition_service import RecognitionService
from attendance.pose_service import PoseService
from attendance.attendance_service import AttendanceService
from utils import constants
from utils import models
import numpy as np

if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'bool'):
    np.bool = bool

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--track_thresh", type=float, default=0.45, help="Tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=60, help="Frame buffer for lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="Matching threshold for tracking")
    parser.add_argument("--mot20", dest="mot20", action="store_true", help="If using MOT20 dataset")

    args = parser.parse_args()
    cam = CameraService.get_instance(0)
    print('camera started')
    
    pose_model = models.pose_model
    face_model = models.face_model
    faces_dir = constants.faces_dir

    output_dir = constants.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    processor = ProcessingService(
        TrackingService(face_model, args),
        RecognitionService(faces_dir),
        PoseService(pose_model),
        AttendanceService()
    )

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        results = processor.process_frame(frame)

        for x, y, w, h, identity, *_ in results:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, identity, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()

if __name__ == "__main__":
    main()