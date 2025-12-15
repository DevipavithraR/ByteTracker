import cv2
import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from deepface import DeepFace
import sys
import os
from utils.attendancecsv import mark_attendance
sys.path.append(os.path.abspath("ByteTrack"))
from openvino.runtime import Core
# from src.centroid_tracker import CentroidTracker
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker as ByteTracker
from utils.headpose_confidence import generate_pose_conf
from utils.headPose_estimator import HeadPoseEstimator
import utils.util as util
import utils.constants as constants

if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'bool'):
    np.bool = bool
# -----------------------
# ID Management
# -----------------------
lpfid_map = {}
next_lpfid = 1
MIN_FACE_SIZE = 30 
lpfid_identity_map = {}  # Maps persistent LPF IDs to recognized person names


def cleanup_old_ids():
    """Remove expired LPF IDs that haven't been seen within TTL."""
    now = time.time()
    expired = [
        oid for oid, (_, last_seen) in lpfid_map.items()
        if now - last_seen > constants.LPFID_TTL
    ]
    for oid in expired:
        del lpfid_map[oid]


def get_lpfid(object_id):
    """Assign or refresh a logical persistent face ID."""
    global next_lpfid
    now = time.time()

    if object_id in lpfid_map:
        lpfid, _ = lpfid_map[object_id]
        lpfid_map[object_id] = (lpfid, now)
        return lpfid

    lpfid = f"LPF{next_lpfid:03d}"
    next_lpfid += 1
    lpfid_map[object_id] = (lpfid, now)
    return lpfid


# -----------------------
# Main Pipeline
# -----------------------
def run(pose_model, video_path, out_dir, device="CPU"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    pose_estimator = HeadPoseEstimator(device)
    pose_estimator.load(pose_model)

    ie = Core()
    model_path = "model/person-detection-0201.xml"
    compiled_model = ie.compile_model(model=model_path, device_name=device)
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    
    # tracker = CentroidTracker()
    tracker = ByteTracker(args)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    # if video_path == "0":
    #     cap = cv2.VideoCapture(0)  # Open webcam
    # else:
    #     cap = cv2.VideoCapture(video_path)
    #     if not cap.isOpened():
    #         raise FileNotFoundError(f"Could not open video: {video_path}")
    

    frame_idx = 0
    all_frames_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        frame_data = {"frame": frame_idx, "detections": []}
        # Resize frame to model input shape
        n, c, h, w = input_layer.shape
        resized = cv2.resize(frame, (w, h))
        input_image = resized.transpose((2, 0, 1))[np.newaxis, :]  # NHWC -> NCHW

        # Run inference
        output = compiled_model([input_image])[output_layer]

        rects_with_conf = []
        ih, iw = frame.shape[:2]

        for detection in output[0][0]:
            conf = float(detection[2])
            if conf < 0.6:
                continue

            x_min = int(detection[3] * iw)
            y_min = int(detection[4] * ih)
            x_max = int(detection[5] * iw)
            y_max = int(detection[6] * ih)

            w_box = x_max - x_min
            h_box = y_max - y_min

            if w_box < MIN_FACE_SIZE or h_box < MIN_FACE_SIZE:
                continue

            rects_with_conf.append(((x_min, y_min, x_max, y_max), conf))

       
        face_info = [(conf, rect) for rect, conf in rects_with_conf]

        # Convert your detection list into numpy format expected by BYTETracker
        if len(rects_with_conf) > 0:
            output_results = np.array([[x1, y1, x2, y2, conf] for ((x1, y1, x2, y2), conf) in rects_with_conf])
        else:
            output_results = np.empty((0, 5))

        # Image info and input size
        img_info = frame.shape[:2]   # (height, width)
        img_size = (frame.shape[0], frame.shape[1])  # same as img_info or (h, w)

        # Update tracker
        objects = tracker.update(output_results, img_info, img_size)

        cleanup_old_ids()
       

        for track in objects:
            tlwh = track.tlwh
            track_id = track.track_id
            x_min, y_min, w, h = map(int, tlwh)
            x_max, y_max = x_min + w, y_min + h


        for track in objects:
            tlwh = track.tlwh  # (top-left x, y, width, height)
            track_id = track.track_id  # unique ID assigned by ByteTrack
            lpfid = get_lpfid(track_id)

            x_min, y_min, w, h = map(int, tlwh)
            x_max, y_max = x_min + w, y_min + h

            face_roi = frame[y_min:y_max, x_min:x_max]
            if face_roi.size == 0:
                continue

           
            # -------------------------
            # 🔹 1. Identify or reuse stored identity
            # -------------------------
            if lpfid in lpfid_identity_map:
                # Reuse previously recognized name
                identity = lpfid_identity_map[lpfid]
            else:
                # Run DeepFace recognition once for new faces
                identity = "Unknown"
                try:
                    result = DeepFace.find(img_path=face_roi, db_path="faces", enforce_detection=False)
                    if len(result[0]) > 0:
                        identity = os.path.basename(os.path.dirname(result[0].iloc[0]["identity"]))
                        lpfid_identity_map[lpfid] = identity  # Save the name for future use
                except Exception as e:
                    print("Recognition error:", e)


            # -------------------------
            # 🔹 2. Head pose estimation
            # -------------------------
            pose_input = util.preprocess_image(face_roi, pose_estimator.input_shape)
            yaw, pitch, roll = pose_estimator.infer(pose_input)
            pose_conf = generate_pose_conf([abs(yaw), abs(pitch), abs(roll)],
                                        constants.POSE_THRESHOLDS)
            head_pos = util.classify_head_position(pitch)

            # -------------------------
            # 🔹 3. Emotion detection (optional)
            # -------------------------
            analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            dominant_emotion = analysis[0]['dominant_emotion']

            # -------------------------
            # 🔹 4. Record data + mark attendance
            # -------------------------
            if identity != "Unknown":
                mark_attendance(identity)   # call function defined globally (see below)

            frame_data["detections"].append({
                "lpfid": lpfid,
                "bbox": [x_min, y_min, x_max, y_max],
                "yaw": round(yaw, 2),
                "pitch": round(pitch, 2),
                "roll": round(roll, 2),
                "pose_conf": round(pose_conf, 4),
                "head_pos": head_pos,
                "emotion": dominant_emotion,
                "identity": identity,
            })


            if dominant_emotion:
                frame_data["detections"].append({
                    "lpfid": lpfid,
                    "bbox": [x_min, y_min, x_max, y_max],
                    "yaw": round(yaw, 2),
                    "pitch": round(pitch, 2),
                    "roll": round(roll, 2),
                    "pose_conf": round(pose_conf, 4),
                    "head_pos": head_pos,
                    "emotion": dominant_emotion,
                })

            # Draw overlay (with ID + emotion)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            display_name = identity if identity != "Unknown" else f"ID {track_id}"
            cv2.putText(frame, f"{display_name} | {dominant_emotion}",
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Save frame and JSON
        cv2.imwrite(str(out_dir / f"frame_{frame_idx:06d}.jpg"), frame)
        if frame_data["detections"]:
            all_frames_data.append(frame_data)

        cv2.imshow("Head-pose demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    with (out_dir / "all_detections.json").open("w", encoding="utf-8") as jf:
        json.dump(all_frames_data, jf, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose", default="model/head-pose-estimation-adas-0001.xml")
    parser.add_argument("--video", default="/home/arffy/Documents/DeviPavithra/ByteTrackWorkspace/video/input_video.mp4")  # Example video path for local
    # parser.add_argument("--video", type=str, default="0", help="Use '0' for webcam or path to video file") # laptop webcam support

    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--device", default="CPU")

    parser.add_argument("--track_thresh", type=float, default=0.5, help="Tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="Frame buffer for lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="Matching threshold for tracking")
    parser.add_argument("--mot20", dest="mot20", action="store_true", help="If using MOT20 dataset")

    args = parser.parse_args()

    for p in [ args.pose]:
        if not Path(p).exists():
            sys.exit(f"[ERROR] File not found: {p}")
            
        # if not Path(args.pose).exists():
        # sys.exit(f"[ERROR] File not found: {args.pose}")

    run(args.pose, args.video, args.out_dir, device=args.device) 

