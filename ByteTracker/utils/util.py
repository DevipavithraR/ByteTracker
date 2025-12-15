import cv2
import numpy as np
import time
import os
from datetime import datetime
import csv

def preprocess_image(image, shape):
    _, _, h, w = shape
    resized = cv2.resize(image, (w, h))
    transposed = np.transpose(resized, (2, 0, 1))
    return np.expand_dims(transposed, axis=0).astype(np.float32)

def classify_head_position(pitch, down_threshold=20, up_threshold=-20):
    if pitch >= down_threshold:
        return "Down"
    elif pitch <= up_threshold:
        return "Up"
    else:
        return "Forward"
    
def getdatapath(current_script_dir, backward, data_dir):
    # Navigate to the parent directory (one level up)
    parent_dir = os.path.join(current_script_dir, backward)
    # Resolve the '..' to get the absolute path of the parent directory
    absolute_parent_dir = os.path.abspath(parent_dir)

    print(f"Parent directory: {absolute_parent_dir}")

    # Example: Accessing a 'data' folder in the parent directory
    data_folder_path = os.path.join(absolute_parent_dir, data_dir)
    print(f"Data folder path: {data_folder_path}")

    return data_folder_path

def save_yolo_label(image_path, bbox, img_w, img_h):
    x_min, y_min, x_max, y_max = bbox

    # convert to YOLO format
    cx = (x_min + x_max) / 2.0 / img_w
    cy = (y_min + y_max) / 2.0 / img_h
    w  = (x_max - x_min) / img_w
    h  = (y_max - y_min) / img_h

    label_path = image_path.replace("images", "labels").replace(".jpg", ".txt")
    os.makedirs(os.path.dirname(label_path), exist_ok=True)

    with open(label_path, "w") as f:
        f.write(f"0 {cx} {cy} {w} {h}\n")



def get_attendance_status():
    now = datetime.now().time()

    # ----- Login time ranges -----
    ontime_start = datetime.strptime("09:30", "%H:%M").time()
    ontime_end   = datetime.strptime("10:00", "%H:%M").time()

    grace_start  = datetime.strptime("10:00", "%H:%M").time()
    grace_end    = datetime.strptime("10:10", "%H:%M").time()

    late_start   = datetime.strptime("10:10", "%H:%M").time()
    late_end     = datetime.strptime("10:30", "%H:%M").time()

    verylate_start = datetime.strptime("10:30", "%H:%M").time()
    verylate_end   = datetime.strptime("11:00", "%H:%M").time()

    excessive_start = datetime.strptime("11:00", "%H:%M").time()
    excessive_end   = datetime.strptime("12:00", "%H:%M").time()

    halfday_start = datetime.strptime("12:00", "%H:%M").time()

    # ----- Logout Time -----
    logout_start = datetime.strptime("17:30", "%H:%M").time()
    logout_end   = datetime.strptime("18:00", "%H:%M").time()

    # ----- Conditions -----

    if ontime_start <= now <= ontime_end:
        return "On-time Login"

    if grace_start <= now <= grace_end:
        return "Grace Late"

    if late_start <= now <= late_end:
        return "Late Login"

    if verylate_start <= now <= verylate_end:
        return "Very Late Login"

    if excessive_start <= now <= excessive_end:
        return "Excessive Late Login"

    if now >= halfday_start and now < logout_start:
        return "Half Day Login"

    if logout_start <= now <= logout_end:
        return "Logout"

    return None  # outside all ranges



def write_attendance_csv(emp_name, status):
    date = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")

    filename = "attendance_log.csv"
    file_exists = os.path.isfile(filename)

    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["Date", "Employee", "Time", "Status"])

        writer.writerow([date, emp_name, time_now, status])

    print(f"✔ Attendance recorded: {emp_name} → {status}")

