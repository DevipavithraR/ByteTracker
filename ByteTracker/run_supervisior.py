import subprocess
import time
import os
from pathlib import Path

IMAGES_PATH = "ByteTracker_data/output/train/images"
LABELS_PATH = "ByteTracker_data/output/train/labels"
MODEL_WEIGHTS = "ByteTracker/runs/detect/train/weights/best.pt"
CONSTANTS_FILE = "utils/constants.py"

IMAGE_THRESHOLD = 100   # run training after every 100 new images

def count_files(folder):
    return len([f for f in os.listdir(folder) if f.endswith(".jpg")])

def update_model_path():
    lines = []
    with open(CONSTANTS_FILE, "r") as f:
        for line in f:
            if "face_model =" in line:
                line = f'face_model = "{MODEL_WEIGHTS}"\n'
            lines.append(line)
    with open(CONSTANTS_FILE, "w") as f:
        f.writelines(lines)
    print("✔ Updated constants.py with new best.pt")

def start_attendance():
    return subprocess.Popen(["python3", "attendance_byte.py"])

def run_training_pipeline():
    print("\n🔄 Running generate_yaml.py...")
    subprocess.run(["python3", "generate_yaml.py"])

    print("\n🔄 Running train_yolo.py...")
    subprocess.run(["python3", "train_yolo.py"])

    update_model_path()
    print("\n✔ Training complete. Model updated.\n")


if __name__ == "__main__":

    # Start attendance system continuously
    attendance_process = start_attendance()
    print("🟢 Attendance running 24×7...")

    prev_count = count_files(IMAGES_PATH)

    while True:
        time.sleep(30)  # check every 30 seconds

        current_count = count_files(IMAGES_PATH)

        new_images = current_count - prev_count

        if new_images >= IMAGE_THRESHOLD:
            print(f"\n📸 {new_images} NEW IMAGES detected → Training Started\n")

            # Run training pipeline
            run_training_pipeline()

            # Reset counter
            prev_count = count_files(IMAGES_PATH)

        else:
            print(f"⏳ Only {new_images} new images. Waiting ...")
