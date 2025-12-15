from pathlib import Path

POSE_THRESHOLDS = [45.0, 45.0, 45.0]  # yaw, pitch, roll limits
FACE_CONF_THRESHOLD = 0.5
MAX_DISAPPEARED = 20  # frames before an ID is removed
MAX_DISTANCE = 50  # max pixel distance to match same ID
lpfid_map = {}   
LPFID_TTL = 10 * 60 
next_lpfid = 1

data_base_dir = "ByteTracker_data"
model_dir = 'models'
faces_dir = 'faces'
videos_dir = 'videos'
labels_dir = 'labels'
output_dir = 'output'
all_detection_json = "all_detections.json"

paused = False

# face_model = model_dir + "/yolov8n.pt"
video = videos_dir + "/input_video.mp4"

save_once_in_count = 10
save_once_in_step = 1

ATTENDANCE_SCRIPT = "attendance_byte.py"
YAML_SCRIPT = "generate_yaml.py"
TRAIN_SCRIPT = "train_yolo.py"

CONSTANTS_FILE = "utils/models.py"
STOP_FILE = "stop.flag"

IMAGE_DIR = Path("/home/arffy/Documents/DeviPavithra/ByteTrackWorkspace/ByteTracker_data/output/train/images")
MIN_IMAGES = 5
first_run = True