from pathlib import Path
import utils.util as util
import os


POSE_THRESHOLDS = [45.0, 45.0, 45.0]  # yaw, pitch, roll limits
FACE_CONF_THRESHOLD = 0.5
MAX_DISAPPEARED = 20  # frames before an ID is removed
MAX_DISTANCE = 50  # max pixel distance to match same ID
lpfid_map = {}   
LPFID_TTL = 10 * 60 
next_lpfid = 1

all_detection_json = "all_detections.json"
current_script_dir = os.path.dirname(os.path.abspath(__file__))
prj_dir = util.getdatapath(current_script_dir, '..')
data_base_dir = util.getdatapath(current_script_dir, '../..', "ByteTracker_data")
 
faces_dir = os.path.join(data_base_dir, 'faces')
videos_dir = os.path.join(data_base_dir, 'videos')
labels_dir = os.path.join(data_base_dir, 'labels')
output_dir = os.path.join(data_base_dir, 'output')
 
train_images_dir = os.path.join(output_dir, 'train/images')
output_yaml = os.path.join(output_dir, 'data.yaml')
 
# face_model = model_dir + "/yolov8n.pt"
video = videos_dir + "/input_video.mp4"
 
paused = False

# face_model = model_dir + "/yolov8n.pt"
video = videos_dir + "/input_video.mp4"

save_once_in_count = 10
save_once_in_step = 1

ATTENDANCE_SCRIPT = "attendance_byte_1.py"
YAML_SCRIPT = "generate_yaml.py"
TRAIN_SCRIPT = "train_yolo.py"

CONSTANTS_FILE = "utils/models.py"
STOP_FILE = "stop.flag"

IMAGE_DIR = Path(train_images_dir)
MIN_IMAGES = 5
first_run = True
print("constants last:")