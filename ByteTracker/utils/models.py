import os
import utils.util as util
from utils import constants


pose_model = os.path.join(constants.data_base_dir, "models/head-pose-estimation-adas-0001.xml")

# trained_model_base_dir = os.path.join(data_base_dir,"models/Trained")
# face_model = os.path.join(trained_model_base_dir, "train5/weights/best.pt")
face_model = os.path.join(constants.data_base_dir, "models/yolov12n-face.pt")
trained_model_base_dir = os.path.join(constants.data_base_dir, "models/trained")
trained_model_dir = 'train'