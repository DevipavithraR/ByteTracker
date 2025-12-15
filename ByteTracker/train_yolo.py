from ultralytics import YOLO
from utils.constants import face_model
from utils.util import getdatapath
import os
import utils.util as util
import utils.constants as constants
import utils.models as models

current_script_dir = os.path.dirname(os.path.abspath(__file__))
data_base_dir = util.getdatapath(current_script_dir, '..', models.data_base_dir)
face_model = os.path.join(data_base_dir, models.face_model)


# Load model
model = YOLO(face_model)

# Train on your dataset
model.train(
    data=os.path.join(data_base_dir, "output/data.yaml"),
    epochs=50,
    imgsz=640,
    batch=16
)

