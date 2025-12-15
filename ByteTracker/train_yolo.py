from ultralytics import YOLO
from utils.models import face_model
from utils.util import getdatapath
import os
import utils.util as util
import utils.constants as constants
import utils.models as models


face_model = models.face_model


# Load model
model = YOLO(face_model)

# Train on your dataset
model.train(
    data=constants.output_yaml,
    epochs=50,
    imgsz=640,
    batch=16,
    project=models.trained_model_base_dir,
    name=models.trained_model_dir
)

