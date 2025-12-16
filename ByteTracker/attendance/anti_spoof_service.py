import cv2
from keras.models import load_model
import numpy as np
from config import BASE_DIR

class AntiSpoofService:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = f"{BASE_DIR}/models/anti_spoof_model.h5"
        self.model = load_model(model_path)

    def predict(self, face_img):
        # resize to model input
        img = cv2.resize(face_img, (160, 160))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        pred = self.model.predict(img)
        # returns probability of being real
        return pred[0][0] > 0.5  # True = real, False = spoof
