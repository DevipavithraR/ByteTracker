from utils.headPose_estimator import HeadPoseEstimator
from utils.headpose_confidence import generate_pose_conf
import utils.constants as constants
import utils.util as util

class PoseService:
    def __init__(self, model, device='CPU'):
        self.estimator = HeadPoseEstimator(device)
        self.estimator.load(model)

    def estimate(self, face):
        inp = util.preprocess_image(face, self.estimator.input_shape)
        yaw, pitch, roll = self.estimator.infer(inp)
        conf = generate_pose_conf([abs(yaw), abs(pitch), abs(roll)], constants.POSE_THRESHOLDS)
        head_pos = util.classify_head_position(pitch)
        return yaw, pitch, roll, conf, head_pos
