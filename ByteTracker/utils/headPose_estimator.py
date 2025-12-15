from openvino.runtime import Core

class HeadPoseEstimator:
    def __init__(self, device="CPU"):
        self.core = Core()
        self.device = device

    def load(self, model_path):
        model_ir = self.core.read_model(model=str(model_path))
        self.compiled_model = self.core.compile_model(model_ir, device_name=self.device)
        self.input_layer = self.compiled_model.input(0)

    @property
    def input_shape(self):
        return self.input_layer.shape

    def infer(self, image):
        outputs = self.compiled_model([image])
        yaw = float(outputs[self.compiled_model.output("angle_y_fc")][0][0])
        pitch = float(outputs[self.compiled_model.output("angle_p_fc")][0][0])
        roll = float(outputs[self.compiled_model.output("angle_r_fc")][0][0])
        return yaw, pitch, roll