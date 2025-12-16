import cv2
import sys
import platform

class CameraService:
    _instance = None
    def __init__(self, device='0'):
        if platform.system() == "Darwin":
            self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        else:
            print('linux camera')
            self.cap = cv2.VideoCapture(int(device) if device == "0" else device)
            print('linux camera 2')
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {device}")
        
    @classmethod
    def get_instance(cls, device=0):
        print('singleton')
        if cls._instance is None:
            cls._instance = CameraService(device)
        return cls._instance
    
    def read(self):
        return self.cap.read()

    def release(self):
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()


# # camera_service.py
# import cv2

# class CameraService:
#     _instance = None

#     def __init__(self, device=0):
#         self.cap = cv2.VideoCapture(device, cv2.CAP_AVFOUNDATION)
#         if not self.cap.isOpened():
#             raise RuntimeError(f"Cannot open camera {device}")

#     @classmethod
#     def get_instance(cls, device=0):
#         if cls._instance is None:
#             cls._instance = CameraService(device)
#         return cls._instance

#     def read(self):
#         ret, frame = self.cap.read()
#         return ret, frame

#     def release(self):
#         self.cap.release()
