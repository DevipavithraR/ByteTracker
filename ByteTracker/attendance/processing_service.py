from attendance.tracking_service import TrackingService
from attendance.recognition_service import RecognitionService
from attendance.pose_service import PoseService
from attendance.attendance_service import AttendanceService


class ProcessingService:
    def __init__(self, tracker:TrackingService, recognizer:RecognitionService, poser:PoseService, attendance:AttendanceService):
        self.tracker = tracker
        self.recognizer = recognizer
        self.poser = poser
        self.attendance = attendance

    def process_frame(self, frame):
        tracks = self.tracker.detect_and_track(frame)

        results = []
        for t in tracks:
            x, y, w, h = map(int, t.tlwh)
            face = frame[y:y+h, x:x+w]
            if face.size == 0:
                continue

            identity = self.recognizer.recognize(face)
            yaw, pitch, roll, conf, head_pos = self.poser.estimate(face)

            self.attendance.update(identity)

            results.append((x, y, w, h, identity, yaw, pitch, roll, conf, head_pos))

        return results
