from datetime import datetime
from utils.util import write_attendance_csv, get_attendance_status

FACE_VERIFY_TIME = 60

class AttendanceService:
    def __init__(self):
        self.state = {}

    def update(self, identity):
        if identity == "Unknown":
            return

        now = datetime.now()
        st = self.state.setdefault(identity, {
            "first_seen": now,
            "verified": False,
            "logged_in": False,
            "logged_out": False
        })

        if not st["verified"] and (now - st["first_seen"]).total_seconds() >= FACE_VERIFY_TIME:
            st["verified"] = True

        if st["verified"] and not st["logged_in"]:
            status = get_attendance_status()
            if status != "Logout":
                write_attendance_csv(identity, status)
                st["logged_in"] = True

        if st["verified"] and not st["logged_out"]:
            if get_attendance_status() == "Logout":
                write_attendance_csv(identity, "Logout")
                st["logged_out"] = True
