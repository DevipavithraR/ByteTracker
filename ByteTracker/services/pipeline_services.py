from pathlib import Path
import os
from utils.constants import STOP_FILE,CONSTANTS_FILE,ATTENDANCE_SCRIPT,YAML_SCRIPT,TRAIN_SCRIPT,IMAGE_DIR,MIN_IMAGES,first_run,paused
import signal
import termios
import tty
import select   
import sys
import subprocess
import utils.models as models



def should_stop():
    return os.path.exists(STOP_FILE)

class Keyboard:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def get_key(self):
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

def handle_keypress(p, key):
    global paused

    if key == "p":
        if not paused:
            print("⏸ Pausing attendance process")
            p.send_signal(signal.SIGSTOP)
            paused = True
        else:
            print("▶ Resuming attendance process")
            p.send_signal(signal.SIGCONT)
            paused = False

    elif key == "s":
        print("🛑 stop.flag CREATED")
        Path(STOP_FILE).touch()
        return "stop"

    elif key == "q":
        if os.path.exists(STOP_FILE):
            print("❌ stop.flag FOUND — terminating pipeline")
            return "quit"
        else:
            print("⚠ stop.flag NOT found — press 's' first")

    return None

def run_script(script):
    print(f"\n=== Running {script} ===\n")
    subprocess.run(["python3", script])

def update_face_model_in_constants():
    output_dir = models.trained_model_base_dir

    all_train_dirs = sorted(
        [d for d in output_dir.glob("train*") if d.is_dir()],
        key=lambda x: x.stat().st_mtime
    )

    if not all_train_dirs:
        print("No train folders found, skipping update!")
        return

    latest_pt = all_train_dirs[-1] / "weights" / "best.pt"

    if latest_pt.exists():
        new_line = f'face_model = "{latest_pt.absolute()}"\n'

        with open(CONSTANTS_FILE, "r") as f:
            lines = f.readlines()

        with open(CONSTANTS_FILE, "w") as f:
            for line in lines:
                if line.strip().startswith("face_model"):
                    f.write(new_line)
                else:
                    f.write(line)

        print("✔ constants.py updated with latest model.")
    else:
        print("⚠ best.pt not found — skipping!")


def count_images():
    return sum(1 for p in IMAGE_DIR.rglob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"])
