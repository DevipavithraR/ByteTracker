import time
import subprocess
import os
import sys
from pathlib import Path
from utils.constants import STOP_FILE,CONSTANTS_FILE,ATTENDANCE_SCRIPT,YAML_SCRIPT,TRAIN_SCRIPT,IMAGE_DIR,MIN_IMAGES,first_run
from services.pipeline_services import count_images,Keyboard,should_stop,handle_keypress,run_script,update_face_model_in_constants

previous_count = count_images()

def attendance_service():
    with Keyboard() as kb:
        while True:
            if should_stop():
                print("🛑 STOP FILE DETECTED — Shutting down pipeline.")
                break

            print("\n==========================")
            print("⚡ STARTING FULL PIPELINE")
            print("==========================\n")

                    # STEP 1 — Run attendance
            print("🟩 STEP 1: Running attendance_byte.py ...")
                    # p = subprocess.Popen(["python3", ATTENDANCE_SCRIPT])
            p = subprocess.Popen(
                ["python3", ATTENDANCE_SCRIPT],
                stdin=subprocess.DEVNULL
            )


                    # Run attendance for 5 mins (300 sec)
            for i in range(3600):

                        # Stop by stop.flag
                        if should_stop():
                            print("🛑 STOP — Terminating attendance system.")
                            p.terminate()
                            raise SystemExit

                        # Stop by 'q'
                        # if stop_on_keypress():
                        #     print("🛑 'q' pressed — Shutting down.")
                        #     p.terminate()
                        #     raise SystemExit
                        # action = stop_on_keypress(p)
                        

                        # if action == "stop":
                        #     print("🛑 Stopping attendance (cycle only).")
                        #     p.terminate()
                        #     break   # exit attendance loop only

                        # if action == "quit":
                        #     print("🛑 Terminating pipeline completely.")
                        #     p.terminate()

                        #     # DELETE stop.flag AFTER TERMINATION
                        #     if os.path.exists(STOP_FILE):
                        #         os.remove(STOP_FILE)
                        #         print("🧹 stop.flag removed.")

                        #     sys.exit()

                        key = kb.get_key()
                        if key:
                            action = handle_keypress(p, key)

                            if action == "stop":
                                p.terminate()
                                break

                            if action == "quit":
                                p.terminate()
                                if os.path.exists(STOP_FILE):
                                    os.remove(STOP_FILE)
                                    print("🧹 stop.flag removed")
                                sys.exit()

                        time.sleep(0.05)
                        # time.sleep(1)

            p.terminate()

                    # STEP 2 — Check new images
            current_count = count_images()

            if first_run:
                        new_images = current_count
                        first_run = False
            else:
                        new_images = current_count - previous_count

            print(f"\n🟦 New images detected: {new_images}")

            if new_images < MIN_IMAGES:
                        print("⚠️ Not enough images — skipping training.")
                        print("⏳ Waiting 5 mins before next cycle...\n")
                        time.sleep(300)
                        continue

                    # STEP 3 — Generate YAML
            print("\n🟩 STEP 2: Creating data.yaml...")
            run_script(YAML_SCRIPT)

                    # STEP 4 — Train YOLO
            print("\n🟩 STEP 3: Training YOLO model...")
            run_script(TRAIN_SCRIPT)
                    # STEP 5 — Update constants.py
            print("\n🟩 STEP 4: Updating best model in constants.py...")
            update_face_model_in_constants()
            previous_count = current_count

            print("\n🔥 Pipeline cycle complete!")
            print("⏳ Waiting 5 minutes before next cycle...\n")
            time.sleep(300)
        