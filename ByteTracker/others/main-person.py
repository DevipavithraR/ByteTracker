import cv2
import numpy as np
from openvino.runtime import Core

ie = Core()

# Load the person detection model
model_path = "model/person-detection-0201.xml"
model = ie.read_model(model=model_path)
compiled_model = ie.compile_model(model=model, device_name="CPU")

# Get input and output layers
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)
n, c, h, w = input_layer.shape

# video_path = "/home/arffy/Documents/DeviPavithra/ByteTrackWorkspace/video/input_video.mp4"   # 👈 your local video file path
video_path = 0   # 👈 your local video file path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Loop through video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break  # end of video
    
    # --- Preprocess the frame ---
    image_resized = cv2.resize(frame, (w, h))
    image_input = image_resized.transpose((2, 0, 1))  # HWC → CHW
    image_input = np.expand_dims(image_input, axis=0)

    # --- Run inference ---
    results = compiled_model([image_input])[output_layer]

    # --- Postprocess detections ---
    for detection in results[0][0]:
        confidence = detection[2]
        if confidence > 0.5:  # confidence threshold
            xmin = int(detection[3] * frame.shape[1])
            ymin = int(detection[4] * frame.shape[0])
            xmax = int(detection[5] * frame.shape[1])
            ymax = int(detection[6] * frame.shape[0])

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {confidence:.2f}", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Person Detection - Video", frame)

    # Press 'q' to stop early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
