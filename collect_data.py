import csv
import os
import cv2
import mediapipe as mp
import numpy as np
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# CONFIGURATION
# ==========================================
TARGET_SIGN = "b"
MODEL_PATH = "hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print(f"Downloading required model '{MODEL_PATH}' from Google...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Download complete!")

# Ensure dataset directory exists
os.makedirs("dataset", exist_ok=True)
csv_file_path = f"dataset/{TARGET_SIGN}.csv"

# Check if model file exists locally
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Could not find '{MODEL_PATH}' in the current directory. Please download it from Google MediaPipe models and place it here."
    )

# ==========================================
# INITIALIZE MODERN MEDIAPIPE HAND LANDMARKER
# ==========================================
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)

# MediaPipe Hand Connections mapping for drawing skeleton manually via OpenCV
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # Thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # Index
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # Middle
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # Ring
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # Pinky
    (0, 17),  # Palm base
]

# ==========================================
# OPEN WEBCAM & START RECORDING LOOP
# ==========================================
cap = cv2.VideoCapture(0)
print(f"--- Recording data for sign: '{TARGET_SIGN}' ---")
print("Instructions:")
print("  Press 's' to capture and save the current hand frame.")
print("  Press 'q' to quit and save your dataset.")

# Open CSV file to append data points
with open(csv_file_path, mode="a", newline="") as f:
    writer = csv.writer(f)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip horizontally for natural selfie view and convert colors
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=rgb_frame
        )

        # Run modern hand detection
        detection_result = detector.detect(mp_image)

        # Draw hand landmarks manually using OpenCV if detected
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Convert normalized coordinates to pixel values for drawing
                pixel_landmarks = []
                for lm in hand_landmarks:
                    px = int(lm.x * width)
                    py = int(lm.y * height)
                    pixel_landmarks.append((px, py))
                    # Draw circles on joints
                    cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

                # Draw skeleton connections
                for connection in HAND_CONNECTIONS:
                    pt1 = pixel_landmarks[connection[0]]
                    pt2 = pixel_landmarks[connection[1]]
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

        # Display status text on screen
        cv2.putText(
            frame,
            f"Target: {TARGET_SIGN} | Press 's' to Save | 'q' to Quit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Modern Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        # Press 's' to save normalized landmarks
        if key == ord("s"):
            if detection_result.hand_landmarks:
                for hand_landmarks in detection_result.hand_landmarks:
                    landmarks = []
                    for lm in hand_landmarks:
                        landmarks.append([lm.x, lm.y, lm.z])

                    landmarks = np.array(landmarks)

                    # Normalize relative to wrist (landmark 0)
                    wrist = landmarks[0]
                    landmarks = landmarks - wrist

                    # Scale by distance from wrist to middle finger MCP (landmark 9)
                    scale_distance = np.linalg.norm(landmarks[9])
                    if scale_distance > 0:
                        landmarks = landmarks / scale_distance

                    # Flatten to 1D vector of 63 features and save
                    flat_row = landmarks.flatten().tolist()
                    writer.writerow(flat_row)
                    print(
                        f"[SUCCESS] Saved sample vector for '{TARGET_SIGN}'!"
                    )
            else:
                print(
                    "[WARNING] No hand detected in frame. Position your hand clearly before pressing 's'."
                )

        # Press 'q' to exit loop
        elif key == ord("q"):
            print("Exiting and closing application...")
            break

cap.release()
cv2.destroyAllWindows()