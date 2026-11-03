import os
import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def main():

    MODEL_PATH = "hand_landmarker.task"

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Could not find '{MODEL_PATH}' in the current directory. Please make sure it's placed here."
        )

    # Load your trained machine learning model classifier
    try:
        clf = joblib.load("sign_model.pkl")
    except FileNotFoundError:
        raise FileNotFoundError(
            "Could not find 'sign_model.pkl'. Make sure you run your training script first!"
        )

    # ==========================================
    # INITIALIZE MODERN MEDIAPIPE HAND LANDMARKER
    # ==========================================
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # MediaPipe Hand Connections mapping for drawing skeleton via OpenCV
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
    # WEBCAM & REAL-TIME INFERENCE LOOP
    # ==========================================
    cap = cv2.VideoCapture(0)
    print("Starting real-time sign recognition... Press 'q' to exit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip horizontally for selfie view and get dimensions
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Run modern hand detection
        detection_result = detector.detect(mp_image)

        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # 1. Draw skeleton landmarks and lines manually using OpenCV
                pixel_landmarks = []
                for lm in hand_landmarks:
                    px = int(lm.x * width)
                    py = int(lm.y * height)
                    pixel_landmarks.append((px, py))
                    cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

                for connection in HAND_CONNECTIONS:
                    pt1 = pixel_landmarks[connection[0]]
                    pt2 = pixel_landmarks[connection[1]]
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

                # 2. Extract, normalize, and flatten landmarks identical to training pipeline
                landmarks = []
                for lm in hand_landmarks:
                    landmarks.append([lm.x, lm.y, lm.z])

                landmarks = np.array(landmarks)
                wrist = landmarks[0]
                landmarks = landmarks - wrist
                scale_distance = np.linalg.norm(landmarks[9])

                if scale_distance > 0:
                    landmarks = landmarks / scale_distance

                features = landmarks.flatten().reshape(1, -1)

                # 3. Predict sign using the classifier
                prediction = clf.predict(features)[0]
                probabilities = clf.predict_proba(features)
                confidence = np.max(probabilities)

                # 4. Display prediction on screen if confidence is high enough
                if confidence > 0.7:
                    text = f"{prediction} ({confidence*100:.1f}%)"
                else:
                    text = "Uncertain"

                cv2.putText(
                    frame,
                    f"{text} | 'q' to Quit",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        cv2.imshow("Sign Recognition", frame)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()