from collections import Counter, deque
import os
import cv2
import joblib
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class AslRecognizeGui:
    def __init__(
            model_path: str = "hand_landmarker.task"
        ):  
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Could not find '{model_path}' in the current directory.")
    
        try:
            clf = joblib.load("sign_model.pkl")
        except FileNotFoundError:
            raise FileNotFoundError("Could not find 'sign_model.pkl'. Train your model first!")
    
        # ==========================================
        # INITIALIZE MEDIAPIPE HAND LANDMARKER
        # ==========================================
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        detector = vision.HandLandmarker.create_from_options(options)
    
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (5, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (0, 17),  # Palm base
        ]
    
        # ==========================================
        # TKINTER GUI SETUP
        # ==========================================
        root = tk.Tk()
        root.title("Sign Language Recognition System (Smoothed)")
        root.geometry("900x550")
        root.configure(bg="#222222")
    
        # Left Frame for Webcam Feed
        left_frame = tk.Frame(root, bg="#222222")
        left_frame.pack(side=tk.LEFT, padx=20, pady=20)
    
        webcam_label = tk.Label(left_frame, bg="#000000")
        webcam_label.pack()
    
        # Right Frame for Prediction & Letter Image Panel
        right_frame = tk.Frame(root, bg="#222222")
        right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=20, pady=20)
    
        title_label = tk.Label(
            right_frame, text="Current Prediction", font=("Arial", 16, "bold"), fg="#ffffff", bg="#222222"
        )
        title_label.pack(pady=10)
    
        prediction_text_var = tk.StringVar(value="Waiting for hand...")
        prediction_label = tk.Label(
            right_frame, textvariable=prediction_text_var, font=("Arial", 22, "bold"), fg="#00ffcc", bg="#222222"
        )
        prediction_label.pack(pady=10)
    
        # Panel for [LETTER].png Image display
        image_panel = tk.Label(right_frame, bg="#333333", width=200, height=200)
        image_panel.pack(pady=15)
    
        # Initialize OpenCV Video Capture & Smoothing Variables
        cap = cv2.VideoCapture(0)
        prediction_queue = deque(maxlen=15)  # Holds predictions over the last 15 frames (~0.5 sec)
        current_displayed_sign = None
        cached_img_tk = None
    
        root.protocol("WM_DELETE_WINDOW", on_closing)
    
        # Start the GUI loop
        root.after(15, update_gui)
        root.mainloop()

def update_gui(
        cap: cv2.VideoCapture,
        root: tk.Tk,
        detector
    ):
    global current_displayed_sign, cached_img_tk
    
    success, frame = cap.read()
    if success:
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = detector.detect(mp_image)
        frame_prediction = "Uncertain"

        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Draw skeleton landmarks manually
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

                # Extract & Normalize landmarks
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

                # Predict
                prediction = clf.predict(features)[0]
                probabilities = clf.predict_proba(features)
                confidence = np.max(probabilities)

                if confidence > 0.7:
                    frame_prediction = prediction

        # Add current frame prediction to rolling window queue
        prediction_queue.append(frame_prediction)

        # Majority voting: Find the most common prediction in the last 15 frames
        counts = Counter(prediction_queue)
        most_common_sign, frequency = counts.most_common(1)[0]

        # Require a consensus (e.g., must appear in at least 8 out of the last 15 frames)
        if frequency >= 8 and most_common_sign != "Uncertain":
            smoothed_sign = most_common_sign
            prediction_text_var.set(f"{smoothed_sign} (Stable)")
        else:
            smoothed_sign = "Uncertain"
            prediction_text_var.set("Uncertain / Transitioning")

        # Update Image Panel ONLY if the smoothed sign actually changed (prevents disk lag)
        if smoothed_sign != current_displayed_sign:
            current_displayed_sign = smoothed_sign
            img_path = os.path.join(os.getcwd(), "images", "instructions", f"{smoothed_sign}.jpg")
            
            if smoothed_sign != "Uncertain" and os.path.exists(img_path):
                img = Image.open(img_path)
                img = img.resize((180, 180), Image.Resampling.LANCZOS)
                cached_img_tk = ImageTk.PhotoImage(img)
                image_panel.configure(image=cached_img_tk, text="")
            else:
                cached_img_tk = None
                image_panel.configure(image="", text=f"No Image / Uncertain", fg="white")

        # Convert OpenCV Frame to Tkinter PhotoImage for the webcam panel
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(cv2image)
        img_tk_webcam = ImageTk.PhotoImage(image=img_pil)
        webcam_label.imgtk = img_tk_webcam
        webcam_label.configure(image=img_tk_webcam)

    # Schedule the next frame update (~30 fps -> 15ms)
    root.after(15, update_gui)

def on_closing(
        cap: cv2.VideoCapture,
        root: tk.Tk
    ):
    cap.release()
    root.destroy()

def main():
    AslRecognizeGui()

if __name__ == "__main__":
    main()