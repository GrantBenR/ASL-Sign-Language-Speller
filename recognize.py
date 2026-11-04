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
            self,
            model_path: str = "hand_landmarker.task",
            hand_connections: list[tuple[int, int]] = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                (5, 9), (9, 10), (10, 11), (11, 12),  # Middle
                (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
                (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                (0, 17),  # Palm base
            ]
        ):  
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Could not find '{model_path}' in the current directory.")
    
        try:
            self.clf = joblib.load("sign_model.pkl")
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
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # ==========================================
        # TKINTER GUI SETUP
        # ==========================================
        self.hand_connections = hand_connections
        self.root = tk.Tk()
        self.root.title("Sign Language Recognition System (Smoothed)")
        self.root.geometry("900x620")  # Slightly increased height to comfortably fit the quit button
        self.root.configure(bg="#222222")
    
        # Left Frame for Webcam Feed
        self.left_frame = tk.Frame(self.root, bg="#222222")
        self.left_frame.pack(side=tk.LEFT, padx=20, pady=20)
    
        self.webcam_label = tk.Label(self.left_frame, bg="#000000")
        self.webcam_label.pack()
    
        # Right Frame for Prediction, Letter Image Panel, & Quit Button
        self.right_frame = tk.Frame(self.root, bg="#222222")
        self.right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=20, pady=20)
    
        self.title_label = tk.Label(
            self.right_frame, text="Current Prediction", font=("Arial", 16, "bold"), fg="#ffffff", bg="#222222"
        )
        self.title_label.pack(pady=10)
    
        self.prediction_text_var = tk.StringVar(value="Waiting for hand...")
        self.prediction_label = tk.Label(
            self.right_frame, textvariable=self.prediction_text_var, font=("Arial", 22, "bold"), fg="#00ffcc", bg="#222222"
        )
        self.prediction_label.pack(pady=10)
    
        # Panel for [LETTER].png Image display
        self.image_panel = tk.Label(self.right_frame, bg="#333333", width=200, height=200)
        self.image_panel.pack(pady=15)

        # Quit Button placed below the image pane
        self.quit_button = tk.Button(
            self.right_frame,
            text="Quit",
            font=("Arial", 12, "bold"),
            fg="#ffffff",
            bg="#d9534f",
            activebackground="#c9302c",
            activeforeground="#ffffff",
            bd=0,
            padx=20,
            pady=8,
            command=self._OnClosing
        )
        self.quit_button.pack(pady=10)
    
        # Initialize OpenCV Video Capture & Smoothing Variables
        self.cap = cv2.VideoCapture(0)
        self.prediction_queue = deque(maxlen=15)  # Holds predictions over the last 15 frames (~0.5 sec)
        self.current_displayed_sign = None
        self.cached_img_tk = None
    
        self.root.protocol("WM_DELETE_WINDOW", self._OnClosing)
    
        # Start the GUI loop
        self.root.after(15, self._UpdateGui)
        self.root.mainloop()

    def _UpdateGui(
            self,
        ):        
        success, frame = self.cap.read()
        if success:
            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            detection_result = self.detector.detect(mp_image)
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

                    for connection in self.hand_connections:
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
                    prediction = self.clf.predict(features)[0]
                    probabilities = self.clf.predict_proba(features)
                    confidence = np.max(probabilities)

                    if confidence > 0.7:
                        frame_prediction = prediction

            # Add current frame prediction to rolling window queue
            self.prediction_queue.append(frame_prediction)

            # Majority voting: Find the most common prediction in the last 15 frames
            counts = Counter(self.prediction_queue)
            most_common_sign, frequency = counts.most_common(1)[0]

            # Require a consensus (e.g., must appear in at least 8 out of the last 15 frames)
            if frequency >= 8 and most_common_sign != "Uncertain":
                smoothed_sign = most_common_sign
                self.prediction_text_var.set(f"{smoothed_sign} (Stable)")
            else:
                smoothed_sign = "Uncertain"
                self.prediction_text_var.set("Uncertain / Transitioning")

            # Update Image Panel ONLY if the smoothed sign actually changed (prevents disk lag)
            if smoothed_sign != self.current_displayed_sign:
                self.current_displayed_sign = smoothed_sign
                img_path = os.path.join(os.getcwd(), "images", "instructions", f"{smoothed_sign}.jpg")
                
                if smoothed_sign != "Uncertain" and os.path.exists(img_path):
                    img = Image.open(img_path)
                    img = img.resize((180, 180), Image.Resampling.LANCZOS)
                    self.cached_img_tk = ImageTk.PhotoImage(img)
                    self.image_panel.configure(image=self.cached_img_tk, text="")
                else:
                    self.cached_img_tk = None
                    self.image_panel.configure(image="", text=f"No Image / Uncertain", fg="white")

            # Convert OpenCV Frame to Tkinter PhotoImage for the webcam panel
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(cv2image)
            img_tk_webcam = ImageTk.PhotoImage(image=img_pil)
            self.webcam_label.imgtk = img_tk_webcam
            self.webcam_label.configure(image=img_tk_webcam)

        # Schedule the next frame update (~30 fps -> 15ms)
        self.root.after(15, self._UpdateGui)

    def _OnClosing(
            self,
        ):
        self.cap.release()
        self.root.destroy()

def main():
    AslRecognizeGui()

if __name__ == "__main__":
    main()