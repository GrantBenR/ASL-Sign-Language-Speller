from collections import Counter, deque
import os
import threading
import cv2
import joblib
import numpy
import tkinter
import pygame
import pyttsx3
import speech_recognition
import sounddevice
from PIL import Image, ImageTk
import mediapipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

SUCCESS = "assets/sounds/success_chime.mp3"
BEEP = "assets/sounds/beep.mp3"
BAD_FORM_FEEDBACK_DICT = {
    "A": {
        "E": "Move your thumb next to your curled fingers not under.",
        "M": "Move your thumb next to your curled fingers not under.",
        "N": "Move your thumb next to your curled fingers not under the middle and index.",
        "S": "Move your thumb next to your curled fingers not in front of them.",
        "T": "Move your thumb next to your curled fingers not under the index.",
    },
    "C": {
        "O": "Spread the tips of your fingers out more",
    },
    "E": {
        "A": "Move your thumb under your curled fingers not next to it.",
        "M": "Curl your fingers more.",
        "N": "Move your thumb under all your curled fingers not under the middle and index.",
        "S": "Move your thumb under your curled fingers not in front of them.",
        "T": "Move your thumb under all your curled fingers not under the index.",
    },
    "G": {
        "H": "Curl middle finger and put your thumb to your index finger.",
    },
    "H": {
        "G": "Extend middle and index fingers.",
    },
    "I": {
        "J": "Keep hand stationary.",
    },
    "J": {
        "I": "Do a sweeping j shaped motion.",
    },
    "K": {
        "R": "Place your thumb on your middle finger and uncross your middle and index fingers."
    },
    "M": {
        "A": "Move your thumb under your flat fingers not next to it.",
        "E": "Curl your fingers less.",
        "N": "Move your thumb under all your curled fingers not under the middle and index.",
        "S": "Move your thumb under your curled fingers not in front of them.",
        "T": "Move your thumb under all your curled fingers not under the index."
    },
    "N": {
        "A": "Move your thumb under your index and middle fingers not next to them.",
        "M": "Move your thumb under your index and middle fingers not the others.",
        "E": "Move your thumb under your index and middle fingers not the others.",
        "S": "Move your thumb under your index and middle fingers not in front of them.",
        "T": "Move your thumb under your index and middle fingers not just the index.",
    },
    "O": {
        "C": "Bring your fingertips closer together.",
    },
    "R": {
        "K": "Cross your fingers and place your thumb on your ring finger.",
    },
    "S": {
        "A": "Move your thumb in front of your curled fingers not next to them.",
        "E": "Move your thumb in front of your curled fingers not under them.",
        "N": "Move your thumb in front of all your curled fingers not under the middle and index.",
        "M": "Move your thumb in front of your curled fingers not in under them.",
        "T": "Move your thumb in front of all your curled fingers not under the index.",
    },
    "T": {
        "A": "Move your thumb under the index not next to them.",
        "E": "Move your thumb under the just the index and curl the rest.",
        "N": "Move your thumb under just the index not the middle.",
        "M": "Move your thumb under the index and curl the rest.",
        "S": "Move your thumb under the index not in front.",
    },
}
class AslRecognizeGui:
    def __init__(
            self,
            handlandmark_path: str = "models/handlandmarker/best.task",
            sign_model_path: str = "models/sign/best.pkl",
            hand_connections: list[tuple[int, int]] = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                (5, 9), (9, 10), (10, 11), (11, 12),  # Middle
                (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
                (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                (0, 17),  # Palm base
            ]
        ) -> None:  
        # 
        # Load the model trained on hand motions
        # 
        if not os.path.exists(handlandmark_path):
            raise FileNotFoundError(f"Could not find '{handlandmark_path}' in the current directory.")
    
        try:
            self.clf = joblib.load(sign_model_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find '{sign_model_path}'. Train your model first!")
    
        base_options = python.BaseOptions(model_asset_path=handlandmark_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Game and correction state variables
        pygame.mixer.init()
        self.success_sound = pygame.mixer.Sound(SUCCESS)
        self.beep_sound = pygame.mixer.Sound(BEEP)
        self.target_word = ""
        self.current_index = 0
        self.is_active_practice = False
        self.match_frame_count = 0  
        self.wrong_frame_count = 0
        self.last_spoken_wrong_sign = None
        self.trajectory_history = deque(maxlen=40)  

        # Tkinter GUI Setup
        self.hand_connections = hand_connections
        self.root = tkinter.Tk()
        self.root.title("HCI: ASL Spelling Practice Tool")
        self.root.geometry("1050x850")  # Expanded height to fit the transcript box cleanly
        self.root.configure(bg="#222222")
    
        # Left Frame for Webcam Feed & Transcript Log
        self.left_frame = tkinter.Frame(self.root, bg="#222222")
        self.left_frame.pack(side=tkinter.LEFT, padx=20, pady=20)
    
        self.webcam_label = tkinter.Label(self.left_frame, bg="#000000")
        self.webcam_label.pack(pady=(0, 10))

        # --- Scrollable TTS Transcript Section (Below Webcam) ---
        self.transcript_label = tkinter.Label(
            self.left_frame, text="Audio / TTS Log:", font=("Arial", 10, "bold"), fg="#ffffff", bg="#222222"
        )
        self.transcript_label.pack(anchor=tkinter.W, pady=(0, 2))

        self.transcript_frame = tkinter.Frame(self.left_frame, bg="#222222")
        self.transcript_frame.pack(fill=tkinter.X)

        self.transcript_scroll = tkinter.Scrollbar(self.transcript_frame)
        self.transcript_scroll.pack(side=tkinter.RIGHT, fill=tkinter.Y)

        self.transcript_box = tkinter.Text(
            self.transcript_frame,
            height=6,
            width=50,
            font=("Consolas", 10),
            bg="#111111",
            fg="#00ffcc",
            insertbackground="white",
            yscrollcommand=self.transcript_scroll.set,
            state=tkinter.DISABLED
        )
        self.transcript_box.pack(side=tkinter.LEFT, fill=tkinter.X, expand=True)
        self.transcript_scroll.config(command=self.transcript_box.yview)
    
        # Right Frame for Inputs, Target Display, Prediction Panels, & Quit
        self.right_frame = tkinter.Frame(self.root, bg="#222222")
        self.right_frame.pack(side=tkinter.RIGHT, expand=True, fill=tkinter.BOTH, padx=20, pady=20)
    
        # Word Input Section
        self.input_frame = tkinter.Frame(self.right_frame, bg="#222222")
        self.input_frame.pack(pady=5, fill=tkinter.X)

        self.instruction_label = tkinter.Label(
            self.input_frame, text="Enter Word to Practice:", font=("Arial", 11, "bold"), fg="#ffffff", bg="#222222"
        )
        self.instruction_label.pack(anchor=tkinter.W)

        self.word_entry = tkinter.Entry(self.input_frame, font=("Arial", 14), bg="#333333", fg="#ffffff", insertbackground="white")
        self.word_entry.pack(side=tkinter.LEFT, fill=tkinter.X, expand=True, pady=5, ipady=3)
        self.word_entry.bind("<Return>", lambda event: self._SubmitWord())

        self.submit_btn = tkinter.Button(
            self.input_frame, text="Submit", font=("Arial", 11, "bold"), bg="#00aa88", fg="#ffffff", padx=10, command=self._SubmitWord
        )
        self.submit_btn.pack(side=tkinter.RIGHT, padx=(5, 0), pady=5)

        # Voice to text button
        self.voice_btn = tkinter.Button(
            self.right_frame, text="🎤 Speak Word (Voice-to-Text)", font=("Arial", 11, "bold"), bg="#4285F4", fg="#ffffff", padx=10, pady=6, command=self._StartVoiceInputThread
        )
        self.voice_btn.pack(fill=tkinter.X, pady=(0, 10))

        # Immutable Target Word Display
        self.target_display_var = tkinter.StringVar(value="Target Word: [ None ]")
        self.target_label = tkinter.Label(
            self.right_frame, textvariable=self.target_display_var, font=("Arial", 13, "italic"), fg="#aaaaaa", bg="#222222"
        )
        self.target_label.pack(anchor=tkinter.W, pady=2)

        # Target Letter Instruction / Expected Sign
        self.expected_text_var = tkinter.StringVar(value="Sign Target: -")
        self.expected_label = tkinter.Label(
            self.right_frame, textvariable=self.expected_text_var, font=("Arial", 14, "bold"), fg="#ffcc00", bg="#222222"
        )
        self.expected_label.pack(anchor=tkinter.W, pady=2)

        # Current Detected Sign Display
        self.prediction_text_var = tkinter.StringVar(value="Current Prediction: Waiting...")
        self.prediction_label = tkinter.Label(
            self.right_frame, textvariable=self.prediction_text_var, font=("Arial", 12, "bold"), fg="#00ffcc", bg="#222222"
        )
        self.prediction_label.pack(anchor=tkinter.W, pady=5)

        # Panel for Target Letter Reference Image ([LETTER].jpg)
        self.image_panel = tkinter.Label(self.right_frame, bg="#333333", width=180, height=180)
        self.image_panel.pack(pady=10)

        # Quit Button
        self.quit_button = tkinter.Button(
            self.right_frame,
            text="Quit",
            font=("Arial", 11, "bold"),
            fg="#ffffff",
            bg="#d9534f",
            activebackground="#c9302c",
            activeforeground="#ffffff",
            bd=0,
            padx=15,
            pady=6,
            command=self._OnClosing
        )
        self.quit_button.pack(pady=5)
    
        # Initialize OpenCV Video Capture & Smoothing Variables
        self.cap = cv2.VideoCapture(0)
        self.prediction_queue = deque(maxlen=15)  
        self.current_displayed_sign = None
        self.cached_img_tk = None
    
        self.root.protocol("WM_DELETE_WINDOW", self._OnClosing)
    
        self.root.after(15, self._UpdateGui)
        self.root.mainloop()

    def _LogSpeech(
            self,
            text: str
        ) -> None:
        """Safely appends spoken text to the scrollable transcript box on the main thread.

        Args:
            text (str): string to display
        """
        def update_box():
            self.transcript_box.config(state=tkinter.NORMAL)
            self.transcript_box.insert(tkinter.END, f"• {text}\n")
            self.transcript_box.see(tkinter.END)  # Auto-scrolls to the bottom
            self.transcript_box.config(state=tkinter.DISABLED)
        
        self.root.after(0, update_box)
    
    def _SubmitWord(
            self
        ) -> None:
        """Submit word from text field to gui and sanitize it
        """
        entered_text = self.word_entry.get().strip().upper()
        if entered_text:
            self.target_word = entered_text
            self.current_index = 0
            self.is_active_practice = True
            self.word_entry.delete(0, tkinter.END)
            self.target_display_var.set(f"Target Word: {self.target_word}")
            self.wrong_frame_count = 0
            self.last_spoken_wrong_sign = None
            threading.Thread(target=self.TextToSpeech, args=(f"Spell {self.target_word} using ASL",), daemon=True).start()
            self._UpdateTargetPrompt()

    def _StartVoiceInputThread(
            self
        ) -> None:
        """Run speech recognition in a background thread to prevent freezing the Tkinter GUI
        """
        self.prediction_text_var.set("Current Prediction: Listening for speech...")
        threading.Thread(target=self._HandleVoiceInput, daemon=True).start()

    def _HandleVoiceInput(
            self
        ) -> None:
        """Get speech to text input and sanitize and display it
        """
        spoken_text = self.SpeechToText()
        if spoken_text:
            # Clean up text (take the first word if multiple are spoken, uppercase it)
            clean_word = spoken_text.split()[0].upper()
            # Safely update GUI elements from the main thread loop
            self.root.after(0, lambda: self._ApplyVoiceWord(clean_word))

    def _ApplyVoiceWord(
            self, 
            word: str
        ) -> None:
        """Using the word gathered from speech to text, update the gui

        Args:
            word (str): word from stt
        """
        self.target_word = word
        self.current_index = 0
        self.is_active_practice = True
        self.target_display_var.set(f"Target Word: {self.target_word}")
        self.wrong_frame_count = 0
        self.last_spoken_wrong_sign = None
        threading.Thread(target=self.TextToSpeech, args=(f"Spell {self.target_word} using ASL",), daemon=True).start()
        self._UpdateTargetPrompt()

    def _UpdateTargetPrompt(
                self
            ) -> None:
        if self.current_index < len(self.target_word):
            current_char = self.target_word[self.current_index]
            progress_str = "".join([f"[{c}]" if i == self.current_index else c for i, c in enumerate(self.target_word)])
            self.target_display_var.set(f"Progress: {progress_str}")
            self.expected_text_var.set(f"Sign Target: '{current_char}'")
            self._LoadLetterImage(current_char)
            self.wrong_frame_count = 0
            self.last_spoken_wrong_sign = None
        else:
            self.success_sound.play()
            self.target_display_var.set(f"Word Completed: {self.target_word} 🎉")
            self.expected_text_var.set("Great job! Enter a new word.")
            self._LoadLetterImage(None)
            self.is_active_practice = False

    def _LoadLetterImage(
            self, 
            letter: str | None = None
        ) -> None:
        """Load the image for letter to the gui

        Args:
            letter (str | None, optional): string name of letter (e.g. "a"). Defaults to None.
        """
        if letter is None:
            self.image_panel.configure(image="", text="Complete!", fg="white")
            self.cached_img_tk = None
            return

        img_path = os.path.join(os.getcwd(), "assets", "images", "instructions", f"{letter}.jpg")
        if os.path.exists(img_path):
            img = Image.open(img_path)
            img = img.resize((160, 160), Image.Resampling.LANCZOS)
            self.cached_img_tk = ImageTk.PhotoImage(img)
            self.image_panel.configure(image=self.cached_img_tk, text="")
        else:
            self.cached_img_tk = None
            self.image_panel.configure(image="", text=f"Missing ref:\n{letter}.jpg", fg="white")

    def _UpdateGui(
            self
        ) -> None:   
        """Update the gui cv2 display with the hand skeleton
        """             
        success, frame = self.cap.read()
        if success:
            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=rgb_frame)

            detection_result = self.detector.detect(mp_image)
            frame_prediction = "Uncertain"

            if detection_result.hand_landmarks:
                for hand_landmarks in detection_result.hand_landmarks:
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

                    # Extract & Normalize landmarks for static prediction
                    landmarks = []
                    for lm in hand_landmarks:
                        landmarks.append([lm.x, lm.y, lm.z])

                    landmarks = numpy.array(landmarks)
                    wrist = landmarks[0]
                    landmarks = landmarks - wrist
                    scale_distance = numpy.linalg.norm(landmarks[9])

                    if scale_distance > 0:
                        landmarks = landmarks / scale_distance

                    features = landmarks.flatten().reshape(1, -1)

                    # Static Model Predict
                    prediction = self.clf.predict(features)[0]
                    probabilities = self.clf.predict_proba(features)
                    confidence = numpy.max(probabilities)

                    if confidence > 0.7:
                        frame_prediction = str(prediction).upper()

            # Smoothing queue logic for static signs
            self.prediction_queue.append(frame_prediction)
            counts = Counter(self.prediction_queue)
            most_common_sign, frequency = counts.most_common(1)[0]

            if frequency >= 8 and most_common_sign != "Uncertain":
                smoothed_sign = most_common_sign
                self.prediction_text_var.set(f"Current Prediction: {smoothed_sign} (Stable)")
            else:
                smoothed_sign = "Uncertain"
                self.prediction_text_var.set("Current Prediction: Transitioning...")

            # ==========================================
            # ACTIVE PRACTICE & DYNAMIC SIGN EVALUATION
            # ==========================================
            if self.is_active_practice and self.current_index < len(self.target_word):
                target_char = self.target_word[self.current_index]

                if target_char == "Z":
                    self.wrong_frame_count = 0
                    self.last_spoken_wrong_sign = None
                    if detection_result.hand_landmarks:
                        for hand_landmarks in detection_result.hand_landmarks:
                            tip = hand_landmarks[8]
                            self.trajectory_history.append((tip.x, tip.y))

                    if len(self.trajectory_history) >= 30:
                        points = numpy.array(self.trajectory_history)
                        p1, p2, p3 = points[:10], points[10:20], points[20:]
                        seg1_dx = p1[-1][0] - p1[0][0]
                        seg2_dx = p2[-1][0] - p2[0][0]
                        seg2_dy = p2[-1][1] - p2[0][1]
                        seg3_dx = p3[-1][0] - p3[0][0]

                        if seg1_dx > 0.04 and seg2_dx < -0.03 and seg2_dy > 0.03 and seg3_dx > 0.04:
                            self.success_sound.play()
                            self.current_index += 1
                            self.trajectory_history.clear()
                            self._UpdateTargetPrompt()

                elif target_char == "J":
                    self.wrong_frame_count = 0
                    self.last_spoken_wrong_sign = None
                    if detection_result.hand_landmarks:
                        for hand_landmarks in detection_result.hand_landmarks:
                            tip = hand_landmarks[20]
                            self.trajectory_history.append((tip.x, tip.y))

                    if len(self.trajectory_history) >= 25:
                        points = numpy.array(self.trajectory_history)
                        start_x = points[0][0]
                        min_y_idx = numpy.argmin(points[:, 1])
                        end_x = points[-1][0]

                        dipped_down = min_y_idx > 5 and min_y_idx < len(points) - 5
                        curved_left = end_x < start_x - 0.02

                        if dipped_down and curved_left:
                            self.success_sound.play()
                            self.current_index += 1
                            self.trajectory_history.clear()
                            self._UpdateTargetPrompt()

                else:
                    self.trajectory_history.clear()
                    if smoothed_sign == target_char:
                        self.match_frame_count += 1
                        self.wrong_frame_count = 0
                        self.last_spoken_wrong_sign = None
                        
                        if self.match_frame_count >= 10:
                            self.success_sound.play()
                            self.current_index += 1
                            self.match_frame_count = 0
                            self._UpdateTargetPrompt()
                    else:
                        self.match_frame_count = max(0, self.match_frame_count - 1)
                        
                        # Monitor wrong form consistency
                        if smoothed_sign != "Uncertain" and smoothed_sign != target_char:
                            if smoothed_sign == self.last_spoken_wrong_sign:
                                self.wrong_frame_count += 1
                            else:
                                self.last_spoken_wrong_sign = smoothed_sign
                                self.wrong_frame_count = 1

                            # Trigger TTS feedback after holding a persistent incorrect sign for ~30 frames (~1s)
                            if self.wrong_frame_count == 30:
                                advice = self.CorrectBadForm(target_char, smoothed_sign)
                                if advice:
                                    threading.Thread(target=self.TextToSpeech, args=(advice,), daemon=True).start()
                                self.wrong_frame_count = 0
                        else:
                            self.wrong_frame_count = max(0, self.wrong_frame_count - 1)

            # Convert OpenCV Frame to Tkinter PhotoImage for the webcam panel
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(cv2image)
            img_tk_webcam = ImageTk.PhotoImage(image=img_pil)
            self.webcam_label.imgtk = img_tk_webcam
            self.webcam_label.configure(image=img_tk_webcam)

        self.root.after(15, self._UpdateGui)

    def _OnClosing(
            self
        ) -> None:
        """Event for window close
        """
        self.cap.release()
        self.root.destroy()

    def TextToSpeech(
            self,
            text: str
        ) -> int:
        """Convert text to speech

        Args:
            text (str): text to convert to speech

        Returns:
            int: return code. 0 if successful
        """
        self._LogSpeech(text)
        try:
            engine = pyttsx3.init()
            engine.setProperty(name="rate", value=150)
            engine.say(text)
            engine.runAndWait()
            return 0
        except Exception as e:
            print(e)
            return -1

    def SpeechToText(
            self,
            sample_rate: int = 44100,
            seconds: int = 3
        ) -> str:
        """Convert speech input to text

        Args:
            sample_rate (int, optional): sample rate for recording. Defaults to 44100.
            seconds (int, optional): seconds for recording_. Defaults to 3.

        Returns:
            str: text string of speech input
        """
        recognizer = speech_recognition.Recognizer()

        while True:
            recording = sounddevice.rec(
                int(seconds * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype="int16"
            )

            self.beep_sound.play()

            sounddevice.wait()

            audio = speech_recognition.AudioData(
                recording.tobytes(),
                sample_rate,
                2
            )

            try:
                text = recognizer.recognize_google(audio)
                text = str(text).lower().strip()
                return text

            except speech_recognition.UnknownValueError:
                self.TextToSpeech("Could not understand. Please speak again after the beep.")

    def CorrectBadForm(self, target: str, actual: str) -> str:
        # .get() safely handles missing keys and returns an empty string if no rule exists
        return BAD_FORM_FEEDBACK_DICT.get(target.upper(), {}).get(actual.upper(), "")



def main():
    AslRecognizeGui()

if __name__ == "__main__":
    main()