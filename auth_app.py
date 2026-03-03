import warnings
import os
warnings.filterwarnings("ignore")
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "3"

import cv2
import mediapipe as mp
import numpy as np
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from gesture_auth import GestureAuth
from liveness import LivenessChecker

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

model_path = 'gesture_recognizer.task' 

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

recognizer = GestureRecognizer.create_from_options(options)

# Create a GestureAuth instance
auth = GestureAuth()
# Create a LivenessChecker instance
liveness = LivenessChecker()


while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
    recognition_result = recognizer.recognize(mp_image)
        
    gesture = "None"
    score = 0.0
        
    if recognition_result.gestures:
        top_gesture = recognition_result.gestures[0][0]
        gesture = top_gesture.category_name
        score = top_gesture.score

    color = (0, 0, 255) if score < 0.5 else (0, 255, 0)
        
    text = f"Gesture: {gesture} ({int(score * 100)}%)"
    cv2.putText(frame, text, (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # --- Face Mesh (Liveness) ---
    face_results = face_mesh.process(rgb_frame)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
            )
            # Pass face_landmarks to your LivenessChecker.process_frame()
            liveness.process_frame(face_landmarks)

    # Only allow gesture auth if liveness.is_alive() is True
    # Display liveness status on screen (liveness.get_status())
    liveness_status = liveness.get_status()
    cv2.putText(frame, liveness_status, (50, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    if liveness.is_alive():
        # Pass the detected gesture and score to your GestureAuth instance every frame
        auth.process_gesture(gesture, score)


    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('e'):
        username = input("Enter username for enrollment: ")
        auth.start_enrollment(username)
        liveness.reset()
    elif key == ord('v'):
        username = input("Enter username for verification: ")
        auth.start_verification(username)
        liveness.reset()
    elif key == 32: # SPACE
        auth.finish_sequence()
    elif key == 27: # ESC
        auth.cancel()


    # Draw UI overlay on the frame
    # Status message (what should the user do next?)
    cv2.putText(frame, auth.status_message, (50, h - 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Current sequence progress
    seq_str = " -> ".join(auth.current_sequence)
    cv2.putText(frame, f"Seq: {seq_str}", (50, h - 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Hold progress bar
    if auth.current_held_gesture and not auth.gesture_registered:
        hold_time = min(time.time() - auth.gesture_start_time, 0.5)
        bar_width = int((hold_time / 0.5) * 200)
        cv2.rectangle(frame, (50, 110), (50 + bar_width, 130), (0, 255, 255), -1)
        cv2.rectangle(frame, (50, 110), (250, 130), (255, 255, 255), 2)

    # Result display (ACCESS GRANTED / ACCESS DENIED)
    if auth.result_display:
        res_color = (0, 255, 0) if auth.result_display == "GRANTED" else (0, 0, 255)
        cv2.putText(frame, f"ACCESS {auth.result_display}", (w // 2 - 150, h // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, res_color, 4)


    cv2.imshow("Gesture Auth", frame)

cap.release()
cv2.destroyAllWindows()
