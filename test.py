import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = 'gesture_recognizer.task' 

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

with GestureRecognizer.create_from_options(options) as recognizer:
    
    cap = cv2.VideoCapture(0)
    
    print("--- TEST MODE STARTED ---")
    print("Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        
        # Convert to MediaPipe's Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Run the AI
        recognition_result = recognizer.recognize(mp_image)
        
        # Get the top result
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
        

        cv2.imshow("Model Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()