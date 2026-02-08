import cv2
import mediapipe as mp
import numpy as np
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


from canvas import Canvas
from screen import Screen

from distance import calc_distance, calc_distance_regular

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)


brush_thickness = 7
eraser_thickness = 50
prev_x, prev_y = 0, 0
save_counter = 1

screen = Screen()
canvas = None

pen = False

index_color = (128, 128, 128)
middle_color = (0, 0, 255)
thumb_color = (0, 255, 0)

current_canvas_count = 0

model_path = 'gesture_recognizer.task' 

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

recognizer = GestureRecognizer.create_from_options(options)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    if canvas is None:
        canvas = Canvas(w, h)
        screen.add_canvas(canvas)
    
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
            
            fingertip_ids = [4, 8, 12]

            for finger in fingertip_ids:
                finger_tip = hand_landmarks.landmark[finger]

                if finger == 8:
                    pointer_x, pointer_y = int(finger_tip.x * w), int(finger_tip.y * h)
                    cv2.circle(frame, (pointer_x, pointer_y), 15, index_color, -1)
                else:
                    cx, cy = int(finger_tip.x * w), int(finger_tip.y * h)
                    if finger == 12:
                        color = middle_color
                    else:
                        color = thumb_color
                    cv2.circle(frame, (cx, cy), 15, color, -1)

            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]

            pen_distance = calc_distance(index_tip, thumb_tip)
            eraser_distance = calc_distance(index_tip, middle_tip)
    
            if (pen_distance < 0.07 and not(thumb_tip.x > hand_landmarks.landmark[2].x) or gesture == "draw"): 
                index_color = (0, 255, 0)
                pen = True
                eraser = False
            elif (eraser_distance < 0.07 and not(middle_tip.x > hand_landmarks.landmark[10].x) or gesture == "eraser"):
                index_color = (0, 0, 255) 
                pen = False
                eraser = True
            else:
                index_color = (128, 128, 128)
                pen = False
                eraser = False

            if gesture == "clear":
                canvas.clear_canvas()
            if gesture == "switch":
                for temp_canvas in screen.get_all_canvas():
                    number_of_canvases = len(screen.get_all_canvas())
                    canvas_length = (w / number_of_canvases + 1)

                new_canvas = Canvas(w, h)
                screen.add_canvas(new_canvas)
                canvas = screen.get_canvas(current_canvas_count)
                current_canvas_count += 1
                 
            if pen:
                if (prev_x != 0 and prev_y != 0) or (calc_distance_regular(prev_x, pointer_x) < 0.3) or (calc_distance_regular(prev_y, pointer_y) < 0.3):
                    cv2.line(canvas.get_canvas(), (prev_x, prev_y), (pointer_x, pointer_y), (0, 255, 0), brush_thickness)
                prev_x, prev_y = pointer_x, pointer_y
            elif eraser:
                if (prev_x != 0 and prev_y != 0) or (calc_distance_regular(prev_x, pointer_x) < 0.3) or (calc_distance_regular(prev_y, pointer_y) < 0.3):
                    cv2.line(canvas.get_canvas(), (prev_x, prev_y), (pointer_x, pointer_y), (0, 0, 0), eraser_thickness)
                prev_x, prev_y = pointer_x, pointer_y
            else:
                prev_x, prev_y = pointer_x, pointer_y

    imgGray = cv2.cvtColor(canvas.get_canvas(), cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 10, 255, cv2.THRESH_BINARY_INV)
    frame = cv2.bitwise_and(frame, frame, mask=imgInv)
    frame = cv2.bitwise_or(frame, canvas.get_canvas())
    cv2.imshow("Virtual Whiteboard", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# get width of screen divide it by the number of elements in the screen length + 1 
# and then place each canvas on the top half of the screen and the height should be the same
# and you can click on the canvas u want to switch to