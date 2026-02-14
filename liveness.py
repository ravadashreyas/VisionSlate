import mediapipe as mp
import numpy as np
import time
import math


# MediaPipe Face Mesh landmark indices for eyes
# Right eye
RIGHT_EYE_TOP = [159, 145]
RIGHT_EYE_BOTTOM = [153, 144]
RIGHT_EYE_LEFT = 33
RIGHT_EYE_RIGHT = 133

# Left eye
LEFT_EYE_TOP = [386, 374]
LEFT_EYE_BOTTOM = [380, 373]
LEFT_EYE_LEFT = 362
LEFT_EYE_RIGHT = 263

BLINK_THRESHOLD = 0.20       # EAR below this = eyes closed
BLINKS_REQUIRED = 2          # blinks needed to pass liveness
LIVENESS_TIMEOUT = 10.0      # seconds to complete blink check
DEPTH_THRESHOLD = 0.01       # minimum z std-dev for a real 3D face


def _landmark_distance(lm1, lm2):
    """Euclidean distance between two face landmarks."""
    return math.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2)


def _eye_aspect_ratio(landmarks, top_ids, bottom_ids, left_id, right_id):
    """Calculate Eye Aspect Ratio (EAR) for one eye.
    
    EAR = average(vertical distances) / horizontal distance
    When eyes are open EAR ≈ 0.25-0.30, when closed EAR < 0.20
    """
    # Average vertical distance (top-to-bottom lid)
    vert_sum = 0
    for t, b in zip(top_ids, bottom_ids):
        vert_sum += _landmark_distance(landmarks[t], landmarks[b])
    vertical = vert_sum / len(top_ids)

    # Horizontal distance (corner to corner)
    horizontal = _landmark_distance(landmarks[left_id], landmarks[right_id])

    if horizontal == 0:
        return 0.3  # avoid division by zero, assume open

    return vertical / horizontal


class LivenessChecker:
    def __init__(self):
        self.blink_count = 0
        self.eyes_closed = False       # tracks if eyes are currently shut
        self.alive = False             # True once liveness confirmed
        self.start_time = time.time()
        self.status = f"Please blink twice (0/{BLINKS_REQUIRED})"

    def process_frame(self, face_landmarks):
        """Called every frame with Face Mesh landmarks. Updates blink counter and depth check.
        
        Args:
            face_landmarks: a single face's landmarks from face_mesh results
                           (face_results.multi_face_landmarks[0])
        """
        if self.alive:
            return

        lms = face_landmarks.landmark

        
        right_ear = _eye_aspect_ratio(lms, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
                                       RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT)
        left_ear = _eye_aspect_ratio(lms, LEFT_EYE_TOP, LEFT_EYE_BOTTOM,
                                      LEFT_EYE_LEFT, LEFT_EYE_RIGHT)
        avg_ear = (right_ear + left_ear) / 2.0

        # Detect blink: eyes close (EAR drops) then open (EAR rises)
        if avg_ear < BLINK_THRESHOLD:
            self.eyes_closed = True
        else:
            if self.eyes_closed:
                # Eyes just opened again → that's one blink
                self.blink_count += 1
                print(f"[Liveness] Blink detected ({self.blink_count}/{BLINKS_REQUIRED})")
            self.eyes_closed = False


        z_values = [lm.z for lm in lms]
        z_std = np.std(z_values)

        if z_std < DEPTH_THRESHOLD:
            self.status = "Flat image detected — move closer"
            return

       
        if self.blink_count >= BLINKS_REQUIRED:
            self.alive = True
            self.status = "Liveness confirmed ✓"
            print("[Liveness] Confirmed — real person detected")
            return

        
        elapsed = time.time() - self.start_time
        if elapsed > LIVENESS_TIMEOUT:
            self.blink_count = 0
            self.start_time = time.time()
            self.status = f"Timed out — try again. Blink twice (0/{BLINKS_REQUIRED})"
            return

        self.status = f"Please blink twice ({self.blink_count}/{BLINKS_REQUIRED})"

    def is_alive(self):
        """Returns True if liveness has been confirmed."""
        return self.alive

    def reset(self):
        """Reset for a new authentication session."""
        self.blink_count = 0
        self.eyes_closed = False
        self.alive = False
        self.start_time = time.time()
        self.status = f"Please blink twice (0/{BLINKS_REQUIRED})"

    def get_status(self):
        """Return a status string for the UI."""
        return self.status
