import json
import os
import time


PROFILES_DIR = "profiles"


class GestureAuth:
    def __init__(self):
        os.makedirs(PROFILES_DIR, exist_ok=True)

        # === STATE VARIABLES ===
        self.mode = "idle"  # Modes: 'idle', 'enrolling', 'confirming', 'verifying'
        self.current_username = None
        
        # SEQUENCES
        self.enrolled_sequence = []
        self.current_sequence = []      # The currently recorded sequence

        # DEBOUNCING / TIMING
        self.current_held_gesture = None
        self.gesture_start_time = 0.0
        self.gesture_registered = False  # Track if the currently held gesture was already registered

        # UI / FEEDBACK
        self.status_message = "Ready"
        self.result_display = None       # GRANTED / DENIED / None
        self.confirm_attempts = 0        # Track confirmation retries

    def start_enrollment(self, username):
        """Begin enrollment for a new user."""
        self.mode = "enrolling"
        self.current_username = username
        self.enrolled_sequence = []
        self.current_sequence = []
        self.result_display = None
        self.status_message = f"Enrolling {username}: Hold 3+ gestures."

    def start_verification(self, username):
        """Begin verification — load the user's profile and prepare to compare."""
        profile = self._load_profile(username)
        if profile is None:
            self.status_message = f"Error: User '{username}' not found."
            return

        self.mode = "verifying"
        self.current_username = username
        self.enrolled_sequence = profile.get("gesture_sequence", [])
        self.current_sequence = []
        self.result_display = None
        self.status_message = f"Verifying {username}: Repeat your sequence."

    def process_gesture(self, gesture, score):
        """Called EVERY FRAME with the current gesture. YOU handle debouncing here."""
        # 1. Ignore if we are idle
        if self.mode == "idle":
            return
            
        # 2. Ignore garbage data
        if score < 0.6 or gesture.lower() == "none" or not gesture:
            self.current_held_gesture = None # Reset so they can do the same gesture twice
            return
            
        # 3. DEBOUNCING LOGIC
        if gesture != self.current_held_gesture:
            self.current_held_gesture = gesture
            self.gesture_start_time = time.time()
            self.gesture_registered = False
        else:
            # Calculate how long it has been held
            held_time = time.time() - self.gesture_start_time
            if held_time > 0.5 and not self.gesture_registered:
                self.current_sequence.append(gesture)
                self.gesture_registered = True
                self.status_message = f"Registered: {gesture}"

    def finish_sequence(self):
        """Called when user presses SPACE to signal their sequence is done."""
        if self.mode == "enrolling":
            if len(self.current_sequence) >= 3:
                self.enrolled_sequence = self.current_sequence.copy()
                self.mode = "confirming"
                self.current_sequence = []
                self.status_message = "Repeat sequence to confirm."
            else:
                self.status_message = "Sequence too short (min 3)."
            
        elif self.mode == "confirming":
            if self.current_sequence == self.enrolled_sequence:
                self._save_profile()
                self.mode = "idle"
                self.status_message = "Enrollment successful!"
            else:
                self.current_sequence = []
                self.confirm_attempts += 1
                self.status_message = "Mismatch. Try confirming again."
            
        elif self.mode == "verifying":
            if self.current_sequence == self.enrolled_sequence:
                self.result_display = "GRANTED"
                self.status_message = "Access Granted!"
            else:
                self.result_display = "DENIED"
                self.status_message = "Access Denied."
            self.mode = "idle"

    def cancel(self):
        """Called when user presses ESC."""
        self.mode = "idle"
        self.current_username = None
        self.current_held_gesture = None
        self.current_sequence = []
        self.result_display = None
        self.status_message = "Cancelled."

    def _save_profile(self):
        """Save the enrolled sequence to profiles/username.json"""
        filepath = os.path.join(PROFILES_DIR, f"{self.current_username}.json")
        data = {
            "username": self.current_username,
            "gesture_sequence": self.enrolled_sequence,
            "timestamp": time.time()
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def _load_profile(self, username):
        """Load a profile from profiles/username.json, return None if not found."""
        filepath = os.path.join(PROFILES_DIR, f"{username}.json")
        if not os.path.exists(filepath):
            return None
            
        with open(filepath, "r") as f:
            return json.load(f)
