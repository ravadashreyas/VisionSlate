import cv2
import numpy as np
from google import genai
from PIL import Image
from dotenv import load_dotenv
import os
import time
import threading

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

PROMPT = (
    "Look at this handwritten whiteboard image. "
    "If there is a math equation or expression, solve it and return ONLY the final answer (e.g. 'x = 5' or '42'). "
    "If there is handwritten text, read it and return what it says. "
    "Keep your response short — one line max."
)

COOLDOWN_SECONDS = 30


class Solver:
    def __init__(self):
        self.result = None
        self.is_solving = False
        self.last_solve_time = 0
        self.display_end_time = 0
        self.DISPLAY_DURATION = 5.0

    def solve(self, canvas_img):
        """Send canvas image to Gemini in a background thread."""
        if self.is_solving:
            return
        if time.time() - self.last_solve_time < COOLDOWN_SECONDS:
            return

        self.is_solving = True
        self.result = "Solving..."

        thread = threading.Thread(target=self._call_api, args=(canvas_img.copy(),))
        thread.daemon = True
        thread.start()

    def _call_api(self, canvas_img):
        """Background API call to Gemini."""
        try:
            img_rgb = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)

            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[PROMPT, pil_image],
            )
            self.result = response.text.strip()
        except Exception as e:
            self.result = f"Error: {str(e)}"
        finally:
            self.is_solving = False
            self.last_solve_time = time.time()
            self.display_end_time = time.time() + self.DISPLAY_DURATION

    def draw_result(self, frame):
        """Draw the solve result on the frame if there's something to show."""
        if self.result is None or time.time() > self.display_end_time:
            return frame

        h, w, _ = frame.shape

        overlay = frame.copy()
        cv2.rectangle(overlay, (20, h - 90), (w - 20, h - 20), (40, 40, 40), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Draw the result text
        label = f"AI: {self.result}"
        cv2.putText(frame, label, (40, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        return frame
