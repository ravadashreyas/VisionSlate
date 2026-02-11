import cv2
import numpy as np
from PIL import Image
from pix2tex.cli import LatexOCR
from sympy.parsing.latex import parse_latex
from sympy import simplify, solve, Symbol, Eq
import sympy
import time
import threading
import re


COOLDOWN_SECONDS = 10


class LocalMathSolver:
    def __init__(self):
        self.model = LatexOCR()
        print("LaTeX-OCR model loaded.")

        self.result = None
        self.is_solving = False
        self.last_solve_time = 0
        self.display_end_time = 0
        self.DISPLAY_DURATION = 5.0

    def solve(self, canvas_img):
        """Send canvas image to local OCR + SymPy in a background thread."""
        if self.is_solving:
            return
        if time.time() - self.last_solve_time < COOLDOWN_SECONDS:
            return

        self.is_solving = True
        self.result = "Solving..."

        thread = threading.Thread(target=self._solve_locally, args=(canvas_img.copy(),))
        thread.daemon = True
        thread.start()

    def _preprocess_canvas(self, canvas_img):
        """Convert green-on-black canvas to clean black-on-white for OCR."""
        # Convert to grayscale
        gray = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)

        # Threshold to get binary mask of drawn content
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # Find bounding box of drawn content
        coords = cv2.findNonZero(binary)
        if coords is None:
            return None  # nothing drawn

        x, y, bw, bh = cv2.boundingRect(coords)

        # Add padding around the content
        pad = 40
        x = max(0, x - pad)
        y = max(0, y - pad)
        bw = min(canvas_img.shape[1] - x, bw + 2 * pad)
        bh = min(canvas_img.shape[0] - y, bh + 2 * pad)

        # Crop to just the drawn area
        cropped = binary[y:y+bh, x:x+bw]

        inverted = cv2.bitwise_not(cropped)

        return inverted

    def _solve_locally(self, canvas_img):
        """Background thread: OCR the image, parse LaTeX, solve with SymPy."""
        try:
            # 1. Preprocess canvas for OCR
            processed = self._preprocess_canvas(canvas_img)
            if processed is None:
                self.result = "Nothing drawn"
                return

            # 2. Convert to PIL Image
            pil_image = Image.fromarray(processed)

            # 3. Run LaTeX-OCR to get a LaTeX string
            latex_str = self.model(pil_image)
            print(f"[OCR] Recognized LaTeX: {latex_str}")

            if not latex_str or latex_str.strip() == "":
                self.result = "Could not read anything"
                return

            # 3. Parse and solve with SymPy
            answer = self._parse_and_solve(latex_str)
            self.result = answer

        except Exception as e:
            print(f"[Solver Error] {e}")
            self.result = f"Error: {str(e)[:50]}"
        finally:
            self.is_solving = False
            self.last_solve_time = time.time()
            self.display_end_time = time.time() + self.DISPLAY_DURATION

    def _parse_and_solve(self, latex_str):
        """Parse a LaTeX string and try to solve or simplify it."""
        latex_str = latex_str.strip().strip("$")

        # Check if it's an equation (contains =)
        if "=" in latex_str:
            sides = latex_str.split("=", 1)
            try:
                lhs = parse_latex(sides[0].strip())
                rhs = parse_latex(sides[1].strip())
                equation = Eq(lhs, rhs)

                # Find free variables and solve
                free_vars = equation.free_symbols
                if free_vars:
                    var = sorted(free_vars, key=str)[0]
                    solutions = solve(equation, var)
                    if solutions:
                        sol_str = ", ".join(str(s) for s in solutions)
                        return f"{var} = {sol_str}"
                    else:
                        return "No solution found"
                else:
                    # No variables — check if equation is true/false
                    result = simplify(lhs - rhs)
                    return "True" if result == 0 else f"False ({lhs} ≠ {rhs})"

            except Exception as e:
                print(f"[Parse Error] Could not parse equation: {e}")
                return f"OCR: {latex_str}"
        else:
            try:
                expr = parse_latex(latex_str)
                simplified = simplify(expr)

                # If it's purely numeric, evaluate it
                if not simplified.free_symbols:
                    evaluated = float(simplified.evalf())
                    if evaluated == int(evaluated):
                        return str(int(evaluated))
                    return f"{evaluated:.4f}"
                else:
                    return f"= {simplified}"

            except Exception as e:
                print(f"[Parse Error] Could not parse expression: {e}")
                return f"OCR: {latex_str}"

    def draw_result(self, frame):
        """Draw the solve result on the frame if there's something to show."""
        if self.result is None or time.time() > self.display_end_time:
            return frame

        h, w, _ = frame.shape

        overlay = frame.copy()
        cv2.rectangle(overlay, (20, h - 90), (w - 20, h - 20), (40, 40, 40), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        label = f"Answer: {self.result}"
        cv2.putText(frame, label, (40, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        return frame
