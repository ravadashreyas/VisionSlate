import cv2
import numpy as np

class Canvas:
    def __init__(self, width, height):
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)

    def get_canvas(self):
        return self.canvas
    
    def save_canvas(self, filename):
        cv2.imwrite(filename, self.canvas)

    def delete_canvas(self):
        self.canvas = None
    
    def clear_canvas(self):
        self.canvas[:] = 0