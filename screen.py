from canvas import Canvas

class Screen:
    def __init__(self):
        self.screen = None
        self.canvas = []
    
    def add_canvas(self, Canvas):
        self.canvas.append(Canvas)
    
    def get_canvas(self, index):
        return self.canvas[index]
    
    def get_all_canvas(self):
        return self.canvas
    
    def delete_screen(self):
        self.screen = None
        for canvas in self.canvas:
            canvas.delete_canvas()
        self.canvas = []