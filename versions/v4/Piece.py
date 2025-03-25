class Piece:
    def __init__(self, width, height):
        self.x = 0  
        self.y = 0  
        self.width = width
        self.height = height
        self.rotated = False 
        self.plate = 0 

    def __repr__(self):
        return f"({self.x},{self.y}) ({self.width}x{self.height})"
    
