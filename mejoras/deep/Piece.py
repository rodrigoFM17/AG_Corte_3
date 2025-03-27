class Piece:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.rotated = False  # Ahora es parte del cromosoma
        self.plate = 0  # Asignación evolutiva de lámina
        self.x = 0
        self.y = 0

    def rotate(self):
        self.width, self.height = self.height, self.width
        self.rotated = not self.rotated