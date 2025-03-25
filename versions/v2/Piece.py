class Piece:
    def __init__(self, width, height):
        self.x = 0  # Coordenada X en la lámina
        self.y = 0  # Coordenada Y en la lámina
        self.width = width
        self.height = height
        self.rotated = False  # Indica si la pieza ha sido rotada
        self.plate = 0  # Índice de la lámina donde se coloca

    def __repr__(self):
        return f"({self.x},{self.y}) ({self.width}x{self.height})"
    
