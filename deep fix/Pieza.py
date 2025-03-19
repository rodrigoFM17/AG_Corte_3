class Pieza:
    def __init__(self, ancho, alto, tipo_id, id=None, rotada=False):
        self.ancho = ancho
        self.alto = alto
        self.tipo_id = tipo_id  # Identificador del tipo de pieza (ej: 1=20x20, 2=30x40)
        self.id = id            # ID único por instancia
        self.rotada = rotada
        self.x = 0
        self.y = 0
        self.en_lamina = 0 # Nueva propiedad para trackear lámina
    
    def rotar(self):
        self.ancho, self.alto = self.alto, self.ancho
        self.rotada = not self.rotada
        return self
    
    def area(self):
        return self.ancho * self.alto
    
    def copia(self):
        return Pieza(self.ancho, self.alto, self.tipo_id, self.id, self.rotada)