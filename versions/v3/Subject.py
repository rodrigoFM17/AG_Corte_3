from Piece import Piece
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil
import numpy as np

class Subject:
    def __init__(self, plate_width, plate_height, pieces):
        self.plates_used = []  # Lista de láminas utilizadas
        self.pieces = pieces  # Lista de piezas asignadas
        self.fitness = 0
        self.plate_width = plate_width
        self.plate_height = plate_height
        self.plates_matrix = []
        self.empty_areas = []  # Lista de áreas vacías en cada lámina

    def set_pieces(self, pieces):
        self.pieces = pieces
    
    def place_pieces_v2(self):

        print(self.pieces)
        plates = [[]]
        pieces_matrix = np.zeros((self.plate_width, self.plate_height))

        for piece in self.pieces :

            piece_w, piece_h = piece.width, piece.height
            placed = False
            
            while not placed :
                y = 0
                while y < self.plate_height:

                    piece_placed_y = y + piece_h
                    x = 0
                    
                    while x < self.plate_width:

                        piece_placed_x = x + piece_w
                        
                        if (pieces_matrix[x, y] == 0 and piece_placed_x < self.plate_width and piece_placed_y < self.plate_height):
                            piece_matrix = pieces_matrix[ x:piece_placed_x , y:piece_placed_y ]
                            if np.all( piece_matrix == 0): 
                                piece.x = x
                                piece.y = y
                                plates[-1].append(piece)
                                pieces_matrix[ x:piece_placed_x, y:piece_placed_y ] = 1
                                x = self.plate_width
                                y = self.plate_height
                                placed = True
                                print("pieza ubicada correctamente")
                                print(pieces_matrix[0])
                            else :
                                x += 1
                        else : 
                            x += 1

                    y += 1

                if not placed :
                    print("no se pudo ubicar en esta lamina")
                    plates.append([])
                    self.plates_matrix.append(pieces_matrix)
                    pieces_matrix = np.zeros((self.plate_width, self.plate_height))

        self.plates_used = plates
        

    def place_pieces(self):
        """Coloca las piezas en las láminas usando una matriz de ocupación de 1 y 0."""
        self.plates_used = []  # Lista de láminas
        plate_matrix = np.zeros((self.plate_height, self.plate_width), dtype=int)  # Matriz de ocupación de la primera lámina
        current_plate = []  # Lista de piezas en la lámina actual

        for piece in self.pieces:
            placed = False  # Bandera para verificar si la pieza fue colocada

            # Buscar la primera posición libre donde quepa la pieza
            for y in range(self.plate_height - piece.height + 1):
                for x in range(self.plate_width - piece.width + 1):
                    # Verificar si la pieza cabe en este espacio sin solaparse
                    if np.all(plate_matrix[y:y + piece.height, x:x + piece.width] == 0):
                        # Colocar la pieza en la lámina
                        piece.x, piece.y = x, y
                        piece.plate = len(self.plates_used) + 1  # Índice de la lámina
                        current_plate.append(piece)

                        # Marcar el espacio ocupado en la matriz
                        plate_matrix[y:y + piece.height, x:x + piece.width] = 1

                        placed = True
                        break
                if placed:
                    break

            # Si la pieza no pudo colocarse, usar una nueva lámina
            if not placed:
                self.plates_used.append(current_plate)  # Guardar la lámina actual
                plate_matrix = np.zeros((self.plate_height, self.plate_width), dtype=int)  # Nueva matriz vacía
                current_plate = []  # Nueva lista de piezas

                # Intentar colocar la pieza en la nueva lámina
                piece.x, piece.y = 0, 0  # Se reinicia la posición de colocación
                piece.plate = len(self.plates_used) + 1
                current_plate.append(piece)
                plate_matrix[:piece.height, :piece.width] = 1  # Marcar la ocupación

        # Agregar la última lámina utilizada
        if current_plate:
            self.plates_used.append(current_plate)

    def get_fitness(self):
        """Calcula el fitness como la sumatoria del aprovechamiento de cada lámina."""
        total_fitness = 0  # Inicializar fitness
        area_lamina = self.plate_width * self.plate_height  # Área total de una lámina

        for lamina in self.plates_used:
            area_usada = sum(pieza.width * pieza.height for pieza in lamina)  # Sumatoria de áreas de piezas
            total_fitness += area_usada / area_lamina # Agregar el aprovechamiento de esta lámina

        self.fitness = total_fitness / len(self.plates_used) # Guardar fitness
        return self.fitness

    def __repr__(self):
        result = ""
        for i, plate in enumerate(self.plates_used):
            result += f"Lámina {i+1}:\n"
            for piece in plate:
                result += f"{piece}\n"
            result += "-" * 40 + "\n"
        return result
    
    def view_plates_distribution(self):
        for i, lamina in enumerate(self.plates_used):
            print(f"Lámina {i+1}:")
            for pieza in lamina:
                print(f"  Pieza en ({pieza.x}, {pieza.y}) de tamaño ({pieza.width}x{pieza.height}) en lámina {pieza.plate}")

    def generate_images(self, folder="laminas"):
        """Genera imágenes de la distribución de piezas en cada lámina, limpiando la carpeta antes de generarlas."""
        # Limpiar la carpeta antes de generar nuevas imágenes
        if os.path.exists(folder):
            shutil.rmtree(folder)  # Elimina la carpeta y todo su contenido
        os.makedirs(folder, exist_ok=True)  # Crear la carpeta limpia

        for i, plate in enumerate(self.plates_used):
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xlim(0, self.plate_width)
            ax.set_ylim(0, self.plate_height)
            ax.set_xlabel("Ancho")  # Etiqueta para el eje X
            ax.set_ylabel("Alto")   # Etiqueta para el eje Y
            ax.set_title(f"Lámina {i+1}")

            # Configurar las marcas en los ejes X e Y para mostrar las dimensiones
            ax.set_xticks(range(0, self.plate_width + 1, max(1, self.plate_width // 10)))
            ax.set_yticks(range(0, self.plate_height + 1, max(1, self.plate_height // 10)))
            ax.grid(True, linestyle="--", linewidth=0.5)  # Agrega líneas de referencia

            # Dibujar cada pieza
            for piece in plate:
                rect = patches.Rectangle(
                    (piece.x, self.plate_height - piece.y - piece.height),
                    piece.width,
                    piece.height,
                    linewidth=1,
                    edgecolor="black",
                    facecolor="lightblue"
                )
                ax.add_patch(rect)
                ax.text(
                    piece.x + piece.width / 2,
                    self.plate_height - piece.y - piece.height / 2,
                    f"{piece.width}x{piece.height}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="black",
                    bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.3")
                )

            # Guardar imagen
            plt.savefig(os.path.join(folder, f"lamina_{i+1}.png"))
            plt.close()

        print(f"✅ Imágenes generadas en la carpeta '{folder}'.")

# piezas = [Piece(50, 50), Piece(50, 50), Piece(50, 50), Piece(50, 50),Piece(50, 50),Piece(50, 50),Piece(50, 50),Piece(50, 50),Piece(50, 50),Piece(50, 50),
#           Piece(20, 20), Piece(20, 20), Piece(20, 20), Piece(20, 20), Piece(20, 20),Piece(20, 20),Piece(20, 20),Piece(20, 20),]
# subject = Subject()
# subject.place_pieces(piezas, 120, 120)

# for i, lamina in enumerate(subject.plates_used):
#     print(f"Lámina {i+1}:")
#     for pieza in lamina:
#         print(f"  Pieza en ({pieza.x}, {pieza.y}) de tamaño ({pieza.width}x{pieza.height}) en lámina {pieza.plate}")

# print(f"fitness: {subject.get_fitness(120, 120)}")