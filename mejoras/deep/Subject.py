from Piece import Piece
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil
import numpy as np
from scipy.ndimage import label

class Subject:
    def __init__(self, plate_width, plate_height, pieces):
        self.plates_used = []
        self.pieces = pieces
        self.fitness = 0
        self.plate_width = plate_width
        self.plate_height = plate_height
        self.plates_matrix = []
        self.empty_areas = []

    def set_pieces(self, pieces):
        self.pieces = pieces
    
    def place_pieces_v2(self):
        plates = [[]]
        pieces_matrix = np.zeros((self.plate_width, self.plate_height))

        for piece in self.pieces:
            piece_w, piece_h = piece.width, piece.height
            placed = False
            
            while not placed:
                y = 0
                while y < self.plate_height:
                    piece_placed_y = y + piece_h
                    x = 0
                    
                    while x < self.plate_width:
                        piece_placed_x = x + piece_w
                        
                        if (pieces_matrix[x, y] == 0 and piece_placed_x < self.plate_width and piece_placed_y < self.plate_height):
                            piece_matrix = pieces_matrix[x:piece_placed_x, y:piece_placed_y]
                            if np.all(piece_matrix == 0): 
                                piece.x = x
                                piece.y = y
                                plates[-1].append(piece)
                                pieces_matrix[x:piece_placed_x, y:piece_placed_y] = 1
                                x = self.plate_width
                                y = self.plate_height
                                placed = True
                            else:
                                x += 1
                        else: 
                            x += 1
                    y += 1

                if not placed:
                    plates.append([])
                    self.plates_matrix.append(pieces_matrix)
                    pieces_matrix = np.zeros((self.plate_width, self.plate_height))

        self.plates_used = plates
        
    def place_pieces(self):
        self.plates_used = []
        plate_matrices = {}  # Diccionario: {índice_lámina: matriz}

        for piece in self.pieces:
            if piece.plate not in plate_matrices:
                plate_matrices[piece.plate] = np.zeros((self.plate_height, self.plate_width), dtype=int)

            matrix = plate_matrices[piece.plate]
            placed = False

            # Aplica rotación del cromosoma
            if piece.rotated:
                piece.width, piece.height = piece.height, piece.width

            # Busca posición en la lámina asignada
            for y in range(self.plate_height - piece.height + 1):
                for x in range(self.plate_width - piece.width + 1):
                    if np.all(matrix[y:y+piece.height, x:x+piece.width] == 0):
                        piece.x, piece.y = x, y
                        matrix[y:y+piece.height, x:x+piece.width] = 1
                        placed = True
                        break
                if placed:
                    break

            if not placed:
                # Si no cabe, asigna nueva lámina automáticamente (penalizada en fitness)
                piece.plate = len(plate_matrices) + 1
                plate_matrices[piece.plate] = np.zeros((self.plate_height, self.plate_width), dtype=int)
                matrix = plate_matrices[piece.plate]
                matrix[0:piece.height, 0:piece.width] = 1
                piece.x, piece.y = 0, 0

        # Convierte el diccionario a lista de láminas
        self.plates_used = [(plate, matrix) for plate, matrix in plate_matrices.items()]

    def calculate_empty_areas(self, plate_matrix):
        empty_spaces = (plate_matrix == 0).astype(int)
        labeled, num_features = label(empty_spaces)
        empty_areas = [np.sum(labeled == i) for i in range(1, num_features + 1)]
        return empty_areas

    def get_fitness(self):
        total_pieces_area = sum(p.width * p.height for p in self.pieces)
        used_sheets = len(self.plates_used)
        sheet_area = self.plate_width * self.plate_height
        
        # Penalización por láminas extra y desperdicio
        wasted_area = (used_sheets * sheet_area) - total_pieces_area
        penalty = 0.5 * wasted_area + 10 * (used_sheets - np.ceil(total_pieces_area / sheet_area))
        
        self.fitness = -penalty  # Minimizar (mayor fitness es mejor)
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
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

        for i, (plate, _) in enumerate(self.plates_used):
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xlim(0, self.plate_width)
            ax.set_ylim(0, self.plate_height)
            ax.set_xlabel("Ancho")
            ax.set_ylabel("Alto")
            ax.set_title(f"Lámina {i+1}")
            ax.set_xticks(range(0, self.plate_width + 1, max(1, self.plate_width // 10)))
            ax.set_yticks(range(0, self.plate_height + 1, max(1, self.plate_height // 10)))
            ax.grid(True, linestyle="--", linewidth=0.5)

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

            plt.savefig(os.path.join(folder, f"lamina_{i+1}.png"))
            plt.close()

        print(f"imagenes generadas en la carpeta '{folder}'.")
