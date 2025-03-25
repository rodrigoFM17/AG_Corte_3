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
        self.empty_areas = []
        plate_matrix = np.zeros((self.plate_height, self.plate_width), dtype=int)
        current_plate = []

        for piece in self.pieces:
            placed = False

            for y in range(self.plate_height - piece.height + 1):
                for x in range(self.plate_width - piece.width + 1):
                    if np.all(plate_matrix[y:y + piece.height, x:x + piece.width] == 0):
                        piece.x, piece.y = x, y
                        piece.plate = len(self.plates_used) + 1
                        current_plate.append(piece)
                        plate_matrix[y:y + piece.height, x:x + piece.width] = 1
                        placed = True
                        break
                if placed:
                    break

            if not placed:
                self.plates_used.append((current_plate, plate_matrix.copy()))
                self.empty_areas.append(self.calculate_empty_areas(plate_matrix))
                plate_matrix = np.zeros((self.plate_height, self.plate_width), dtype=int)
                current_plate = []
                piece.x, piece.y = 0, 0
                piece.plate = len(self.plates_used) + 1
                current_plate.append(piece)
                plate_matrix[:piece.height, :piece.width] = 1

        if current_plate:
            self.plates_used.append((current_plate, plate_matrix.copy()))
            self.empty_areas.append(self.calculate_empty_areas(plate_matrix))

        self.get_fitness()

    def calculate_empty_areas(self, plate_matrix):
        empty_spaces = (plate_matrix == 0).astype(int)
        labeled, num_features = label(empty_spaces)
        empty_areas = [np.sum(labeled == i) for i in range(1, num_features + 1)]
        return empty_areas

    def get_fitness(self):

        total_area = 0
        for _ in range (len(self.plates_used)) :
            total_area += self.plate_width * self.plate_height
        
        total_pieces_area = 0
        for piece in self.pieces:
            total_pieces_area += piece.width * piece.height

        theoretical_sheets = np.ceil(total_pieces_area / (self.plate_width * self.plate_height))
        used_sheets = len(self.plates_used)

        used_area = 0
        # used_area = sum(piece.width * piece.height for piece in self.pieces)

        fitness = 0
        for i in range(len(self.plates_used)):
            pieces_area = 0
            print(self.plates_used)
            current_plate, _ = self.plates_used[i]
            for piece in current_plate:
                pieces_area += piece.width * piece.height
                used_area += piece.width * piece.height
            
            wasted_area_per_plate = 0
            for wasted_area in self.empty_areas[i]:
                wasted_area_per_plate += wasted_area
            
            fitness += pieces_area / (self.plate_width * self.plate_height + wasted_area)

        penalty_for_excesive_sheets = (used_sheets - theoretical_sheets) * 0.1

        fitness = fitness / len(self.plates_used) - penalty_for_excesive_sheets
            
        # for  plate, _ in self.plates_used :
        #     used_area_per_plate = 0
        #     for piece in plate : 
        #         used_area_per_plate += piece.width * piece.height
        #     used_area += used_area_per_plate / ( self.plate_width * self.plate_height )

        # used_area = used_area / len(self.plates_used)

        

        # wasted_area = 0
        # for area_wasted_per_plate in self.empty_areas:
        #     for area in area_wasted_per_plate :
        #         wasted_area += area

        # wasted_area = wasted_area / ( self.plate_width * self.plate_height * len(self.plates_used) )
        
        print(used_area)

        # fitness = used_area  - (penalty_for_excesive_sheets + wasted_area)
        self.fitness = fitness 
        print("------------------------")
        print(f"fit: {fitness}")
        print(f"used_area: {used_area} \n" + 
              f"total_area: {total_area}\n" +
              f"theorical_sheets: {theoretical_sheets}\n"+
              f"used_plates: {used_sheets}\n"+
              f"penalty_for_excesive_plates: {penalty_for_excesive_sheets}\n"+ 
              f"wasted_area: {wasted_area}\n"
              f"empty_areas: {len(self.empty_areas)}\n")
        for plate in self.empty_areas:
            print(f"lamina: {plate}\n")
        return fitness

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
