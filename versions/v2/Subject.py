from Piece import Piece
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil

class Subject:
    def __init__(self, plate_width, plate_height, pieces):
        self.plates_used = []  # Lista de láminas utilizadas
        self.pieces = pieces  # Lista de piezas asignadas
        self.fitness = 0
        self.plate_width = plate_width
        self.plate_height = plate_height

    def set_pieces(self, pieces):
        self.pieces = pieces

    def place_pieces(self):
        """
        Coloca las piezas en el orden que aparecen, sin solapamientos.
        """
        self.plates_used = [[]]  # Primera lámina

        x_actual = 0
        y_actual = 0
        altura_fila = 0

        for piece in self.pieces:
            pw, ph = piece.width, piece.height

            # Si la pieza no cabe en X, mover a la siguiente fila
            if x_actual + pw > self.plate_width:
                x_actual = 0
                y_actual += altura_fila  # Mover hacia abajo
                altura_fila = 0  # Reset de altura de fila

            # Si la pieza no cabe en Y, usar nueva lámina
            if y_actual + ph > self.plate_height:
                self.plates_used.append([])  # Nueva lámina
                x_actual = 0
                y_actual = 0
                altura_fila = 0

            # Colocar la pieza en la posición actual
            piece.x = x_actual
            piece.y = y_actual
            piece.plate = len(self.plates_used) - 1  # Índice de la lámina

            # self.pieces.append(piece)
            self.plates_used[-1].append(piece)

            # Actualizar la posición para la siguiente pieza
            x_actual += pw
            altura_fila = max(altura_fila, ph)  # Mantener altura máxima de fila
        self.get_fitness()

    def __repr__(self):
        result = ""
        for i, plate in enumerate(self.plates_used):
            result += f"Lámina {i+1}:\n"
            for piece in plate:
                result += f"{piece}\n"
            result += "-" * 40 + "\n"
        return result

    def get_fitness(self):
        """Calcula el fitness como la sumatoria del aprovechamiento de cada lámina."""
        total_fitness = 0  # Inicializar fitness
        area_lamina = self.plate_width * self.plate_height  # Área total de una lámina

        for lamina in self.plates_used:
            area_usada = sum(pieza.width * pieza.height for pieza in lamina)  # Sumatoria de áreas de piezas
            total_fitness += area_usada / area_lamina # Agregar el aprovechamiento de esta lámina

        self.fitness = total_fitness / len(self.plates_used) # Guardar fitness
        return self.fitness
    
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