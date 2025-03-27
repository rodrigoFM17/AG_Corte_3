class Piece:
    def __init__(self, width, height, id=None):
        self.x = 0  
        self.y = 0  
        self.width = width
        self.height = height
        self.rotated = False 
        self.plate = 0
        self.id = id  # Identificador único
        # Nuevos atributos para evolución
        self.placement_strategy = None  # Estrategia de colocación para esta pieza

    def rotate(self):
        """Rota la pieza intercambiando ancho y alto"""
        self.width, self.height = self.height, self.width
        self.rotated = not self.rotated
        return self

    def __repr__(self):
        rotation = "R" if self.rotated else ""
        return f"P{self.id}{rotation}({self.width}x{self.height}) at ({self.x},{self.y})"


import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil
import numpy as np
from scipy.ndimage import label
import random

class PlacementStrategy:
    """Clase que representa diferentes estrategias de colocación de piezas"""
    BOTTOM_LEFT = 0  # Esquina inferior izquierda
    TOP_LEFT = 1     # Esquina superior izquierda
    BEST_FIT = 2     # Mejor ajuste (menor desperdicio)
    RANDOM_FIT = 3   # Ajuste aleatorio en un espacio válido
    
    @staticmethod
    def get_random_strategy():
        """Retorna una estrategia aleatoria"""
        return random.randint(0, 3)
    
    @staticmethod
    def get_name(strategy):
        names = {
            0: "Bottom-Left",
            1: "Top-Left",
            2: "Best-Fit",
            3: "Random-Fit"
        }
        return names.get(strategy, "Unknown")


class Subject:

    def __init__(self, plate_width, plate_height, pieces):
        self.plates_used = []
        self.pieces = pieces
        self.fitness = 0
        self.plate_width = plate_width
        self.plate_height = plate_height
        self.plates_matrix = []
        self.empty_areas = []
        
        # Calcular el área total de las piezas para establecer un número razonable de láminas
        total_pieces_area = sum(piece.width * piece.height for piece in pieces)
        theoretical_min_plates = max(1, int(np.ceil(total_pieces_area / (plate_width * plate_height))))
        
        # Nuevos atributos para evolución avanzada
        self.global_strategies = {
            # Limitar el número de láminas entre el mínimo teórico y un valor razonable
            'max_plates': random.randint(theoretical_min_plates, 
                                        max(theoretical_min_plates + 3, int(theoretical_min_plates * 1.5))),
            'default_placement': PlacementStrategy.get_random_strategy(),  # Estrategia por defecto
            'rotation_policy': random.random(),  # Probabilidad de rotación automática
            'grouping_similar': random.random() > 0.5  # Si agrupar piezas similares
        }
        
        # Asignar estrategias individuales a cada pieza
        for i, piece in enumerate(self.pieces):
            if piece.id is None:
                piece.id = i
            # Cada pieza puede tener su propia estrategia o usar la global
            piece.placement_strategy = PlacementStrategy.get_random_strategy()

    def set_pieces(self, pieces):
        self.pieces = pieces
    
    def place_pieces(self):
        """Método principal de colocación que utiliza estrategias evolutivas"""
        # Reiniciar arreglos de resultados
        self.plates_used = []
        self.empty_areas = []
        
        # Crear una copia de trabajo de las piezas para no modificar la original
        pieces_to_place = self.pieces.copy()
        
        # Ordenar piezas según estrategia global si es necesario
        if self.global_strategies['grouping_similar']:
            pieces_to_place.sort(key=lambda p: p.width * p.height, reverse=True)
        
        # Aplicar política de rotación global
        for piece in pieces_to_place:
            if random.random() < self.global_strategies['rotation_policy']:
                if piece.width < piece.height:  # Intentar orientación horizontal si es más alta que ancha
                    piece.rotate()
        
        # Inicializar la primera lámina
        current_plate_pieces = []
        plate_matrix = np.zeros((self.plate_height, self.plate_width), dtype=int)
        current_plate_idx = 0
        
        # Obtener el número máximo de láminas permitidas
        max_plates = self.global_strategies['max_plates']
        
        # Contador de piezas colocadas
        placed_count = 0
        
        # Intentar colocar cada pieza
        for piece in pieces_to_place:
            # Verificar si ya usamos todas las láminas permitidas
            if current_plate_idx >= max_plates:
                break
                
            # Decidir estrategia para esta pieza (propia o global)
            strategy = piece.placement_strategy if piece.placement_strategy is not None else self.global_strategies['default_placement']
            
            # Intentar colocar según la estrategia
            placed = self._place_piece_with_strategy(piece, plate_matrix, strategy)
            
            # Si no se pudo colocar y podemos usar una nueva lámina
            if not placed:
                if current_plate_pieces:  # Guardar lámina actual si tiene piezas
                    self.plates_used.append((current_plate_pieces.copy(), plate_matrix.copy()))
                    self.empty_areas.append(self.calculate_empty_areas(plate_matrix))
                
                # Verificar si podemos usar una nueva lámina
                current_plate_idx += 1
                if current_plate_idx >= max_plates:
                    # No podemos usar más láminas, esta pieza no se colocará
                    continue
                    
                # Crear nueva lámina
                plate_matrix = np.zeros((self.plate_height, self.plate_width), dtype=int)
                current_plate_pieces = []
                
                # Colocar en la nueva lámina
                if self._place_piece_with_strategy(piece, plate_matrix, strategy):
                    piece.plate = current_plate_idx
                    current_plate_pieces.append(piece)
                    placed_count += 1
            else:
                # La pieza se colocó correctamente
                piece.plate = current_plate_idx
                current_plate_pieces.append(piece)
                placed_count += 1
        
        # Agregar la última lámina si tiene piezas
        if current_plate_pieces:
            self.plates_used.append((current_plate_pieces.copy(), plate_matrix.copy()))
            self.empty_areas.append(self.calculate_empty_areas(plate_matrix))
        
        # Verificar que no tengamos láminas vacías
        self.plates_used = [(pieces, matrix) for pieces, matrix in self.plates_used if pieces]
        
        # Recalcular áreas vacías si fue necesario
        if len(self.empty_areas) != len(self.plates_used):
            self.empty_areas = []
            for _, matrix in self.plates_used:
                self.empty_areas.append(self.calculate_empty_areas(matrix))
            
        # Calcular fitness
        self.get_fitness()
        
        # Verificación final de consistencia
        total_placed = sum(len(plate[0]) for plate in self.plates_used)
        if total_placed != placed_count:
            print(f"ADVERTENCIA: Inconsistencia en el conteo de piezas: {total_placed} vs {placed_count}")
    
    def _place_piece_with_strategy(self, piece, plate_matrix, strategy):
        """Coloca una pieza según la estrategia específica"""
        if strategy == PlacementStrategy.BOTTOM_LEFT:
            return self._place_bottom_left(piece, plate_matrix)
        elif strategy == PlacementStrategy.TOP_LEFT:
            return self._place_top_left(piece, plate_matrix)
        elif strategy == PlacementStrategy.BEST_FIT:
            return self._place_best_fit(piece, plate_matrix)
        elif strategy == PlacementStrategy.RANDOM_FIT:
            return self._place_random_fit(piece, plate_matrix)
        else:
            return self._place_bottom_left(piece, plate_matrix)  # Por defecto
    
    def _place_bottom_left(self, piece, plate_matrix):
        """Coloca la pieza en la posición más abajo y a la izquierda posible"""
        for y in range(self.plate_height - piece.height + 1):
            for x in range(self.plate_width - piece.width + 1):
                if np.all(plate_matrix[y:y + piece.height, x:x + piece.width] == 0):
                    piece.x, piece.y = x, y
                    plate_matrix[y:y + piece.height, x:x + piece.width] = 1
                    return True
        return False
    
    def _place_top_left(self, piece, plate_matrix):
        """Coloca la pieza desde arriba hacia abajo, de izquierda a derecha"""
        for y in range(self.plate_height - 1, -1, -1):
            if y + piece.height > self.plate_height:
                continue
            for x in range(self.plate_width - piece.width + 1):
                if np.all(plate_matrix[y:y + piece.height, x:x + piece.width] == 0):
                    piece.x, piece.y = x, y
                    plate_matrix[y:y + piece.height, x:x + piece.width] = 1
                    return True
        return False
    
    def _place_best_fit(self, piece, plate_matrix):
        """Busca la posición que minimice el desperdicio"""
        best_x, best_y = -1, -1
        best_waste = float('inf')
        
        for y in range(self.plate_height - piece.height + 1):
            for x in range(self.plate_width - piece.width + 1):
                if np.all(plate_matrix[y:y + piece.height, x:x + piece.width] == 0):
                    # Calcular desperdicio simulando la colocación
                    temp_matrix = plate_matrix.copy()
                    temp_matrix[y:y + piece.height, x:x + piece.width] = 1
                    
                    # Contar el número de celdas vacías adyacentes
                    waste = 0
                    for i in range(max(0, y-1), min(self.plate_height, y+piece.height+1)):
                        for j in range(max(0, x-1), min(self.plate_width, x+piece.width+1)):
                            if temp_matrix[i, j] == 0:
                                waste += 1
                    
                    if waste < best_waste:
                        best_waste = waste
                        best_x, best_y = x, y
        
        if best_x != -1:
            piece.x, piece.y = best_x, best_y
            plate_matrix[best_y:best_y + piece.height, best_x:best_x + piece.width] = 1
            return True
        return False
    
    def _place_random_fit(self, piece, plate_matrix):
        """Coloca la pieza en una posición aleatoria válida"""
        valid_positions = []
        
        for y in range(self.plate_height - piece.height + 1):
            for x in range(self.plate_width - piece.width + 1):
                if np.all(plate_matrix[y:y + piece.height, x:x + piece.width] == 0):
                    valid_positions.append((x, y))
        
        if valid_positions:
            x, y = random.choice(valid_positions)
            piece.x, piece.y = x, y
            plate_matrix[y:y + piece.height, x:x + piece.width] = 1
            return True
        return False
    
    def calculate_strategy_stats(self):
        """Calcula estadísticas sobre las estrategias utilizadas"""
        stats = {
            'total_pieces': len(self.pieces),
            'placed_pieces': sum(len(plate[0]) for plate in self.plates_used),
            'strategies': {
                'global': PlacementStrategy.get_name(self.global_strategies['default_placement']),
                'piece_specific': sum(1 for p in self.pieces if p.placement_strategy is not None),
                'rotation_policy': f"{self.global_strategies['rotation_policy']:.2f}",
                'max_plates': self.global_strategies['max_plates'],
                'grouping': str(self.global_strategies['grouping_similar'])
            }
        }
        return stats

    def calculate_empty_areas(self, plate_matrix):
        """Calcula las áreas vacías en una lámina"""
        empty_spaces = (plate_matrix == 0).astype(int)
        labeled, num_features = label(empty_spaces)
        empty_areas = [np.sum(labeled == i) for i in range(1, num_features + 1)]
        return empty_areas

    def get_fitness(self):
        """Calcula el fitness del individuo considerando múltiples factores"""
        # Si no hay láminas usadas, el fitness es 0
        if not self.plates_used:
            self.fitness = 0
            return 0
            
        # Área total disponible
        total_area = len(self.plates_used) * (self.plate_width * self.plate_height)
        
        # Área total de piezas
        total_pieces_area = sum(piece.width * piece.height for piece in self.pieces)
        
        # Piezas colocadas (conteo exacto)
        placed_pieces = sum(len(plate[0]) for plate in self.plates_used)
        unplaced_penalty = (len(self.pieces) - placed_pieces) * 0.2
        
        # Área utilizada efectivamente
        used_area = 0
        for plate, _ in self.plates_used:
            for piece in plate:
                used_area += piece.width * piece.height
        
        # Número teórico de láminas
        theoretical_sheets = max(1, np.ceil(total_pieces_area / (self.plate_width * self.plate_height)))
        
        # Número de láminas utilizadas
        used_sheets = len(self.plates_used)
        
        # Calcular compactación (minimizar espacios vacíos dispersos)
        compactness = 0
        for areas in self.empty_areas:
            # Penalizar muchos espacios pequeños vs. pocos espacios grandes
            if areas:  # Verificar que no esté vacío
                compactness += len(areas) * 0.01
        
        # Penalización por exceso de láminas
        sheets_penalty = max(0, (used_sheets - theoretical_sheets)) * 0.15
        
        # Eficiencia de uso (porcentaje de área utilizada)
        efficiency = used_area / total_area if total_area > 0 else 0
        
        # Fitness final combinando factores (con valores de debug)
        raw_fitness = efficiency - sheets_penalty - compactness - unplaced_penalty
        
        # Para evitar fitness negativo, establecer un mínimo
        fitness = max(0.0001, raw_fitness)
        
        # Bonificación por usar menos láminas que las teóricas (muy raro pero posible)
        if used_sheets < theoretical_sheets:
            fitness += 0.5
        
        # Debug info detallado
        print(f"DEBUG FITNESS:")
        print(f"  - Eficiencia: {efficiency:.4f}")
        print(f"  - Láminas: {used_sheets}/{theoretical_sheets} (Penalización: {sheets_penalty:.4f})")
        print(f"  - Piezas: {placed_pieces}/{len(self.pieces)} (Penalización: {unplaced_penalty:.4f})")
        print(f"  - Compactación: {compactness:.4f}")
        print(f"  - FITNESS TOTAL: {fitness:.4f}")
        
        # Guardar fitness
        self.fitness = fitness
        
        return fitness

    def __repr__(self):
        result = f"Individuo (Fitness: {self.fitness:.4f})\n"
        result += f"Estrategia global: {PlacementStrategy.get_name(self.global_strategies['default_placement'])}\n"
        result += f"Política rotación: {self.global_strategies['rotation_policy']:.2f}, Agrupación: {self.global_strategies['grouping_similar']}\n"
        result += f"Máx láminas: {self.global_strategies['max_plates']}\n"
        
        for i, (plate, _) in enumerate(self.plates_used):
            result += f"Lámina {i+1} ({len(plate)} piezas):\n"
            for piece in plate:
                strategy = "Global" if piece.placement_strategy is None else PlacementStrategy.get_name(piece.placement_strategy)
                result += f"  {piece} [Estrategia: {strategy}]\n"
            result += "-" * 40 + "\n"
        return result
    
    def generate_images(self, folder="laminas"):
        """Genera imágenes de las láminas con información de estrategias"""
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

        # Colores según estrategia de colocación
        strategy_colors = {
            None: "lightblue",  # Global (default)
            PlacementStrategy.BOTTOM_LEFT: "lightgreen",
            PlacementStrategy.TOP_LEFT: "lightcoral",
            PlacementStrategy.BEST_FIT: "lightyellow",
            PlacementStrategy.RANDOM_FIT: "plum"
        }

        for i, (plate, _) in enumerate(self.plates_used):
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.set_xlim(0, self.plate_width)
            ax.set_ylim(0, self.plate_height)
            ax.set_xlabel("Ancho")
            ax.set_ylabel("Alto")
            
            # Título con información de la lámina
            title = f"Lámina {i+1} - {len(plate)} piezas"
            ax.set_title(title)
            
            # Información de estrategias
            strategies_text = f"Estrategia global: {PlacementStrategy.get_name(self.global_strategies['default_placement'])}"
            plt.figtext(0.5, 0.01, strategies_text, ha="center")
            
            ax.set_xticks(range(0, self.plate_width + 1, max(1, self.plate_width // 10)))
            ax.set_yticks(range(0, self.plate_height + 1, max(1, self.plate_height // 10)))
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

            # Dibujar piezas
            for piece in plate:
                # Color según estrategia
                color = strategy_colors.get(piece.placement_strategy, strategy_colors[None])
                
                rect = patches.Rectangle(
                    (piece.x, self.plate_height - piece.y - piece.height),
                    piece.width,
                    piece.height,
                    linewidth=1,
                    edgecolor="black",
                    facecolor=color,
                    alpha=0.8
                )
                ax.add_patch(rect)
                
                # Texto con información de la pieza
                strategy_name = "G" if piece.placement_strategy is None else piece.placement_strategy
                text = f"{piece.width}x{piece.height}\nID:{piece.id}"
                ax.text(
                    piece.x + piece.width / 2,
                    self.plate_height - piece.y - piece.height / 2,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="black",
                    bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.3", alpha=0.7)
                )

            # Leyenda para estrategias
            legend_elements = [
                patches.Patch(facecolor=strategy_colors[None], edgecolor="black", label="Global"),
                patches.Patch(facecolor=strategy_colors[PlacementStrategy.BOTTOM_LEFT], edgecolor="black", label="Bottom-Left"),
                patches.Patch(facecolor=strategy_colors[PlacementStrategy.TOP_LEFT], edgecolor="black", label="Top-Left"),
                patches.Patch(facecolor=strategy_colors[PlacementStrategy.BEST_FIT], edgecolor="black", label="Best-Fit"),
                patches.Patch(facecolor=strategy_colors[PlacementStrategy.RANDOM_FIT], edgecolor="black", label="Random-Fit")
            ]
            ax.legend(handles=legend_elements, title="Estrategias", loc="upper right")

            plt.tight_layout()
            plt.savefig(os.path.join(folder, f"lamina_{i+1}.png"), dpi=150)
            plt.close()

        # Generar resumen
        self._generate_summary_image(folder)
        
        print(f"Imágenes generadas en la carpeta '{folder}'.")
        
    def _generate_summary_image(self, folder):
        """Genera una imagen de resumen con estadísticas del individuo"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        
        # Título
        plt.figtext(0.5, 0.95, f"Resumen del Individuo (Fitness: {self.fitness:.4f})", ha="center", fontsize=14, fontweight="bold")
        
        # Estadísticas generales
        stats = [
            f"Número de láminas: {len(self.plates_used)}",
            f"Total de piezas: {len(self.pieces)}",
            f"Piezas colocadas: {sum(len(plate[0]) for plate in self.plates_used)}",
            f"Estrategia global: {PlacementStrategy.get_name(self.global_strategies['default_placement'])}",
            f"Política de rotación: {self.global_strategies['rotation_policy']:.2f}",
            f"Agrupación por tamaño: {self.global_strategies['grouping_similar']}",
            f"Máximo de láminas: {self.global_strategies['max_plates']}"
        ]
        
        y_pos = 0.85
        for stat in stats:
            plt.figtext(0.1, y_pos, stat, fontsize=12)
            y_pos -= 0.05
        
        # Estadísticas por lámina
        plt.figtext(0.5, 0.6, "Estadísticas por Lámina", ha="center", fontsize=12, fontweight="bold")
        
        y_pos = 0.55
        for i, (plate, _) in enumerate(self.plates_used):
            used_area = sum(p.width * p.height for p in plate)
            total_area = self.plate_width * self.plate_height
            efficiency = (used_area / total_area) * 100
            
            plt.figtext(0.1, y_pos, f"Lámina {i+1}: {len(plate)} piezas, Eficiencia: {efficiency:.1f}%", fontsize=10)
            y_pos -= 0.04
        
        plt.tight_layout()
        plt.savefig(os.path.join(folder, "resumen.png"), dpi=150)
        plt.close()