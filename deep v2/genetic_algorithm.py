# genetic_algorithm.py
import random
import math
import numpy as np
from typing import List, Tuple

class Individual:
    def __init__(self, permutation: List[int], rotations: List[bool]):
        self.permutation = permutation  # Orden de colocación
        self.rotations = rotations      # Rotación de cada pieza
        self.fitness = 0.0
        self.layout = []                # Almacena (hoja, x, y, l, a)

    def __lt__(self, other):
        return self.fitness < other.fitness

class GeneticAlgorithm:
    def __init__(self, pieces: List[Tuple[int, int]], sheet_size: Tuple[int, int], 
                 penalty: float, pop_size=100, elite=2, mut_rate=0.1):
        self.pieces = pieces
        self.sheet_w, self.sheet_h = sheet_size
        self.penalty = penalty
        self.pop_size = pop_size
        self.elite = elite
        self.mut_rate = mut_rate
        self.n_pieces = len(pieces)
        self.adaptive_mutation = True
        self.stagnation = 0
        self.last_best = 0

    def create_individual(self):
        perm = np.random.permutation(self.n_pieces).tolist()
        rot = [random.random() < 0.5 for _ in range(self.n_pieces)]
        return Individual(perm, rot)

    def initialize_population(self):
        return [self.create_individual() for _ in range(self.pop_size)]

    def calculate_fitness(self, individual: Individual):
        individual.layout = []
        sheets = []
        current_sheet = []
        sheet_area = self.sheet_w * self.sheet_h
        
        for idx in individual.permutation:
            original_l, original_w = self.pieces[idx]
            l, w = (original_w, original_l) if individual.rotations[idx] else (original_l, original_w)
            
            placed = False
            best_position = None
            best_remaining_area = float('inf')
            
            # Intentar colocar en todas las láminas existentes
            for sheet_idx, sheet in enumerate(sheets + [current_sheet]):
                for y in range(self.sheet_h - w + 1):
                    for x in range(self.sheet_w - l + 1):
                        if not self.check_overlap(sheet, x, y, l, w):
                            # Calcular espacio restante después de colocar
                            remaining_area = (self.sheet_w * self.sheet_h) - (len(sheet)+1)*(l*w)
                            if remaining_area < best_remaining_area:
                                best_position = (sheet_idx, x, y)
                                best_remaining_area = remaining_area
                                placed = True
                                break
                    if placed: break
                if placed: break
            
            if placed:
                target_sheet = sheets[best_position[0]] if best_position[0] < len(sheets) else current_sheet
                target_sheet.append((best_position[1], best_position[2], l, w))
                individual.layout.append((best_position[0], best_position[1], best_position[2], l, w))
            else:
                sheets.append(current_sheet)
                current_sheet = [(0, 0, l, w)]
                individual.layout.append((len(sheets), 0, 0, l, w))
        
        # Asegurar última lámina
        if current_sheet:
            sheets.append(current_sheet)
            
        # Cálculo real del área utilizada
        used_area = sum(l*w for sheet in sheets for (x, y, l, w) in sheet)
        total_sheet_area = len(sheets) * sheet_area
        
        # Cálculo teórico mínimo
        min_sheets = math.ceil(sum(l*w for l, w in self.pieces) / sheet_area)
        
        # Fitness ajustado
        efficiency = used_area / total_sheet_area if total_sheet_area > 0 else 0
        penalty = self.penalty * max(0, len(sheets) - min_sheets)
        
        individual.fitness = efficiency - penalty
        return individual.fitness

    def check_overlap(self, placed, x, y, l, w):
        for (px, py, pl, pw) in placed:
            if (x < px + pl and x + l > px and 
                y < py + pw and y + w > py):
                return True
        return False

    def crossover(self, p1: Individual, p2: Individual):
        # Cruza OX (Order Crossover)
        size = self.n_pieces
        start, end = sorted(random.sample(range(size), 2))
        child_perm = [-1]*size
        child_rot = [False]*size
        
        # Copiar segmento de p1
        child_perm[start:end] = p1.permutation[start:end]
        child_rot[start:end] = p1.rotations[start:end]
        
        # Completar con p2 manteniendo orden
        ptr = 0
        for i in range(size):
            if not (start <= i < end):
                while p2.permutation[ptr] in child_perm:
                    ptr += 1
                child_perm[i] = p2.permutation[ptr]
                child_rot[i] = p2.rotations[ptr]
                ptr += 1
        
        return Individual(child_perm, child_rot)

    def mutate(self, individual: Individual):
        # Mutación adaptativa
        mutation_rate = min(0.5, 1/(self.n_pieces**0.5))
        
        if random.random() < mutation_rate:
            # Mutación de inversión
            i, j = sorted(random.sample(range(self.n_pieces), 2))
            individual.permutation[i:j+1] = reversed(individual.permutation[i:j+1])
        
        # Mutación de rotación inteligente
        for i in range(self.n_pieces):
            if random.random() < mutation_rate:
                l, w = self.pieces[individual.permutation[i]]
                aspect_ratio = max(l/w, w/l)
                if aspect_ratio > 1.5:  # Solo rotar si es beneficioso
                    individual.rotations[i] = not individual.rotations[i]
        
        return individual
    
    def select_parents(self, pop: List[Individual]):
        # Torneo de tamaño 3
        candidates = random.sample(pop, 3)
        return [max(candidates, key=lambda x: x.fitness)]

    def evolve(self, pop: List[Individual]):
        current_best = max(pop, key=lambda x: x.fitness).fitness
        if abs(current_best - self.last_best) < 0.001:
            self.stagnation += 1
            if self.stagnation > 5:
                self.mut_rate = min(0.5, self.mut_rate * 1.5)
        else:
            self.stagnation = 0
            self.mut_rate = max(0.1, self.mut_rate * 0.95)
        
        self.last_best = current_best
        
        pop.sort(reverse=True, key=lambda x: x.fitness)
        new_pop = pop[:self.elite]  # Elitismo
        
        while len(new_pop) < self.pop_size:
            parents = [self.select_parents(pop)[0] for _ in range(2)]
            child = self.crossover(parents[0], parents[1])
            child = self.mutate(child)
            new_pop.append(child)
        
        return new_pop