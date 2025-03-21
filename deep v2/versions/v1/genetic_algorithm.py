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

    def create_individual(self):
        perm = np.random.permutation(self.n_pieces).tolist()
        rot = [random.random() < 0.5 for _ in range(self.n_pieces)]
        return Individual(perm, rot)

    def initialize_population(self):
        return [self.create_individual() for _ in range(self.pop_size)]

    def calculate_fitness(self, individual: Individual):
        sheets = []
        current_sheet = []
        sheet_index = 0
        
        for idx in individual.permutation:
            l, w = self.pieces[idx]
            if individual.rotations[idx]:
                l, w = w, l
            
            placed = False
            for y in range(self.sheet_h - w + 1):
                for x in range(self.sheet_w - l + 1):
                    if not self.check_overlap(current_sheet, x, y, l, w):
                        current_sheet.append((x, y, l, w))
                        individual.layout.append((sheet_index, x, y, l, w))
                        placed = True
                        break
                if placed: break
            
            if not placed:
                sheets.append(current_sheet)
                current_sheet = [(0, 0, l, w)]
                individual.layout.append((sheet_index+1, 0, 0, l, w))
                sheet_index += 1
        
        sheets.append(current_sheet)
        used_sheets = len(sheets)
        total_area = sum(l*w for l, w in self.pieces)
        sheet_area = self.sheet_w * self.sheet_h
        min_sheets = math.ceil(total_area / sheet_area)
        
        efficiency = total_area / (used_sheets * sheet_area)
        penalty = self.penalty * max(0, used_sheets - min_sheets)
        individual.fitness = efficiency - penalty
        return individual.fitness

    def check_overlap(self, placed, x, y, l, w):
        for (px, py, pl, pw) in placed:
            if not (x >= px+pl or x+l <= px or y >= py+pw or y+w <= py):
                return True
        return False

    def select_parents(self, pop: List[Individual]):
        return random.choices(pop, weights=[i.fitness for i in pop], k=2)

    def crossover(self, p1: Individual, p2: Individual):
        size = self.n_pieces
        start, end = sorted(random.sample(range(size), 2))
        child_perm = p1.permutation[start:end] + [g for g in p2.permutation if g not in p1.permutation[start:end]]
        child_rot = [p1.rotations[i] if start <= i < end else p2.rotations[i] for i in range(size)]
        return Individual(child_perm, child_rot)

    def mutate(self, individual: Individual):
        if random.random() < self.mut_rate:
            i, j = random.sample(range(self.n_pieces), 2)
            individual.permutation[i], individual.permutation[j] = individual.permutation[j], individual.permutation[i]
        
        for i in range(self.n_pieces):
            if random.random() < self.mut_rate:
                individual.rotations[i] = not individual.rotations[i]
        return individual

    def evolve(self, pop: List[Individual]):
        new_pop = sorted(pop, reverse=True)[:self.elite]
        while len(new_pop) < self.pop_size:
            parents = self.select_parents(pop)
            child = self.crossover(parents[0], parents[1])
            child = self.mutate(child)
            new_pop.append(child)
        return new_pop