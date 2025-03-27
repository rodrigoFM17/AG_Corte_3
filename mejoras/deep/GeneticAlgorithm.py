from Subject import Subject
from Piece import Piece
from copy import deepcopy
import copy
import random

class GeneticAlgorithm:
    
    def __init__(self, population_size, n_generations, base_plate, pieces, crossover_rate=0.8, mutation_rate=0.1):
        self.population_size = population_size
        self.n_generations = n_generations
        self.base_plate = base_plate
        self.generations = []
        self.pieces = pieces
        self.parents = []
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def set_first_generation(self):
        first_generation = []
        for _ in range(self.population_size):
            pieces = deepcopy(self.pieces)
            # Asignación aleatoria de rotación y lámina
            for p in pieces:
                p.rotated = random.random() < 0.5
                p.plate = random.randint(1, 3)  # Láminas iniciales estimadas
            random.shuffle(pieces)
            subject = Subject(120, 120, pieces)
            subject.place_pieces()
            first_generation.append(subject)
        self.generations.append(first_generation)

    def get_last_generation(self):
        return self.generations[-1]

    def select_parents(self, tournament_size=3):
        if not self.generations:
            print("No hay generaciones previas para seleccionar padres.")
            return

        population = self.generations[-1]
        parents = []
        
        # Seleccionar parejas de padres mediante torneos
        for _ in range(self.population_size // 2):  # Generar N/2 parejas
            # Torneo para padre 1
            candidates = random.sample(population, tournament_size)
            parent1 = max(candidates, key=lambda ind: ind.fitness)
            
            # Torneo para padre 2 (excluyendo a parent1 para mayor diversidad)
            candidates = random.sample([ind for ind in population if ind != parent1], tournament_size)
            parent2 = max(candidates, key=lambda ind: ind.fitness)
            
            parents.append((parent1, parent2))

        self.parents = parents
        print(f"Se seleccionaron {len(parents)} parejas de padres.")
    
    def crossover(self):
        new_generation = []
        
        for parent1, parent2 in self.parents:  # Itera sobre parejas
            if random.random() > self.crossover_rate:
                # Sin cruce: añadir padres directamente
                new_generation.extend([deepcopy(parent1), deepcopy(parent2)])
                continue
            
            # Realizar cruce
            child1, child2 = self._perform_crossover(parent1, parent2)
            new_generation.extend([child1, child2])
        
        self.generations.append(new_generation)

    def _perform_crossover(self, parent1, parent2):
        child1 = deepcopy(parent1)
        child2 = deepcopy(parent2)
        
        # Cruzamiento de orden (OX)
        cut_point = random.randint(1, len(child1.pieces) - 1)
        child1.pieces[cut_point:], child2.pieces[cut_point:] = (
            child2.pieces[cut_point:], 
            child1.pieces[cut_point:]
        )
        
        return child1, child2

    def mutate(self):
        for subject in self.generations[-1]:
            for piece in subject.pieces:
                # Mutación de rotación
                if random.random() < self.mutation_rate:
                    piece.rotate()
                # Mutación de lámina
                if random.random() < self.mutation_rate:
                    piece.plate += random.choice([-1, 1])
                    piece.plate = max(1, piece.plate)
            subject.place_pieces()
                    
    def prune(self):
        population = self.generations[-1]
        population_size = self.population_size
        elite_count = max(1, int(0.2 * population_size))
        random_count = population_size - elite_count
        sorted_population = sorted(population, key=lambda ind: ind.fitness, reverse=True)
        elite_individuals = sorted_population[:elite_count]
        random_individuals = random.sample(population, random_count)
        self.generations[-1] = elite_individuals + random_individuals

    def start(self):
        self.set_first_generation()

        for _ in range(self.n_generations):
            self.select_parents()
            self.crossover()
            self.mutate()
            self.prune()

    def get_best_subject(self):
        sortedG = sorted(self.generations[-1], key=lambda piece : piece.fitness, reverse=True)
        return sortedG[0]

lamina_base = (120, 120)
piezas = [Piece(50, 50) for _ in range(10)] + [Piece(20, 20) for _ in range(10)] + [Piece(30, 40) for _ in range(10)]

ga = GeneticAlgorithm(population_size=20, n_generations=20, base_plate=lamina_base, pieces=piezas, crossover_rate=0.6, mutation_rate=0.3)
ga.start()

best_subject = ga.get_best_subject()
best_subject.generate_images()

print(len(ga.generations))
print(f"fitness: {best_subject.fitness}")