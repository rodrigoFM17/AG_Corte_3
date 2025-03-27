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
            shuffled_pieces = deepcopy(self.pieces)
            random.shuffle(shuffled_pieces)
            subject = Subject(120, 120, shuffled_pieces)
            subject.place_pieces()
            subject.get_fitness()
            first_generation.append(subject)

        self.generations.append(first_generation)
        print(f"primera generacion {len(self.generations[-1])}")

    def get_last_generation(self):
        return self.generations[-1]

    def select_parents(self):
        if not self.generations:
            print("No hay generaciones previas para seleccionar padres.")
            return

        population = self.generations[-1]
        parents = []
        sorted_population = sorted(population, key=lambda ind: ind.fitness, reverse=True)

        for i in range(len(sorted_population)):
            for j in range(i, len(sorted_population)):  # Empieza en i para incluir (i, i)
                parent1 = sorted_population[i]
                parent2 = sorted_population[j]
                parents.append((parent1, parent2))

        self.parents = parents
        print(f"Se seleccionaron {len(parents)} parejas de padres.")

    
    def advanced_crossover(self):
        """
        Cruzamiento de orden parcial (PMX) con consideraciones de rotación
        """
        for couple in self.parents: 
            if random.random() > self.crossover_rate:
                return
            
            parent1, parent2 = couple

            # Crear copias de los padres
            child1 = parent1.clone()
            child2 = parent2.clone()

            # Seleccionar punto de cruce
            crossover_point = random.randint(0, len(parent1.pieces) - 1)

            # Mapear piezas de un padre al otro
            mapping1 = {p.id: p for p in parent1.pieces}
            mapping2 = {p.id: p for p in parent2.pieces}

            # Intercambiar segmentos
            child1.pieces[:crossover_point], child2.pieces[:crossover_point] = \
                parent2.pieces[:crossover_point], parent1.pieces[:crossover_point]

            # Completar con piezas restantes manteniendo características
            for i in range(crossover_point, len(child1.pieces)):
                if child1.pieces[i].id in mapping1:
                    child1.pieces[i] = mapping1[child1.pieces[i].id].clone()
                if child2.pieces[i].id in mapping2:
                    child2.pieces[i] = mapping2[child2.pieces[i].id].clone()

            self.generations[-1].append(child1)
            self.generations[-1].append(child2)

    def advanced_mutation(self):
        """
        Mutación con múltiples estrategias
        """
        # Probabilidad de mutación a nivel de sujeto
        for subject in self.generations[-1]:
            if random.random() < self.mutation_rate:
                # Diferentes tipos de mutación
                mutation_type = random.choice([
                    'swap_pieces',     # Intercambiar piezas
                    'rotate_piece',    # Rotar una pieza
                    'partial_shuffle' # Barajear un subconjunto
                ])

                if mutation_type == 'swap_pieces':
                    # Intercambiar dos piezas aleatorias
                    i, j = random.sample(range(len(subject.pieces)), 2)
                    subject.pieces[i], subject.pieces[j] = subject.pieces[j], subject.pieces[i]
                
                elif mutation_type == 'rotate_piece':
                    # Rotar una pieza aleatoria
                    piece = random.choice(subject.pieces)
                    piece.rotate()
                
                elif mutation_type == 'partial_shuffle':
                    # Barajear un subconjunto de piezas
                    start = random.randint(0, len(subject.pieces) // 2)
                    end = random.randint(start + 1, len(subject.pieces))
                    subset = subject.pieces[start:end]
                    random.shuffle(subset)
                    subject.pieces[start:end] = subset

                subject.get_fitness()
                    
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
            self.advanced_crossover()
            self.advanced_mutation()
            self.prune()

    def get_best_subject(self):
        sortedG = sorted(self.generations[-1], key=lambda piece : piece.fitness, reverse=True)
        return sortedG[0]

lamina_base = (120, 120)
piezas = [Piece(50, 50) for _ in range(10)] + [Piece(20, 20) for _ in range(10)] + [Piece(30, 40) for _ in range(10)]

ga = GeneticAlgorithm(population_size=20, n_generations=20, base_plate=lamina_base, pieces=piezas, crossover_rate=0.6, mutation_rate=0.5)
ga.start()

best_subject = ga.get_best_subject()
best_subject.generate_images()

print(len(ga.generations))
print(f"fitness: {best_subject.fitness}")