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
            print("no hay generaciones previas para seleccionar padres.")
            return

        population = self.generations[-1]
        parents = []
        sorted_population = sorted(population, key=lambda ind: ind.fitness, reverse=True)

        for i in range(0, len(sorted_population) - 1, 2):
            parent1 = sorted_population[i]
            parent2 = sorted_population[i + 1]
            parents.append((parent1, parent2))

        self.parents = parents
        print(f" se seleccionaron {len(parents)} parejas de padres.")
    
    def crossover(self, crossover_prob=0.8):
        new_generation = []

        for parent1, parent2 in self.parents:
            if random.random() > crossover_prob:
                new_generation.extend([copy.deepcopy(parent1), copy.deepcopy(parent2)])
                continue

            child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
            piece1 = random.choice(child1.pieces)  
            piece2 = random.choice(child2.pieces)
            index1, index2 = child1.pieces.index(piece1), child2.pieces.index(piece2)
            child1.pieces[index1], child2.pieces[index2] = child2.pieces[index2], child1.pieces[index1]

            piece1_match = next((p for p in child1.pieces if p.width == piece2.width and p.height == piece2.height and p != piece1), None)
            piece2_match = next((p for p in child2.pieces if p.width == piece1.width and p.height == piece1.height and p != piece2), None)

            if piece1_match and piece2_match:
                index1_match, index2_match = child1.pieces.index(piece1_match), child2.pieces.index(piece2_match)
                child1.pieces[index1_match], child2.pieces[index2_match] = child2.pieces[index2_match], child1.pieces[index1_match]

            new_generation.extend([child1, child2])

        self.generations.append(new_generation)

    def mutate(self):
        for subject in self.generations[-1]:
            if random.random() < self.mutation_rate:
                for piece in subject.pieces:
                    if random.random() < self.mutation_rate:
                        piece.width, piece.height = piece.height, piece.width
                        piece.rotated = not piece.rotated
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

ga = GeneticAlgorithm(population_size=20, n_generations=20, base_plate=lamina_base, pieces=piezas, crossover_rate=0.6, mutation_rate=0.5)
ga.start()

best_subject = ga.get_best_subject()
best_subject.generate_images()

print(len(ga.generations))
print(f"fitness: {best_subject.fitness}")