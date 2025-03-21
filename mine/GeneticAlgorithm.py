from Subject import Subject
from Piece import Piece
from copy import deepcopy
import random

class GeneticAlgorithm:
    
    def __init__(self, population_size, n_generations, base_plate, pieces, crossover_rate=0.8, mutation_rate=0.1):
        self.population_size = population_size  # Tamaño de la población
        self.n_generations = n_generations  # Número de generaciones
        self.base_plate = base_plate  # Lámina base (tupla con ancho y alto)
        self.generations = []  # Lista de generaciones con individuos
        self.pieces = pieces
        self.parents = []  # Lista de parejas de padres para la cruza
        self.crossover_rate = crossover_rate  # Probabilidad de cruzamiento
        self.mutation_rate = mutation_rate  # Probabilidad de mutación

    def set_first_generation(self):
        """Genera la primera generación con orden aleatorio de las piezas."""
        first_generation = []   

        for _ in range(self.population_size):
            shuffled_pieces = deepcopy(self.pieces)  # Copiar la lista para no modificar la original
            random.shuffle(shuffled_pieces)  # Mezclar la lista en su lugar
            print(shuffled_pieces)
            subject = Subject(120 , 120, shuffled_pieces)
            subject.place_pieces()  # Colocar en lámina base
            subject.get_fitness()  # Calcular fitness
            first_generation.append(subject)

        self.generations.append(first_generation)

    def select_parents(self):
        """Selecciona parejas de padres de la última generación usando un punto de corte aleatorio y las almacena en self.parents."""
        if not self.generations:
            print("⚠️ No hay generaciones previas para seleccionar padres.")
            return

        population = self.generations[-1]  # Última generación
        self.parents = []  # Resetear lista de padres

        sorted_population = sorted(population, key=lambda ind: ind.fitness, reverse=True)

        for i in range(0, len(sorted_population) - 1, 2):
            parent1 = sorted_population[i]
            parent2 = sorted_population[i + 1]

            cut_point = random.randint(1, len(parent1.pieces) - 1)  # Punto de corte aleatorio

            # Dividir los padres en dos partes
            parent1_part1 = parent1.pieces[:cut_point]
            parent1_part2 = parent1.pieces[cut_point:]

            parent2_part1 = parent2.pieces[:cut_point]
            parent2_part2 = parent2.pieces[cut_point:]

            self.parents.append(((parent1_part1, parent1_part2), (parent2_part1, parent2_part2)))
    
    def crossover(self):
        """Realiza la cruza con todas las parejas en self.parents según la probabilidad de cruce, generando una nueva generación."""
        if not self.parents:
            print("⚠️ No hay parejas de padres seleccionadas para la cruza.")
            return []

        new_generation = []

        x, y = self.base_plate
        for parent1_parts, parent2_parts in self.parents:
            if random.random() < self.crossover_rate:  # Aplicar cruce solo si se cumple la probabilidad
                (p1_part1, p1_part2), (p2_part1, p2_part2) = parent1_parts, parent2_parts

                # Crear nuevos hijos intercambiando las partes
                child1_pieces = p1_part1 + p2_part2
                child2_pieces = p2_part1 + p1_part2

                # Crear nuevos individuos con las piezas mezcladas
                child1 = Subject(x, y, child1_pieces)
                child2 = Subject(x, y, child2_pieces)

                child1.place_pieces()
                child2.place_pieces()

                new_generation.extend([child1, child2])

        self.generations.append(new_generation)

    def mutate(self):
        """Muta las piezas de la población rotándolas con una probabilidad dada."""
        for subject in self.generations[-1]:
            for piece in subject.pieces:
                if random.random() < self.mutation_rate:
                    piece.width, piece.height = piece.height, piece.width  # Rotar la pieza
                    piece.rotated = not piece.rotated
                    subject.place_pieces()

    def prune(self):
        """Reduce la población al tamaño original manteniendo los mejores individuos."""
        sorted_subjects = sorted(self.generations[-1], key=lambda ind: ind.fitness, reverse=True)  # Ordenar por fitness
        self.generations[-1] = sorted_subjects[:self.population_size]  # Mantener los mejores

    

lamina_base = (120, 120)

# Lista de piezas (ejemplo)
piezas = [Piece(50, 50) for _ in range(10)] + [Piece(20, 20) for _ in range(10)] +  [Piece(30, 40) for _ in range(10)]

# Crear algoritmo genético con 10 individuos y 50 generaciones
ga = GeneticAlgorithm(population_size=20, n_generations=50, base_plate=lamina_base, pieces=piezas)

# Generar primera generación
ga.set_first_generation()

# Ver la primera generación
# for i, subject in enumerate(ga.generations[0]):
#     print(f"Individuo {i+1}: Fitness = {subject.fitness:.5f}")
#     subject.view_plates_distribution()
#     if (i == len(ga.generations)):
#         subject.generate_images()

ga.select_parents()
print("Parejas formadas")
ga.crossover()
print("cruza realizada")
ga.mutate()
ga.prune()

ga.generations[-1][0].generate_images()
print(ga.generations[-1][0].fitness)
print(len(ga.generations))

          
