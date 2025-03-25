from Subject import Subject
from Piece import Piece
from copy import deepcopy
import copy
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
            subject = Subject(120 , 120, shuffled_pieces)
            subject.place_pieces()  # Colocar en lámina base
            subject.get_fitness()  # Calcular fitness
            first_generation.append(subject)

        self.generations.append(first_generation)
        print(f"primera generacion {len(self.generations[-1])}")

    def get_last_generation(self):
        return self.generations[-1]

    def select_parents(self):
        """Selecciona parejas de padres de la última generación y las almacena en self.parents."""
        if not self.generations:
            print("⚠️ No hay generaciones previas para seleccionar padres.")
            return

        population = self.generations[-1]  # Última generación
        parents = []

        # Ordenar la población (puede ser por fitness si se desea)
        sorted_population = sorted(population, key=lambda ind: ind.fitness, reverse=True)

        # Emparejar individuos de la población ordenada
        for i in range(0, len(sorted_population) - 1, 2):
            parent1 = sorted_population[i]
            parent2 = sorted_population[i + 1]
            parents.append((parent1, parent2))

        self.parents = parents
        print(f"📌 Se seleccionaron {len(parents)} parejas de padres.")
    
    def crossover(self, crossover_prob=0.8):
        """Realiza cruza por intercambio equilibrado de piezas."""
        new_generation = []

        for parent1, parent2 in self.parents:
            if random.random() > crossover_prob:
                # Si no cruza, los padres pasan sin cambios
                new_generation.extend([copy.deepcopy(parent1), copy.deepcopy(parent2)])
                continue

            # Copia profunda de los padres
            child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

            # Seleccionar una pieza aleatoria en cada padre
            piece1 = random.choice(child1.pieces)  
            piece2 = random.choice(child2.pieces)

            # Intercambiar estas piezas
            index1, index2 = child1.pieces.index(piece1), child2.pieces.index(piece2)
            child1.pieces[index1], child2.pieces[index2] = child2.pieces[index2], child1.pieces[index1]

            # Buscar otra pieza del mismo tipo en cada padre para reequilibrar
            piece1_match = next((p for p in child1.pieces if p.width == piece2.width and p.height == piece2.height and p != piece1), None)
            piece2_match = next((p for p in child2.pieces if p.width == piece1.width and p.height == piece1.height and p != piece2), None)

            if piece1_match and piece2_match:
                # Intercambiar estas piezas secundarias para mantener la cantidad original
                index1_match, index2_match = child1.pieces.index(piece1_match), child2.pieces.index(piece2_match)
                child1.pieces[index1_match], child2.pieces[index2_match] = child2.pieces[index2_match], child1.pieces[index1_match]

            # Agregar los hijos a la nueva generación
            new_generation.extend([child1, child2])

        self.generations.append(new_generation)

    def mutate(self):
        """Muta las piezas de la población rotándolas con una probabilidad dada."""
        for subject in self.generations[-1]:
            if random.random() < self.mutation_rate:
                for piece in subject.pieces:
                    if random.random() < self.mutation_rate:
                        piece.width, piece.height = piece.height, piece.width  # Rotar la pieza
                        piece.rotated = not piece.rotated
                    subject.place_pieces()
                    

    def prune(self):
        """Reduce la población al tamaño original manteniendo el 20% de los mejores individuos y el 80% aleatorio."""
        population = self.generations[-1]
        population_size = self.population_size
        
        # Determinar cuántos individuos mantener por elitismo (20%)
        elite_count = max(1, int(0.2 * population_size))  # Asegura al menos 1 individuo elite
        random_count = population_size - elite_count  # El resto será aleatorio

        # Ordenar la población por fitness (mejores primero)
        sorted_population = sorted(population, key=lambda ind: ind.fitness, reverse=True)

        # Seleccionar los mejores (elitismo)
        elite_individuals = sorted_population[:elite_count]

        # Seleccionar el resto aleatoriamente del total de la población (sin importar fitness)
        random_individuals = random.sample(population, random_count)

        # Nueva generación con mezcla de élite y aleatorio
        self.generations[-1] = elite_individuals + random_individuals

    def start(self):
        self.set_first_generation()

        for _ in range(self.n_generations):
            self.select_parents()
            self.crossover()
            self.mutate()
            self.prune()

    def get_best_subject(self):
        return self.generations[-1][0]

    

lamina_base = (120, 120)

# Lista de piezas (ejemplo)
piezas = [Piece(50, 50) for _ in range(10)] + [Piece(20, 20) for _ in range(10)] +  [Piece(30, 40) for _ in range(10)]

# Crear algoritmo genético con 10 individuos y 50 generaciones
ga = GeneticAlgorithm(population_size=50, n_generations=500, base_plate=lamina_base, pieces=piezas, crossover_rate=0.6, mutation_rate=0.5)

ga.start()

best_subject = ga.get_best_subject()
best_subject.generate_images()

print(len(ga.generations))
print(f"fitness: {best_subject.fitness}")

# Generar primera generación

# Ver la primera generación
# for i, subject in enumerate(ga.generations[0]):
#     print(f"Individuo {i+1}: Fitness = {subject.fitness:.5f}")
#     subject.view_plates_distribution()
#     if (i == len(ga.generations)):
#         subject.generate_images()



          
