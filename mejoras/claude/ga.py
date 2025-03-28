from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from main import Piece, Subject, PlacementStrategy

class GeneticAlgorithm:
    
    def __init__(self, population_size, n_generations, plate_width, plate_height, pieces, crossover_rate=0.8, mutation_rate=0.1):
        self.population_size = population_size
        self.n_generations = n_generations
        self.plate_width = plate_width
        self.plate_height = plate_height
        self.generations = []
        self.original_pieces = pieces  # Guardamos las piezas originales
        self.parents = []
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
        # Nuevos parámetros para adaptabilidad
        self.adaptive_mutation = True
        self.diversity_threshold = 0.2
        self.elitism_rate = 0.1
        self.tournament_size = 3

    def set_first_generation(self):
        """Crea la primera generación con diversidad de estrategias"""
        first_generation = []

        for _ in range(self.population_size):
            # Copiar las piezas para este individuo
            shuffled_pieces = deepcopy(self.original_pieces)
            
            # Asignar IDs a las piezas si no los tienen
            for i, piece in enumerate(shuffled_pieces):
                if piece.id is None:
                    piece.id = i
                
                # Asignar estrategias aleatorias a algunas piezas
                if random.random() > 0.7:  # 30% de las piezas tienen estrategia propia
                    piece.placement_strategy = PlacementStrategy.get_random_strategy()
            
            # Aleatorizar el orden para cada individuo
            random.shuffle(shuffled_pieces)
            
            # Crear el sujeto con estas piezas
            subject = Subject(self.plate_width, self.plate_height, shuffled_pieces)
            
            # Colocar las piezas según las estrategias evolucionadas
            subject.place_pieces()
            
            # Agregar a la generación
            first_generation.append(subject)

        self.generations.append(first_generation)
        print(f"Primera generación creada con {len(self.generations[-1])} individuos")

    def get_last_generation(self):
        return self.generations[-1]

    def select_parents(self):
        """Selecciona parejas de padres mediante torneos"""
        if not self.generations:
            print("No hay generaciones previas para seleccionar padres.")
            return

        population = self.generations[-1]
        parents = []
        
        # Selección por torneo
        for _ in range(self.population_size // 2):  # Crear population_size/2 parejas
            # Primer padre (torneo)
            tournament1 = random.sample(population, self.tournament_size)
            parent1 = max(tournament1, key=lambda ind: ind.fitness)
            
            # Segundo padre (torneo)
            tournament2 = random.sample(population, self.tournament_size)
            parent2 = max(tournament2, key=lambda ind: ind.fitness)
            
            # Añadir la pareja
            parents.append((parent1, parent2))

        self.parents = parents
        print(f"Se seleccionaron {len(parents)} parejas de padres mediante torneos.")
    
    def crossover(self):
        """Cruce avanzado que combina estrategias de colocación además del orden de piezas"""
        new_generation = []

        for parent1, parent2 in self.parents:
            # Decidir si realizar cruce
            if random.random() > self.crossover_rate:
                # Agregar copias de los padres sin cambios
                new_generation.extend([deepcopy(parent1), deepcopy(parent2)])
                continue

            # Crear dos hijos
            child1 = deepcopy(parent1)
            child2 = deepcopy(parent2)
            
            # 1. Cruzar estrategias globales (intercambio de rasgos)
            self._crossover_global_strategies(child1, child2)
            
            # 2. Cruzar piezas y sus estrategias (cruce de orden parcial - PMX)
            self._crossover_pieces_pmx(child1, child2)
            
            # 3. Cruzar estrategias individuales de piezas
            self._crossover_piece_strategies(child1, child2)
            
            # Recalcular la colocación con las nuevas estrategias
            child1.place_pieces()
            child2.place_pieces()
            
            # Agregar a la nueva generación
            new_generation.extend([child1, child2])

        # Asegurar que tenemos el tamaño correcto de población
        if len(new_generation) > self.population_size:
            new_generation = new_generation[:self.population_size]
        
        self.generations.append(new_generation)
        print(f"Nueva generación creada con {len(new_generation)} individuos")
    
    def _crossover_global_strategies(self, child1, child2):
        """Intercambia estrategias globales entre dos individuos"""
        # Lista de estrategias a intercambiar
        strategies = list(child1.global_strategies.keys())
        
        # Seleccionar aleatoriamente algunas estrategias para intercambiar
        n_strategies = random.randint(1, len(strategies))
        strategies_to_swap = random.sample(strategies, n_strategies)
        
        # Intercambiar las estrategias seleccionadas
        for strategy in strategies_to_swap:
            child1.global_strategies[strategy], child2.global_strategies[strategy] = \
                child2.global_strategies[strategy], child1.global_strategies[strategy]
    
    def _crossover_pieces_pmx(self, child1, child2):
        """Partially Mapped Crossover (PMX) para las piezas"""
        n = len(child1.pieces)
        
        # Seleccionar dos puntos de cruce aleatorios
        point1, point2 = sorted(random.sample(range(n), 2))
        
        # Crear mapeos de índices basados en IDs de piezas
        mapping1 = {piece.id: i for i, piece in enumerate(child1.pieces)}
        mapping2 = {piece.id: i for i, piece in enumerate(child2.pieces)}
        
        # Crear nuevas listas de piezas
        new_pieces1 = child1.pieces.copy()
        new_pieces2 = child2.pieces.copy()
        
        # Sección de mapeo (entre los puntos de cruce)
        for i in range(point1, point2 + 1):
            # Obtener IDs de piezas en esta posición
            id1 = child1.pieces[i].id
            id2 = child2.pieces[i].id
            
            # Intercambiar piezas
            pos1 = mapping1[id2]
            pos2 = mapping2[id1]
            
            # Realizar el intercambio
            new_pieces1[i], new_pieces1[pos1] = new_pieces1[pos1], new_pieces1[i]
            new_pieces2[i], new_pieces2[pos2] = new_pieces2[pos2], new_pieces2[i]
            
            # Actualizar mapeos
            mapping1[id1], mapping1[id2] = mapping1[id2], mapping1[id1]
            mapping2[id1], mapping2[id2] = mapping2[id2], mapping2[id1]
        
        # Actualizar las piezas en los hijos
        child1.pieces = new_pieces1
        child2.pieces = new_pieces2
    
    def _crossover_piece_strategies(self, child1, child2):
        """Intercambia estrategias específicas de piezas entre dos individuos"""
        # Crear mapeos de piezas por ID para facilitar el acceso
        pieces1 = {piece.id: piece for piece in child1.pieces}
        pieces2 = {piece.id: piece for piece in child2.pieces}
        
        # Para cada pieza, hay una probabilidad de intercambiar estrategias
        for piece_id in pieces1.keys():
            if random.random() < 0.3:  # 30% de probabilidad de intercambio
                pieces1[piece_id].placement_strategy, pieces2[piece_id].placement_strategy = \
                    pieces2[piece_id].placement_strategy, pieces1[piece_id].placement_strategy
                
    def mutate(self):
        """Mutación avanzada que afecta tanto al orden como a las estrategias"""
        generation = self.generations[-1]
        
        # Calcular la diversidad actual para posible adaptación
        if self.adaptive_mutation:
            fitnesses = [subject.fitness for subject in generation]
            std_dev = np.std(fitnesses) if len(fitnesses) > 1 else 0
            normalized_std = std_dev / (max(fitnesses) - min(fitnesses)) if max(fitnesses) > min(fitnesses) else 0
            
            # Ajustar tasa de mutación basado en diversidad
            if normalized_std < self.diversity_threshold:
                # Aumentar mutación si hay poca diversidad
                current_mutation_rate = min(0.9, self.mutation_rate * 2)
                print(f"Poca diversidad detectada. Aumentando tasa de mutación a {current_mutation_rate:.2f}")
            else:
                current_mutation_rate = self.mutation_rate
        else:
            current_mutation_rate = self.mutation_rate
        
        for subject in generation:
            # 1. Mutación de estrategias globales
            if random.random() < current_mutation_rate:
                strategy_key = random.choice(list(subject.global_strategies.keys()))
                
                if strategy_key == 'default_placement':
                    subject.global_strategies[strategy_key] = PlacementStrategy.get_random_strategy()
                elif strategy_key == 'max_plates':
                    subject.global_strategies[strategy_key] = random.randint(1, 10)
                elif strategy_key in ('rotation_policy', 'grouping_similar'):
                    if isinstance(subject.global_strategies[strategy_key], bool):
                        subject.global_strategies[strategy_key] = not subject.global_strategies[strategy_key]
                    else:
                        subject.global_strategies[strategy_key] = random.random()
            
            # 2. Mutación de estrategias individuales de piezas
            for piece in subject.pieces:
                if random.random() < current_mutation_rate:
                    # Decidir si asignar una estrategia específica o usar la global
                    if random.random() < 0.7:
                        piece.placement_strategy = PlacementStrategy.get_random_strategy()
                    else:
                        piece.placement_strategy = None
                
                # Mutación de rotación de piezas
                if random.random() < current_mutation_rate:
                    piece.rotate()
            
            # 3. Mutación de orden (swap)
            if random.random() < current_mutation_rate:
                # Número de swaps a realizar
                n_swaps = max(1, int(len(subject.pieces) * 0.1))
                
                for _ in range(n_swaps):
                    idx1, idx2 = random.sample(range(len(subject.pieces)), 2)
                    subject.pieces[idx1], subject.pieces[idx2] = subject.pieces[idx2], subject.pieces[idx1]
            
            # 4. Mutación de inversión (invertir un segmento)
            if random.random() < current_mutation_rate * 0.5:  # Menos frecuente
                idx1, idx2 = sorted(random.sample(range(len(subject.pieces)), 2))
                subject.pieces[idx1:idx2+1] = reversed(subject.pieces[idx1:idx2+1])
            
            # Recalcular la colocación con las nuevas estrategias y orden
            subject.place_pieces()  

    def prune(self):
        """Selección para la próxima generación con elitismo y diversidad"""
        population = self.generations[-1]
        
        # Calcular el número de elites a mantener
        elite_count = max(1, int(self.elitism_rate * self.population_size))
        
        # Ordenar por fitness
        sorted_population = sorted(population, key=lambda ind: ind.fitness, reverse=True)
        
        # Seleccionar los mejores individuos (elitismo)
        elite_individuals = sorted_population[:elite_count]
        
        # Para el resto, usar selección por torneo para mantener diversidad
        remaining_count = self.population_size - elite_count
        remaining_individuals = []
        
        for _ in range(remaining_count):
            # Selección por torneo
            tournament = random.sample(population, self.tournament_size)
            winner = max(tournament, key=lambda ind: ind.fitness)
            remaining_individuals.append(deepcopy(winner))
        
        # Actualizar la población
        self.generations[-1] = elite_individuals + remaining_individuals
        
        # Registrar estadísticas
        all_fitnesses = [ind.fitness for ind in self.generations[-1]]
        if all_fitnesses:
            self.best_fitness_history.append(max(all_fitnesses))
            self.avg_fitness_history.append(sum(all_fitnesses) / len(all_fitnesses))
                
    def start(self):
        """Ejecuta el algoritmo genético completo"""
        print("Iniciando algoritmo genético evolucionado...")
        self.set_first_generation()

        for gen in range(self.n_generations):
            print(f"\n===== Generación {gen+1}/{self.n_generations} =====")
            
            # Obtener estadísticas de la generación actual
            population = self.generations[-1]
            fitnesses = [ind.fitness for ind in population]
            avg_fitness = sum(fitnesses) / len(fitnesses) if fitnesses else 0
            best_fitness = max(fitnesses) if fitnesses else 0
            best_individual = max(population, key=lambda ind: ind.fitness) if population else None
            
            print(f"Fitness promedio: {avg_fitness:.4f}")
            print(f"Mejor fitness: {best_fitness:.4f}")
            
            if best_individual:
                strategies_used = best_individual.calculate_strategy_stats()
                print(f"Mejor individuo usa estrategia global: {strategies_used['strategies']['global']}")
                print(f"Piezas con estrategia específica: {strategies_used['strategies']['piece_specific']}/{strategies_used['total_pieces']}")
            
            # Evolucionar
            self.select_parents()
            self.crossover()
            self.mutate()
            self.prune()
        
        print("\n===== Evolución completada =====")
        self.plot_evolution()

    def get_best_subject(self):
        """Retorna el mejor individuo de la última generación"""
        if not self.generations:
            return None
        
        sorted_population = sorted(self.generations[-1], key=lambda ind: ind.fitness, reverse=True)
        return sorted_population[0] if sorted_population else None
        
    def plot_evolution(self, filename="evolucion.png"):
        """Genera un gráfico con la evolución del fitness a lo largo de las generaciones"""
        if not self.best_fitness_history:
            print("No hay datos de evolución para graficar.")
            return
        
        generations = range(1, len(self.best_fitness_history) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, self.best_fitness_history, 'b-', label='Mejor Fitness')
        plt.plot(generations, self.avg_fitness_history, 'r-', label='Fitness Promedio')
        
        plt.title('Evolución del Fitness')
        plt.xlabel('Generación')
        plt.ylabel('Fitness')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
        print(f"Gráfico de evolución guardado como '{filename}'")
                
def run_genetic_algorithm(plate_width, plate_height, pieces, population_size=20, n_generations=20, crossover_rate=0.8, mutation_rate=0.2):
    """Función para ejecutar el algoritmo genético con parámetros específicos"""
    # Asignar IDs a las piezas originales
    for i, piece in enumerate(pieces):
        piece.id = i
    
    # Crear y ejecutar el algoritmo
    ga = GeneticAlgorithm(
        population_size=population_size, 
        n_generations=n_generations, 
        plate_width=plate_width, 
        plate_height=plate_height, 
        pieces=pieces, 
        crossover_rate=crossover_rate, 
        mutation_rate=mutation_rate
    )
    
    ga.start()
    
    # Obtener y mostrar el mejor resultado
    best_subject = ga.get_best_subject()
    
    if best_subject:
        print("\n===== Mejor solución encontrada =====")
        print(f"Fitness: {best_subject.fitness:.4f}")
        print(f"Número de láminas utilizadas: {len(best_subject.plates_used)}")
        
        # Generar imágenes de la mejor solución
        best_subject.generate_images(folder="mejor_solucion")
    
    return ga, best_subject
                

def main():
    # Definir dimensiones de la lámina base
    lamina_base_width = 120
    lamina_base_height = 120
    
    # Crear piezas con diferentes tamaños
    piezas = [
        Piece(50, 50) for _ in range(10)  # Piezas grandes cuadradas
    ] + [
        Piece(20, 20) for _ in range(10)  # Piezas pequeñas cuadradas
    ] + [
        Piece(30, 40) for _ in range(10)  # Piezas rectangulares medianas
    ] + [
        Piece(60, 20) for _ in range(5)   # Piezas rectangulares largas
    ] + [
        Piece(15, 80) for _ in range(5)   # Piezas rectangulares altas
    ]
    
    # Parámetros del algoritmo genético
    poblacion = 30
    generaciones = 20
    tasa_cruce = 0.8
    tasa_mutacion = 0.2
    
    print(f"Ejecutando algoritmo genético evolucionado con:")
    print(f"- {len(piezas)} piezas")
    print(f"- Lámina base: {lamina_base_width}x{lamina_base_height}")
    print(f"- Población: {poblacion}")
    print(f"- Generaciones: {generaciones}")
    print(f"- Tasa de cruce: {tasa_cruce}")
    print(f"- Tasa de mutación: {tasa_mutacion}")
    
    # Ejecutar el algoritmo
    ga, mejor_solucion = run_genetic_algorithm(
        plate_width=lamina_base_width,
        plate_height=lamina_base_height,
        pieces=piezas,
        population_size=poblacion,
        n_generations=generaciones,
        crossover_rate=tasa_cruce,
        mutation_rate=tasa_mutacion
    )
    
    # Mostrar estadísticas de la mejor solución
    if mejor_solucion:
        print("\nEstadísticas de la mejor solución:")
        stats = mejor_solucion.calculate_strategy_stats()
        print(f"- Piezas colocadas: {stats['placed_pieces']}/{stats['total_pieces']}")
        print(f"- Estrategia global: {stats['strategies']['global']}")
        print(f"- Política de rotación: {stats['strategies']['rotation_policy']}")
        print(f"- Máximo de láminas: {stats['strategies']['max_plates']}")
        print(f"- Piezas con estrategia específica: {stats['strategies']['piece_specific']}")
        
        # La función generate_images ya se llama dentro de run_genetic_algorithm

if __name__ == "__main__":
    main()