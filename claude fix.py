import tkinter as tk
from tkinter import ttk, messagebox
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv
import os
import numpy as np
from copy import deepcopy

# Datos iniciales
laminas = []  # Lista de laminas (ancho, alto)
piezas = []  # Lista de piezas (ancho, alto)
piezas_seleccionadas = []
lamina_seleccionada = None

# Algoritmo Genético Configuración
POP_SIZE = 30  # Tamaño de la población
GENERATIONS = 30  # Número de generaciones
MUTATION_RATE = 0.1  # Probabilidad de mutación
CROSSOVER_RATE = 0.8  # Probabilidad de cruce

# Verificar y crear archivos CSV si no existen
def inicializar_csv():
    if not os.path.exists("laminas.csv"):
        with open("laminas.csv", "w", newline='') as file:
            pass
    if not os.path.exists("piezas.csv"):
        with open("piezas.csv", "w", newline='') as file:
            pass

# Cargar datos desde CSV
def cargar_datos():
    global laminas, piezas
    try:
        with open("laminas.csv", "r") as file:
            reader = csv.reader(file)
            laminas = [(int(row[0]), int(row[1])) for row in reader if row]
    except FileNotFoundError:
        laminas = []
    
    try:
        with open("piezas.csv", "r") as file:
            reader = csv.reader(file)
            piezas = [(int(row[0]), int(row[1])) for row in reader if row]
    except FileNotFoundError:
        piezas = []

# Guardar datos en CSV
def guardar_datos():
    with open("laminas.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(laminas)
    
    with open("piezas.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(piezas)

# Función para registrar una nueva lámina
def agregar_lamina():
    try:
        ancho = int(entry_ancho_lamina.get())
        alto = int(entry_alto_lamina.get())
        laminas.append((ancho, alto))
        guardar_datos()
        actualizar_listas()
    except ValueError:
        messagebox.showerror("Error", "Ingrese valores numéricos válidos")

# Función para registrar una nueva pieza
def agregar_pieza():
    try:
        ancho = int(entry_ancho_pieza.get())
        alto = int(entry_alto_pieza.get())
        piezas.append((ancho, alto))
        guardar_datos()
        actualizar_listas()
    except ValueError:
        messagebox.showerror("Error", "Ingrese valores numéricos válidos")

# Función para actualizar listas en la interfaz
def actualizar_listas():
    lista_laminas.delete(0, tk.END)
    for i, (ancho, alto) in enumerate(laminas):
        lista_laminas.insert(tk.END, f"Lámina {i+1}: {ancho}x{alto} cm")
    
    lista_piezas.delete(0, tk.END)
    for i, (ancho, alto) in enumerate(piezas):
        lista_piezas.insert(tk.END, f"Pieza {i+1}: {ancho}x{alto} cm")

# Función para seleccionar una lámina
def seleccionar_lamina():
    global lamina_seleccionada
    seleccion = lista_laminas.curselection()
    if seleccion:
        lamina_seleccionada = laminas[seleccion[0]]
        etiqueta_lamina.config(text=f"Lámina seleccionada: {lamina_seleccionada[0]}x{lamina_seleccionada[1]} cm")
    else:
        messagebox.showwarning("Advertencia", "Seleccione una lámina")

# Función para seleccionar piezas con cantidad
def seleccionar_pieza():
    seleccion = lista_piezas.curselection()
    if seleccion:
        try:
            cantidad = int(entry_cantidad_pieza.get())
            pieza = piezas[seleccion[0]]
            piezas_seleccionadas.append((pieza, cantidad))
            actualizar_tabla_piezas()
        except ValueError:
            messagebox.showerror("Error", "Ingrese una cantidad válida")
    else:
        messagebox.showwarning("Advertencia", "Seleccione una pieza")

# Función para actualizar la tabla de piezas seleccionadas
def actualizar_tabla_piezas():
    for row in tabla_piezas.get_children():
        tabla_piezas.delete(row)
    for pieza, cantidad in piezas_seleccionadas:
        tabla_piezas.insert("", tk.END, values=(f"{pieza[0]}x{pieza[1]}", cantidad))

# Clases para el algoritmo genético
class Pieza:
    def __init__(self, ancho, alto, id=None, rotada=False):
        self.ancho = ancho
        self.alto = alto
        self.id = id
        self.tipo_id = None  # Agregar esta línea
        self.rotada = rotada
        self.x = 0
        self.y = 0
        self.en_lamina = 0  # Nueva propiedad para trackear lámina
    
    def rotar(self):
        self.ancho, self.alto = self.alto, self.ancho
        self.rotada = not self.rotada
        return self
    
    def area(self):
        return self.ancho * self.alto
    
    def copia(self):
        nueva_pieza = Pieza(self.ancho, self.alto, self.id, self.rotada)
        nueva_pieza.tipo_id = self.tipo_id  # Copiar el tipo_id
        nueva_pieza.x = self.x
        nueva_pieza.y = self.y
        nueva_pieza.en_lamina = self.en_lamina
        return nueva_pieza
    
class Individuo:
    def __init__(self, piezas_candidatas, lamina_ancho, lamina_alto):
        self.piezas = piezas_candidatas
        self.lamina_ancho = lamina_ancho
        self.lamina_alto = lamina_alto
        self.fitness = 0
        self.distribuciones = []  # Lista de diccionarios con distribución por lámina
        self.num_laminas = 1
        self.areas_residuales = []
    
    def _calcular_distribucion_laminar(self, piezas_por_ubicar):
        """Coloca piezas en una lámina y devuelve la distribución y piezas no ubicadas"""
        matriz_lamina = np.zeros((self.lamina_alto, self.lamina_ancho))
        distribucion = []
        piezas_no_ubicadas = []
        area_utilizada = 0
        
        # Ordenar piezas por área (de mayor a menor)
        piezas_ordenadas = sorted(piezas_por_ubicar, key=lambda p: p.area(), reverse=True)
        
        for pieza in piezas_ordenadas:
            ubicada = False
            
            # Intentar en la orientación actual
            for y in range(self.lamina_alto - pieza.alto + 1):
                for x in range(self.lamina_ancho - pieza.ancho + 1):
                    if self._puede_ubicar(matriz_lamina, x, y, pieza.ancho, pieza.alto):
                        # Ubicar la pieza
                        self._marcar_matriz(matriz_lamina, x, y, pieza.ancho, pieza.alto)
                        pieza_ubicada = pieza.copia()
                        pieza_ubicada.x = x
                        pieza_ubicada.y = y
                        pieza_ubicada.en_lamina = len(self.distribuciones) + 1
                        distribucion.append(pieza_ubicada)
                        area_utilizada += pieza.area()
                        ubicada = True
                        break
                if ubicada:
                    break
            
            # Si no se pudo ubicar, intentar rotada (solo si no es cuadrada)
            if not ubicada and pieza.ancho != pieza.alto:
                pieza_rotada = pieza.copia().rotar()
                for y in range(self.lamina_alto - pieza_rotada.alto + 1):
                    for x in range(self.lamina_ancho - pieza_rotada.ancho + 1):
                        if self._puede_ubicar(matriz_lamina, x, y, pieza_rotada.ancho, pieza_rotada.alto):
                            # Ubicar la pieza rotada
                            self._marcar_matriz(matriz_lamina, x, y, pieza_rotada.ancho, pieza_rotada.alto)
                            pieza_ubicada = pieza_rotada.copia()
                            pieza_ubicada.x = x
                            pieza_ubicada.y = y
                            pieza_ubicada.en_lamina = len(self.distribuciones) + 1
                            distribucion.append(pieza_ubicada)
                            area_utilizada += pieza_rotada.area()
                            ubicada = True
                            break
                    if ubicada:
                        break
            
            if not ubicada:
                piezas_no_ubicadas.append(pieza)
        
        # Calcular áreas residuales
        areas_residuales = self._calcular_areas_residuales(matriz_lamina)
        
        return {
            'matriz': matriz_lamina,
            'piezas': distribucion,
            'piezas_no_ubicadas': piezas_no_ubicadas,
            'area_utilizada': area_utilizada,
            'areas_residuales': areas_residuales
        }
    
    def _calcular_areas_residuales(self, matriz):
        """Identifica áreas residuales usando algoritmo de búsqueda por regiones"""
        areas = []
        visitados = np.zeros_like(matriz)
        
        for y in range(matriz.shape[0]):
            for x in range(matriz.shape[1]):
                if matriz[y, x] == 0 and visitados[y, x] == 0:
                    # Encontrar el área residual máxima desde este punto
                    ancho_max = 0
                    alto_max = 0
                    
                    # Buscar ancho máximo
                    for j in range(x, matriz.shape[1]):
                        if j < matriz.shape[1] and matriz[y, j] == 0:
                            ancho_max += 1
                        else:
                            break
                    
                    # Buscar alto máximo
                    for i in range(y, matriz.shape[0]):
                        if i < matriz.shape[0] and matriz[i, x] == 0:
                            alto_max += 1
                        else:
                            break
                    
                    # Verificar si el área es rectangular
                    es_rectangular = True
                    for i in range(y, y + alto_max):
                        for j in range(x, x + ancho_max):
                            if i < matriz.shape[0] and j < matriz.shape[1]:
                                if matriz[i, j] == 1:
                                    es_rectangular = False
                                    break
                        if not es_rectangular:
                            break
                    
                    if es_rectangular:
                        areas.append(ancho_max * alto_max)
                        visitados[y:y+alto_max, x:x+ancho_max] = 1
        
        return areas
    
    def _puede_ubicar(self, matriz, x, y, ancho, alto):
        """Verifica si la pieza cabe en la posición (x,y)"""
        if y + alto > matriz.shape[0] or x + ancho > matriz.shape[1]:
            return False
        return np.all(matriz[y:y+alto, x:x+ancho] == 0)
    
    def _marcar_matriz(self, matriz, x, y, ancho, alto):
        """Marca el área ocupada en la matriz"""
        matriz[y:y+alto, x:x+ancho] = 1
    
    def calcular_fitness(self):
        """Nuevo cálculo de fitness considerando múltiples láminas y áreas residuales"""
        self.distribuciones = []
        piezas_restantes = self.piezas.copy()
        total_area_piezas = sum(p.area() for p in self.piezas)
        total_area_utilizada = 0
        max_residual = 0
        total_residual = 0
        
        while len(piezas_restantes) > 0:
            distribucion = self._calcular_distribucion_laminar(piezas_restantes)
            self.distribuciones.append(distribucion)
            piezas_restantes = distribucion['piezas_no_ubicadas']
            total_area_utilizada += distribucion['area_utilizada']
            
            # Calcular áreas residuales para esta lámina
            if distribucion['areas_residuales']:
                max_residual = max(max_residual, max(distribucion['areas_residuales']))
                total_residual += sum(distribucion['areas_residuales'])
        
        # Penalización por múltiples láminas
        penalizacion_laminas = 0.1 * (len(self.distribuciones) - 1)
        
        # Área de una lámina
        area_lamina = self.lamina_ancho * self.lamina_alto
        
        # Eficiencia de uso de material
        eficiencia = total_area_utilizada / (area_lamina * len(self.distribuciones))
        
        # Cálculo de fitness considerando eficiencia y penalización por múltiples láminas
        self.fitness = eficiencia - penalizacion_laminas
        
        # Asegurar que el fitness esté entre 0 y 1
        self.fitness = max(0, min(1, self.fitness))
        
        self.num_laminas = len(self.distribuciones)
        return self.fitness

# Funciones para el algoritmo genético
def crear_poblacion_inicial(piezas_seleccionadas, lamina_ancho, lamina_alto):
    """Crea una población inicial de individuos manteniendo las cantidades exactas de cada tipo de pieza"""
    poblacion = []
    
    # Crear la lista base de piezas con las cantidades correctas
    piezas_base = []
    contador_id = 0
    tipo_id = 0  # Identificador para el tipo de pieza
    
    for pieza_raw, cantidad in piezas_seleccionadas:
        for i in range(cantidad):
            # Decidir aleatoriamente si rotar la pieza inicialmente
            rotada = random.random() > 0.5
            pieza = Pieza(pieza_raw[0], pieza_raw[1], id=contador_id)
            pieza.tipo_id = tipo_id  # Guardar a qué tipo pertenece
            contador_id += 1
            
            if rotada and pieza.ancho != pieza.alto:  # No rotar si es cuadrada
                pieza.rotar()
            piezas_base.append(pieza)
        tipo_id += 1  # Incrementar para el siguiente tipo de pieza
    
    # Para cada individuo en la población
    for _ in range(POP_SIZE):
        # Copiar la lista de piezas base para este individuo
        piezas_candidatas = deepcopy(piezas_base)
        
        # Barajar las piezas para crear un orden aleatorio
        random.shuffle(piezas_candidatas)
        
        # Crear individuo
        individuo = Individuo(piezas_candidatas, lamina_ancho, lamina_alto)
        individuo.calcular_fitness()
        poblacion.append(individuo)
    
    return poblacion

def seleccion_torneo(poblacion):
    """Selección por torneo"""
    seleccionados = []
    for _ in range(POP_SIZE):
        # Seleccionar 3 individuos al azar
        competidores = random.sample(poblacion, 3)
        # Elegir el mejor
        ganador = max(competidores, key=lambda x: x.fitness)
        seleccionados.append(ganador)
    return seleccionados

def cruce(padre1, padre2):
    """Cruce de dos individuos manteniendo las cantidades exactas de cada tipo de pieza"""
    if random.random() > CROSSOVER_RATE:
        return deepcopy(padre1), deepcopy(padre2)
    
    # Mapeo de piezas por tipo para padre 1
    piezas_por_tipo_p1 = {}
    for pieza in padre1.piezas:
        if pieza.tipo_id not in piezas_por_tipo_p1:
            piezas_por_tipo_p1[pieza.tipo_id] = []
        piezas_por_tipo_p1[pieza.tipo_id].append(pieza)
    
    # Mapeo de piezas por tipo para padre 2
    piezas_por_tipo_p2 = {}
    for pieza in padre2.piezas:
        if pieza.tipo_id not in piezas_por_tipo_p2:
            piezas_por_tipo_p2[pieza.tipo_id] = []
        piezas_por_tipo_p2[pieza.tipo_id].append(pieza)
    
    # Para cada tipo de pieza, seleccionar aleatoriamente cómo se distribuyen entre los hijos
    hijo1_piezas = []
    hijo2_piezas = []
    
    for tipo_id in piezas_por_tipo_p1.keys():
        # Determinar cuántas piezas de este tipo hay
        num_piezas = len(piezas_por_tipo_p1[tipo_id])
        punto_corte = random.randint(0, num_piezas)
        
        # Hijo 1 recibe piezas del padre 1 y padre 2
        hijo1_piezas.extend(deepcopy(piezas_por_tipo_p1[tipo_id][:punto_corte]))
        hijo1_piezas.extend(deepcopy(piezas_por_tipo_p2[tipo_id][punto_corte:]))
        
        # Hijo 2 recibe piezas del padre 2 y padre 1
        hijo2_piezas.extend(deepcopy(piezas_por_tipo_p2[tipo_id][:punto_corte]))
        hijo2_piezas.extend(deepcopy(piezas_por_tipo_p1[tipo_id][punto_corte:]))
    
    # Crear nuevos individuos
    hijo1 = Individuo(hijo1_piezas, padre1.lamina_ancho, padre1.lamina_alto)
    hijo2 = Individuo(hijo2_piezas, padre2.lamina_ancho, padre2.lamina_alto)
    
    return hijo1, hijo2


def mutacion(individuo):
    """Mutación de un individuo sin alterar la composición de piezas"""
    for i in range(len(individuo.piezas)):
        if random.random() < MUTATION_RATE:
            # Tipos de mutación:
            # 1. Cambiar posición
            # 2. Rotar pieza
            tipo_mutacion = random.randint(1, 2)
            
            if tipo_mutacion == 1:
                # Cambiar posición (intercambiar con otra pieza)
                # Buscar otra pieza del mismo tipo para intercambiar
                mismo_tipo = [j for j in range(len(individuo.piezas)) 
                               if individuo.piezas[j].tipo_id == individuo.piezas[i].tipo_id]
                if len(mismo_tipo) > 1:  # Solo intercambiar si hay más de una pieza del mismo tipo
                    j = random.choice(mismo_tipo)
                    if i != j:  # Evitar intercambiar consigo misma
                        individuo.piezas[i], individuo.piezas[j] = individuo.piezas[j], individuo.piezas[i]
            else:
                # Rotar pieza
                if individuo.piezas[i].ancho != individuo.piezas[i].alto:  # No rotar si es cuadrada
                    individuo.piezas[i].rotar()
    
    return individuo

def algoritmo_genetico(piezas_seleccionadas, lamina_ancho, lamina_alto):
    """Algoritmo genético principal"""
    # Crear población inicial
    poblacion = crear_poblacion_inicial(piezas_seleccionadas, lamina_ancho, lamina_alto)
    
    # Evolución
    mejor_individuo = max(poblacion, key=lambda x: x.fitness)
    mejores_fitness = [mejor_individuo.fitness]
    
    # Función de evaluación considerando el número de láminas
    def evaluar_individuo(ind):
        return ind.fitness * (1 - 0.05 * ind.num_laminas)
    
    for generacion in range(GENERATIONS):
        # Selección usando la métrica personalizada
        seleccionados = []
        for _ in range(POP_SIZE):
            competidores = random.sample(poblacion, min(3, len(poblacion)))
            ganador = max(competidores, key=lambda x: evaluar_individuo(x))
            seleccionados.append(ganador)
        
        # Cruce
        nueva_poblacion = []
        for i in range(0, POP_SIZE, 2):
            if i + 1 < POP_SIZE:
                hijo1, hijo2 = cruce(seleccionados[i], seleccionados[i + 1])
                nueva_poblacion.append(hijo1)
                nueva_poblacion.append(hijo2)
            else:
                nueva_poblacion.append(seleccionados[i])
        
        # Mutación
        for i in range(len(nueva_poblacion)):
            nueva_poblacion[i] = mutacion(nueva_poblacion[i])
        
        # Terminar temprano si encontramos una solución óptima
        if mejor_individuo.fitness >= 0.95 and mejor_individuo.num_laminas == 1:
            print("Solución óptima encontrada, terminando temprano")
            break
        
        # Calcular fitness de la nueva población
        for individuo in nueva_poblacion:
            individuo.calcular_fitness()
        
        # Elitismo (mantener al mejor individuo)
        mejor_actual = max(nueva_poblacion, key=lambda x: x.fitness)
        if evaluar_individuo(mejor_actual) > evaluar_individuo(mejor_individuo):
            mejor_individuo = mejor_actual
        else:
            # Reemplazar el peor individuo por el mejor de la generación anterior
            if len(nueva_poblacion) > 0:
                peor_idx = min(range(len(nueva_poblacion)), key=lambda i: evaluar_individuo(nueva_poblacion[i]))
                nueva_poblacion[peor_idx] = mejor_individuo
        
        # Actualizar población
        poblacion = nueva_poblacion
        
        # Registrar el mejor fitness
        mejores_fitness.append(mejor_individuo.fitness)
        
        # Mostrar progreso
        if generacion % 5 == 0:
            print(f"Generación {generacion}: Mejor fitness = {mejor_individuo.fitness}, Láminas = {mejor_individuo.num_laminas}")
    
    print(f"Mejor fitness final: {mejor_individuo.fitness}, Láminas utilizadas: {mejor_individuo.num_laminas}")
    
    return mejor_individuo, mejores_fitness

# Expandir las piezas seleccionadas para el algoritmo genético
def expandir_piezas(piezas_seleccionadas):
    todas_piezas = []
    id_counter = 0
    
    for pieza, cantidad in piezas_seleccionadas:
        for i in range(cantidad):
            nueva_pieza = Pieza(pieza[0], pieza[1], id=id_counter)
            todas_piezas.append(nueva_pieza)
            id_counter += 1
    
    return todas_piezas

# Algoritmo Genético - Ejecución
def ejecutar_algoritmo_genetico():
    global mejor_individuo_global
    
    if not lamina_seleccionada:
        messagebox.showerror("Error", "Seleccione una lámina")
        return
    if not piezas_seleccionadas:
        messagebox.showerror("Error", "Seleccione al menos una pieza")
        return
    
    print("Ejecutando algoritmo genético...")
    
    # Obtener dimensiones de la lámina
    lamina_ancho = lamina_seleccionada[0]
    lamina_alto = lamina_seleccionada[1]
    
    # Verificar que las piezas quepan en la lámina
    for pieza, _ in piezas_seleccionadas:
        if (pieza[0] > lamina_ancho and pieza[0] > lamina_alto) and (pieza[1] > lamina_ancho and pieza[1] > lamina_alto):
            messagebox.showerror("Error", f"La pieza {pieza[0]}x{pieza[1]} no cabe en la lámina incluso rotada")
            return
    
    # Actualizar parámetros del algoritmo genético
    try:
        global POP_SIZE, GENERATIONS, MUTATION_RATE, CROSSOVER_RATE
        POP_SIZE = int(entry_pop_size.get())
        GENERATIONS = int(entry_generations.get())
        MUTATION_RATE = float(entry_mutation.get())
        CROSSOVER_RATE = float(entry_crossover.get())
    except ValueError:
        messagebox.showwarning("Advertencia", "Se usarán valores predeterminados para los parámetros")
    
    # Ejecutar el algoritmo genético con las piezas seleccionadas directamente
    mejor_individuo, historico_fitness = algoritmo_genetico(piezas_seleccionadas, lamina_ancho, lamina_alto)
    mejor_individuo_global = mejor_individuo
    
    # Actualizar estadísticas
    area_lamina = lamina_ancho * lamina_alto * mejor_individuo.num_laminas
    area_total_piezas = sum(pieza[0] * pieza[1] * cantidad for pieza, cantidad in piezas_seleccionadas)
    
    lbl_area_total.config(text=f"Área total de lámina: {area_lamina} cm²")
    lbl_area_usada.config(text=f"Área total de piezas: {area_total_piezas} cm²")
    lbl_aprovechamiento.config(text=f"Aprovechamiento: {mejor_individuo.fitness*100:.2f}%")
    
    total_piezas = sum(cantidad for _, cantidad in piezas_seleccionadas)
    lbl_piezas_colocadas.config(text=f"Piezas colocadas: {total_piezas}")
    
    # Activar botones de exportación
    btn_guardar_resultado.config(state=tk.NORMAL, command=lambda: guardar_resultado(mejor_individuo))
    btn_exportar_imagen.config(state=tk.NORMAL, command=lambda: exportar_imagen(mejor_individuo))
    
    messagebox.showinfo("Info", f"Optimización completada. Aprovechamiento: {mejor_individuo.fitness*100:.2f}%")
    
    # Mostrar resultados
    mostrar_resultado(mejor_individuo)
    mostrar_grafica_evolucion(historico_fitness)

# Función para mostrar el resultado gráfico
def mostrar_resultado(individuo):
    if not individuo.distribuciones:
        messagebox.showwarning("Advertencia", "No hay distribución para mostrar")
        return
    
    # Crear carpeta para guardar las imágenes
    output_dir = "resultados_lamina"
    os.makedirs(output_dir, exist_ok=True)
    
    # Para cada lámina/distribución
    for idx, distribucion in enumerate(individuo.distribuciones):
        # Crear figura individual para cada lámina
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.set_title(f"Lámina {idx+1} - Piezas colocadas: {len(distribucion['piezas'])}")
        ax.set_xlim(0, individuo.lamina_ancho)
        ax.set_ylim(0, individuo.lamina_alto)
        
        # Dibujar lámina
        ax.add_patch(patches.Rectangle(
            (0, 0), individuo.lamina_ancho, individuo.lamina_alto,
            edgecolor='black', facecolor='none', linewidth=2
        ))
        
        # Dibujar piezas
        cmap = plt.cm.get_cmap('tab20', len(distribucion['piezas']))
        for i, pieza in enumerate(distribucion['piezas']):
            color = cmap(i % 20)
            ax.add_patch(patches.Rectangle(
                (pieza.x, pieza.y), pieza.ancho, pieza.alto,
                edgecolor='black', facecolor=color, alpha=0.7
            ))
            ax.text(
                pieza.x + pieza.ancho/2, pieza.y + pieza.alto/2,
                f"{pieza.ancho}x{pieza.alto}", ha='center', va='center'
            )
        
        # Agregar una cuadrícula
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Guardar figura
        filename = os.path.join(output_dir, f"lamina_{idx+1}.png")
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)  # Cerrar figura para liberar memoria
    
    messagebox.showinfo("Info", f"Gráficas guardadas en la carpeta: '{output_dir}'")

# Función para mostrar la gráfica de evolución
def mostrar_grafica_evolucion(historico_fitness):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(historico_fitness)), historico_fitness, 'b-', linewidth=2)
    ax.set_title("Evolución del fitness a través de las generaciones")
    ax.set_xlabel("Generación")
    ax.set_ylabel("Fitness (% de aprovechamiento)")
    ax.grid(True)
    plt.savefig("evolucion_fitness.png")
    plt.show()

# Función para limpiar piezas seleccionadas
def limpiar_seleccion():
    global piezas_seleccionadas
    piezas_seleccionadas = []
    actualizar_tabla_piezas()

# Función para guardar el resultado
def guardar_resultado(individuo):
    try:
        with open("resultado_optimizacion.csv", "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Lámina", "Pieza", "Ancho", "Alto", "Posicion X", "Posicion Y", "Rotada"])
            for idx, distribucion in enumerate(individuo.distribuciones):
                for pieza in distribucion['piezas']:
                    writer.writerow([idx+1, pieza.id, pieza.ancho, pieza.alto, 
                                     pieza.x, pieza.y, pieza.rotada])
        messagebox.showinfo("Info", "Resultado guardado en resultado_optimizacion.csv")
    except Exception as e:
        messagebox.showerror("Error", f"Error al guardar el resultado: {str(e)}")

# Inicializar CSV e interfaz
inicializar_csv()
cargar_datos()

# Interfaz gráfica
root = tk.Tk()
root.title("Optimización de Corte con Algoritmos Genéticos")
root.geometry("950x600")

# Frame principal
frame_principal = tk.Frame(root)
frame_principal.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Notebook (pestañas)
notebook = ttk.Notebook(frame_principal)
notebook.pack(fill=tk.BOTH, expand=True)

# Pestaña de configuración
tab_config = ttk.Frame(notebook)
notebook.add(tab_config, text="Configuración")

# Dividir la pestaña de configuración en dos frames
frame_izq = tk.Frame(tab_config)
frame_izq.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
frame_der = tk.Frame(tab_config)
frame_der.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

# Sección de catálogo de láminas
frame_laminas = tk.LabelFrame(frame_izq, text="Catálogo de Láminas")
frame_laminas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

lista_laminas = tk.Listbox(frame_laminas, height=10)
lista_laminas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

# Botón para seleccionar lámina
btn_seleccionar_lamina = tk.Button(frame_laminas, text="Seleccionar Lámina", command=seleccionar_lamina)
btn_seleccionar_lamina.pack(fill=tk.X, padx=5, pady=5)

# Etiqueta para mostrar la lámina seleccionada
etiqueta_lamina = tk.Label(frame_laminas, text="Lámina seleccionada: Ninguna")
etiqueta_lamina.pack(fill=tk.X, padx=5, pady=5)

# Sección para agregar láminas
frame_agregar_lamina = tk.LabelFrame(frame_izq, text="Agregar Nueva Lámina")
frame_agregar_lamina.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

tk.Label(frame_agregar_lamina, text="Ancho (cm):").pack(anchor=tk.W, padx=5)
entry_ancho_lamina = tk.Entry(frame_agregar_lamina)
entry_ancho_lamina.pack(fill=tk.X, padx=5, pady=5)

tk.Label(frame_agregar_lamina, text="Alto (cm):").pack(anchor=tk.W, padx=5)
entry_alto_lamina = tk.Entry(frame_agregar_lamina)
entry_alto_lamina.pack(fill=tk.X, padx=5, pady=5)

tk.Button(frame_agregar_lamina, text="Agregar Lámina", command=agregar_lamina).pack(fill=tk.X, padx=5, pady=5)

# Sección de catálogo de piezas
frame_piezas = tk.LabelFrame(frame_der, text="Catálogo de Piezas")
frame_piezas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

lista_piezas = tk.Listbox(frame_piezas, height=10)
lista_piezas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

# Sección para seleccionar piezas
frame_seleccionar_pieza = tk.LabelFrame(frame_der, text="Seleccionar Pieza")
frame_seleccionar_pieza.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

tk.Label(frame_seleccionar_pieza, text="Cantidad:").pack(anchor=tk.W, padx=5)
entry_cantidad_pieza = tk.Entry(frame_seleccionar_pieza)
entry_cantidad_pieza.pack(fill=tk.X, padx=5, pady=5)
entry_cantidad_pieza.insert(0, "1")  # Valor por defecto

tk.Button(frame_seleccionar_pieza, text="Seleccionar Pieza", command=seleccionar_pieza).pack(fill=tk.X, padx=5, pady=5)

# Sección para agregar piezas
frame_agregar_pieza = tk.LabelFrame(frame_der, text="Agregar Nueva Pieza")
frame_agregar_pieza.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

tk.Label(frame_agregar_pieza, text="Ancho (cm):").pack(anchor=tk.W, padx=5)
entry_ancho_pieza = tk.Entry(frame_agregar_pieza)
entry_ancho_pieza.pack(fill=tk.X, padx=5, pady=5)

tk.Label(frame_agregar_pieza, text="Alto (cm):").pack(anchor=tk.W, padx=5)
entry_alto_pieza = tk.Entry(frame_agregar_pieza)
entry_alto_pieza.pack(fill=tk.X, padx=5, pady=5)

tk.Button(frame_agregar_pieza, text="Agregar Pieza", command=agregar_pieza).pack(fill=tk.X, padx=5, pady=5)

# Pestaña de ejecución
tab_ejecucion = ttk.Frame(notebook)
notebook.add(tab_ejecucion, text="Ejecución")

# Frame para piezas seleccionadas
frame_seleccionadas = tk.LabelFrame(tab_ejecucion, text="Piezas Seleccionadas")
frame_seleccionadas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Tabla de piezas seleccionadas
tabla_piezas = ttk.Treeview(frame_seleccionadas, columns=("Pieza", "Cantidad"), show="headings")
tabla_piezas.heading("Pieza", text="Pieza (Ancho x Alto)")
tabla_piezas.heading("Cantidad", text="Cantidad")
tabla_piezas.column("Pieza", width=150)
tabla_piezas.column("Cantidad", width=100)
tabla_piezas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

# Frame para controles de ejecución
frame_controles = tk.Frame(tab_ejecucion)
frame_controles.pack(fill=tk.X, padx=10, pady=10)

# Botones de ejecución
btn_limpiar = tk.Button(frame_controles, text="Limpiar Selección", command=limpiar_seleccion)
btn_limpiar.pack(side=tk.LEFT, padx=5, pady=5)

btn_ejecutar = tk.Button(frame_controles, text="Ejecutar Algoritmo Genético", command=ejecutar_algoritmo_genetico)
btn_ejecutar.pack(side=tk.LEFT, padx=5, pady=5)

# Frame para configuración de algoritmo genético
frame_config_ag = tk.LabelFrame(tab_ejecucion, text="Configuración del Algoritmo Genético")
frame_config_ag.pack(fill=tk.X, padx=10, pady=10)

# Parámetros del AG
frame_params = tk.Frame(frame_config_ag)
frame_params.pack(fill=tk.X, padx=5, pady=5)

# Población
tk.Label(frame_params, text="Tamaño de Población:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
entry_pop_size = tk.Entry(frame_params, width=10)
entry_pop_size.grid(row=0, column=1, padx=5, pady=5)
entry_pop_size.insert(0, str(POP_SIZE))

# Generaciones
tk.Label(frame_params, text="Número de Generaciones:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
entry_generations = tk.Entry(frame_params, width=10)
entry_generations.grid(row=0, column=3, padx=5, pady=5)
entry_generations.insert(0, str(GENERATIONS))

# Tasa de mutación
tk.Label(frame_params, text="Tasa de Mutación:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
entry_mutation = tk.Entry(frame_params, width=10)
entry_mutation.grid(row=1, column=1, padx=5, pady=5)
entry_mutation.insert(0, str(MUTATION_RATE))

# Tasa de cruce
tk.Label(frame_params, text="Tasa de Cruce:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
entry_crossover = tk.Entry(frame_params, width=10)
entry_crossover.grid(row=1, column=3, padx=5, pady=5)
entry_crossover.insert(0, str(CROSSOVER_RATE))

# Función para actualizar los parámetros del AG
def actualizar_parametros_ag():
    global POP_SIZE, GENERATIONS, MUTATION_RATE, CROSSOVER_RATE
    try:
        POP_SIZE = int(entry_pop_size.get())
        GENERATIONS = int(entry_generations.get())
        MUTATION_RATE = float(entry_mutation.get())
        CROSSOVER_RATE = float(entry_crossover.get())
        
        # Validar valores
        if POP_SIZE < 10:
            messagebox.showwarning("Advertencia", "El tamaño de población debería ser al menos 10")
        if GENERATIONS < 10:
            messagebox.showwarning("Advertencia", "El número de generaciones debería ser al menos 10")
        if MUTATION_RATE < 0 or MUTATION_RATE > 1:
            messagebox.showwarning("Advertencia", "La tasa de mutación debe estar entre 0 y 1")
        if CROSSOVER_RATE < 0 or CROSSOVER_RATE > 1:
            messagebox.showwarning("Advertencia", "La tasa de cruce debe estar entre 0 y 1")
            
        messagebox.showinfo("Info", "Parámetros actualizados correctamente")
    except ValueError:
        messagebox.showerror("Error", "Ingrese valores numéricos válidos")

# Botón para actualizar parámetros
btn_actualizar_params = tk.Button(frame_params, text="Actualizar Parámetros", command=actualizar_parametros_ag)
btn_actualizar_params.grid(row=2, column=0, columnspan=4, padx=5, pady=5)

# Pestaña de resultados
tab_resultados = ttk.Frame(notebook)
notebook.add(tab_resultados, text="Resultados")

# Frame para mostrar estadísticas
frame_stats = tk.LabelFrame(tab_resultados, text="Estadísticas")
frame_stats.pack(fill=tk.X, padx=10, pady=10)

# Labels para estadísticas
lbl_area_total = tk.Label(frame_stats, text="Área total de lámina: -")
lbl_area_total.pack(anchor=tk.W, padx=5, pady=2)

lbl_area_usada = tk.Label(frame_stats, text="Área utilizada: -")
lbl_area_usada.pack(anchor=tk.W, padx=5, pady=2)

lbl_aprovechamiento = tk.Label(frame_stats, text="Aprovechamiento: -")
lbl_aprovechamiento.pack(anchor=tk.W, padx=5, pady=2)

lbl_piezas_colocadas = tk.Label(frame_stats, text="Piezas colocadas: -")
lbl_piezas_colocadas.pack(anchor=tk.W, padx=5, pady=2)

lbl_tiempo_ejecucion = tk.Label(frame_stats, text="Tiempo de ejecución: -")
lbl_tiempo_ejecucion.pack(anchor=tk.W, padx=5, pady=2)

# Frame para botones de acciones
frame_acciones = tk.Frame(tab_resultados)
frame_acciones.pack(fill=tk.X, padx=10, pady=10)

btn_guardar_resultado = tk.Button(frame_acciones, text="Guardar Resultado", state=tk.DISABLED)
btn_guardar_resultado.pack(side=tk.LEFT, padx=5, pady=5)

btn_exportar_imagen = tk.Button(frame_acciones, text="Exportar Imagen", state=tk.DISABLED)
btn_exportar_imagen.pack(side=tk.LEFT, padx=5, pady=5)

# Variable para almacenar el mejor individuo
mejor_individuo_global = None

# Función para exportar una imagen del resultado
def exportar_imagen(individuo):
    if not individuo:
        messagebox.showerror("Error", "No hay resultados para exportar")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, individuo.lamina_ancho)
    ax.set_ylim(0, individuo.lamina_alto)
    ax.set_title(f"Distribución optimizada - Aprovechamiento: {individuo.fitness*100:.2f}%")
    
    # Dibujar la lámina
    ax.add_patch(patches.Rectangle((0, 0), individuo.lamina_ancho, individuo.lamina_alto, 
                                   edgecolor='black', facecolor='none', linewidth=2))
    
    # Asignar colores diferentes a cada pieza
    cmap = plt.cm.get_cmap('tab20', len(individuo.distribucion))
    
    # Dibujar cada pieza
    for i, pieza in enumerate(individuo.distribucion):
        color = cmap(i % 20)
        ax.add_patch(patches.Rectangle((pieza.x, pieza.y), pieza.ancho, pieza.alto, 
                                       edgecolor='black', facecolor=color, alpha=0.7))
        
        # Agregar texto con dimensiones
        text_x = pieza.x + pieza.ancho/2
        text_y = pieza.y + pieza.alto/2
        ax.text(text_x, text_y, f"{pieza.ancho}x{pieza.alto}", ha='center', va='center', fontweight='bold')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Guardar la imagen
    try:
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(defaultextension=".png",
                                              filetypes=[("PNG files", "*.png"),
                                                         ("JPEG files", "*.jpg"),
                                                         ("PDF files", "*.pdf"),
                                                         ("All files", "*.*")])
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Info", f"Imagen guardada como {filename}")
    except Exception as e:
        messagebox.showerror("Error", f"Error al guardar la imagen: {str(e)}")
    
    plt.close()

# Inicializar la interfaz
actualizar_listas()
root.mainloop()