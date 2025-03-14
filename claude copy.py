import tkinter as tk
from tkinter import ttk, messagebox
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv
import os
import numpy as np
import time
from copy import deepcopy
import shutil
import threading
import queue
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Datos iniciales
laminas = []  # Lista de laminas (ancho, alto)
piezas = []  # Lista de piezas (ancho, alto)
piezas_seleccionadas = []
lamina_seleccionada = None

# Algoritmo Genético Configuración
POP_SIZE = 50  # Tamaño de la población
GENERATIONS = 100  # Número de generaciones
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
        self.rotada = rotada
        self.x = 0
        self.y = 0
    
    def rotar(self):
        self.ancho, self.alto = self.alto, self.ancho
        self.rotada = not self.rotada
        return self
    
    def area(self):
        return self.ancho * self.alto
    
    def copia(self):
        nueva_pieza = Pieza(self.ancho, self.alto, self.id, self.rotada)
        nueva_pieza.x = self.x
        nueva_pieza.y = self.y
        return nueva_pieza

class Individuo:
    def __init__(self, piezas_candidatas, lamina_ancho, lamina_alto):
        self.piezas = piezas_candidatas
        self.lamina_ancho = lamina_ancho
        self.lamina_alto = lamina_alto
        self.fitness = 0
        self.distribucion = []  # Almacenará la distribución final de piezas

    def _colocar_piezas(self, piezas):
        """Algoritmo de colocación mejorado con detección de residuos"""
        matriz = np.zeros((self.lamina_alto, self.lamina_ancho))
        colocadas = []
        areas_residuales = []
        
        # Ordenar piezas por área, de mayor a menor
        piezas_ordenadas = sorted(piezas, key=lambda x: x.area(), reverse=True)
        
        for p in piezas_ordenadas:
            mejor_x, mejor_y = -1, -1
            mejor_puntuacion = -1
            mejor_rotacion = False
            
            # Probar las dos orientaciones posibles
            for rotacion in [False, True]:
                p_ancho = p.alto if rotacion else p.ancho
                p_alto = p.ancho if rotacion else p.alto
                
                # Si la pieza no cabe en ninguna orientación, continuar con la siguiente
                if p_ancho > self.lamina_ancho or p_alto > self.lamina_alto:
                    continue
                
                # Buscar la mejor posición
                for y in range(self.lamina_alto - p_alto + 1):
                    for x in range(self.lamina_ancho - p_ancho + 1):
                        if self._cabe_en(matriz, x, y, p_ancho, p_alto):
                            # Puntuar esta posición según varios criterios
                            puntuacion = self._puntuar_posicion(matriz, x, y, p_ancho, p_alto)
                            
                            if puntuacion > mejor_puntuacion:
                                mejor_puntuacion = puntuacion
                                mejor_x = x
                                mejor_y = y
                                mejor_rotacion = rotacion
            
            # Si se encontró una posición válida
            if mejor_x >= 0 and mejor_y >= 0:
                p_copy = p.copia()
                
                # Aplicar rotación si es necesario
                if mejor_rotacion and p.ancho != p.alto:
                    p_copy.rotar()
                
                # Colocar la pieza
                p_copy.x = mejor_x
                p_copy.y = mejor_y
                matriz[mejor_y:mejor_y+p_copy.alto, mejor_x:mejor_x+p_copy.ancho] = 1
                colocadas.append(p_copy)
        
        # Detectar áreas residuales contiguas
        areas_residuales = self._detectar_areas_residuales(matriz)
        
        # Identificar piezas que no pudieron ser colocadas
        sobrantes = [p for p in piezas if not any(c.id == p.id for c in colocadas)]
        
        return matriz, colocadas, sobrantes, areas_residuales
    
    def calcular_fitness(self):
        self.distribuciones = []
        piezas_restantes = self.piezas.copy()
        area_total_lamina = self.lamina_ancho * self.lamina_alto
        area_total_piezas = sum(p.area() for p in self.piezas)
        area_utilizada_total = 0
        laminas_usadas = 0
        
        # Coeficientes para equilibrar objetivos
        PESO_APROVECHAMIENTO = 0.7  # Mayor peso al aprovechamiento de área
        PESO_LAMINAS = 0.2  # Penalización por usar muchas láminas
        PESO_RESIDUOS = 0.1  # Recompensa por residuos grandes
        
        while piezas_restantes:
            matriz, colocadas, sobrantes, areas_residuales = self._colocar_piezas(piezas_restantes)
            area_utilizada = sum(p.area() for p in colocadas)
            area_utilizada_total += area_utilizada
            
            # Calcular estadísticas de residuos
            if areas_residuales:
                area_residual_max = max(areas_residuales) if areas_residuales else 0
                area_residual_promedio = sum(areas_residuales) / len(areas_residuales) if areas_residuales else 0
            else:
                area_residual_max = 0
                area_residual_promedio = 0
            
            # Almacenar datos de esta distribución
            self.distribuciones.append({
                'piezas': colocadas,
                'matriz': matriz,
                'area_utilizada': area_utilizada,
                'areas_residuales': areas_residuales,
                'area_residual_max': area_residual_max
            })
            
            piezas_restantes = sobrantes
            laminas_usadas += 1
        
        # Factor de aprovechamiento (área utilizada / área total de láminas)
        aprovechamiento = area_utilizada_total / (area_total_lamina * laminas_usadas) if laminas_usadas > 0 else 0
        
        # Factor de eficiencia de láminas (reducir número de láminas)
        factor_laminas = min(1.0, area_total_piezas / (area_total_lamina * laminas_usadas)) if laminas_usadas > 0 else 0
        
        # Factor de calidad de residuos (preferir residuos grandes a muchos pequeños)
        residuos_totales = [res for dist in self.distribuciones for res in dist.get('areas_residuales', [])]
        max_residuo = max(residuos_totales) if residuos_totales else 0
        factor_residuos = max_residuo / area_total_lamina if area_total_lamina > 0 else 0
        
        # Fitness combinado
        self.fitness = (PESO_APROVECHAMIENTO * aprovechamiento + 
                    PESO_LAMINAS * factor_laminas + 
                    PESO_RESIDUOS * factor_residuos)
        
        return self.fitness
    
    def _puntuar_posicion(self, matriz, x, y, ancho, alto):
        """Evalúa qué tan buena es una posición para colocar una pieza"""
        # Puntuación base - preferir esquinas y bordes
        puntuacion = 0
        
        # Bonificación si la pieza toca el borde izquierdo
        if x == 0:
            puntuacion += 3
        
        # Bonificación si la pieza toca el borde inferior
        if y == 0:
            puntuacion += 3
        
        # Bonificación si la pieza toca el borde derecho
        if x + ancho == self.lamina_ancho:
            puntuacion += 3
        
        # Bonificación si la pieza toca el borde superior
        if y + alto == self.lamina_alto:
            puntuacion += 3
        
        # Bonificación por piezas adyacentes - máxima compactación
        # Lado izquierdo
        if x > 0:
            for dy in range(alto):
                if y + dy < self.lamina_alto and matriz[y + dy, x - 1] == 1:
                    puntuacion += 1
                    
        # Lado derecho
        if x + ancho < self.lamina_ancho:
            for dy in range(alto):
                if y + dy < self.lamina_alto and matriz[y + dy, x + ancho] == 1:
                    puntuacion += 1
                    
        # Lado inferior
        if y > 0:
            for dx in range(ancho):
                if x + dx < self.lamina_ancho and matriz[y - 1, x + dx] == 1:
                    puntuacion += 1
                    
        # Lado superior
        if y + alto < self.lamina_alto:
            for dx in range(ancho):
                if x + dx < self.lamina_ancho and matriz[y + alto, x + dx] == 1:
                    puntuacion += 1
        
        return puntuacion
    
    def _detectar_areas_residuales(self, matriz):
        """Detecta y mide las áreas residuales continuas en la matriz"""
        visitados = np.zeros_like(matriz)
        areas_residuales = []
        
        for y in range(matriz.shape[0]):
            for x in range(matriz.shape[1]):
                if matriz[y, x] == 0 and visitados[y, x] == 0:
                    # Encontramos un nuevo residuo, aplicar flood fill
                    area = self._medir_area_residual(matriz, visitados, x, y)
                    
                    # Solo considerar residuos que tengan un tamaño mínimo útil
                    if area >= 25:  # Por ejemplo, residuos de al menos 5x5 cm
                        areas_residuales.append(area)
        
        return areas_residuales
    
    def _medir_area_residual(self, matriz, visitados, x, y):
        """Mide un área residual continua usando flood fill"""
        if x < 0 or y < 0 or x >= matriz.shape[1] or y >= matriz.shape[0]:
            return 0
        
        if matriz[y, x] == 1 or visitados[y, x] == 1:
            return 0
        
        # Marcar como visitado
        visitados[y, x] = 1
        area = 1
        
        # Visitar celdas adyacentes
        area += self._medir_area_residual(matriz, visitados, x+1, y)
        area += self._medir_area_residual(matriz, visitados, x-1, y)
        area += self._medir_area_residual(matriz, visitados, x, y+1)
        area += self._medir_area_residual(matriz, visitados, x, y-1)
        
        return area
    
    def _cabe_en(self, matriz, x, y, ancho, alto):
        """Verificación precisa de si una pieza cabe en una posición"""
        if x + ancho > matriz.shape[1] or y + alto > matriz.shape[0]:
            return False
        return np.all(matriz[y:y+alto, x:x+ancho] == 0)

# Funciones para el algoritmo genético
def crear_poblacion_inicial(piezas_raw, lamina_ancho, lamina_alto):
    """Crea una población inicial de individuos"""
    poblacion = []
    
    for _ in range(POP_SIZE):
        # Crear lista de piezas para este individuo
        piezas_candidatas = []
        for pieza_raw, cantidad in piezas_raw:
            for i in range(cantidad):
                # Decidir aleatoriamente si rotar la pieza
                rotada = random.random() > 0.5
                pieza = Pieza(pieza_raw[0], pieza_raw[1], id=len(piezas_candidatas))
                if rotada and pieza.ancho != pieza.alto:  # No rotar si es cuadrada
                    pieza.rotar()
                piezas_candidatas.append(pieza)
        
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
    """Cruce de dos individuos"""
    if random.random() > CROSSOVER_RATE:
        return padre1, padre2
    
    # Elegir punto de cruce
    punto_cruce = random.randint(1, len(padre1.piezas) - 1)
    
    # Crear nuevos individuos
    hijo1_piezas = padre1.piezas[:punto_cruce] + padre2.piezas[punto_cruce:]
    hijo2_piezas = padre2.piezas[:punto_cruce] + padre1.piezas[punto_cruce:]
    
    # Crear nuevos individuos
    hijo1 = Individuo(hijo1_piezas, padre1.lamina_ancho, padre1.lamina_alto)
    hijo2 = Individuo(hijo2_piezas, padre2.lamina_ancho, padre2.lamina_alto)
    
    return hijo1, hijo2

def mutacion(individuo):
    """Mutación de un individuo"""
    for i in range(len(individuo.piezas)):
        if random.random() < MUTATION_RATE:
            # Tipos de mutación:
            # 1. Cambiar posición
            # 2. Rotar pieza
            tipo_mutacion = random.randint(1, 2)
            
            if tipo_mutacion == 1:
                # Cambiar posición (intercambiar con otra pieza)
                j = random.randint(0, len(individuo.piezas) - 1)
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
    
    for generacion in range(GENERATIONS):
        # Selección
        seleccionados = seleccion_torneo(poblacion)
        
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
        for i in range(POP_SIZE):
            nueva_poblacion[i] = mutacion(nueva_poblacion[i])
        
        # Calcular fitness de la nueva población
        for individuo in nueva_poblacion:
            individuo.calcular_fitness()
        
        # Elitismo (mantener al mejor individuo)
        mejor_actual = max(nueva_poblacion, key=lambda x: x.fitness)
        if mejor_actual.fitness > mejor_individuo.fitness:
            mejor_individuo = mejor_actual
        else:
            # Reemplazar el peor individuo por el mejor de la generación anterior
            peor_idx = min(range(POP_SIZE), key=lambda i: nueva_poblacion[i].fitness)
            nueva_poblacion[peor_idx] = mejor_individuo
        
        # Actualizar población
        poblacion = nueva_poblacion
        
        # Registrar el mejor fitness
        mejores_fitness.append(mejor_individuo.fitness)
        
        # Mostrar progreso
        if generacion % 10 == 0:
            print(f"Generación {generacion}: Mejor fitness = {mejor_individuo.fitness}")
    
    print(f"Mejor fitness final: {mejor_individuo.fitness}")
    return mejor_individuo, mejores_fitness

# Expandir las piezas seleccionadas para el algoritmo genético
def expandir_piezas(piezas_seleccionadas):
    todas_piezas = []
    for pieza, cantidad in piezas_seleccionadas:
        for i in range(cantidad):
            todas_piezas.append(Pieza(pieza[0], pieza[1], id=len(todas_piezas)))
    return todas_piezas

# Algoritmo Genético - Ejecución
def ejecutar_algoritmo_genetico():
    global piezas_seleccionadas, lamina_seleccionada, mejor_individuo_global
    
    if not lamina_seleccionada:
        messagebox.showerror("Error", "Seleccione una lámina primero")
        return
    
    if not piezas_seleccionadas:
        messagebox.showerror("Error", "Seleccione al menos una pieza")
        return
    
    # Configurar ventana de progreso
    ventana_progreso = tk.Toplevel(root)
    ventana_progreso.title("Progreso del Algoritmo Genético")
    ventana_progreso.geometry("500x300")
    
    # Frame principal de progreso
    frame_progreso = ttk.Frame(ventana_progreso)
    frame_progreso.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    lbl_progreso = ttk.Label(frame_progreso, text="Inicializando algoritmo genético...")
    lbl_progreso.pack(pady=5)
    
    barra_progreso = ttk.Progressbar(frame_progreso, length=400, mode='determinate')
    barra_progreso.pack(pady=5)
    
    # Frame para detalles
    frame_detalles = ttk.LabelFrame(frame_progreso, text="Detalles de ejecución")
    frame_detalles.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    lbl_generacion = ttk.Label(frame_detalles, text="Generación: 0/0")
    lbl_generacion.pack(anchor=tk.W, padx=5, pady=2)
    
    lbl_fitness = ttk.Label(frame_detalles, text="Mejor fitness: 0.0")
    lbl_fitness.pack(anchor=tk.W, padx=5, pady=2)
    
    lbl_laminas = ttk.Label(frame_detalles, text="Láminas utilizadas: 0")
    lbl_laminas.pack(anchor=tk.W, padx=5, pady=2)
    
    lbl_aprovechamiento = ttk.Label(frame_detalles, text="Aprovechamiento: 0.0%")
    lbl_aprovechamiento.pack(anchor=tk.W, padx=5, pady=2)
    
    lbl_residuos = ttk.Label(frame_detalles, text="Residuos óptimos: 0")
    lbl_residuos.pack(anchor=tk.W, padx=5, pady=2)
    
    lbl_tiempo = ttk.Label(frame_detalles, text="Tiempo: 0s")
    lbl_tiempo.pack(anchor=tk.W, padx=5, pady=2)
    
    # Cola para comunicación entre hilos
    progreso_queue = queue.Queue()
    start_time = time.time()
    
    def actualizar_interfaz():
        while not progreso_queue.empty():
            tipo, valor = progreso_queue.get()
            if tipo == 'progreso':
                generacion, total, fitness, laminas, aprovechamiento, residuos = valor
                # Actualizar barra y labels
                barra_progreso['value'] = (generacion / total) * 100
                lbl_generacion.config(text=f"Generación: {generacion}/{total}")
                lbl_fitness.config(text=f"Mejor fitness: {fitness:.4f}")
                lbl_laminas.config(text=f"Láminas utilizadas: {laminas}")
                lbl_aprovechamiento.config(text=f"Aprovechamiento: {aprovechamiento:.2f}%")
                lbl_residuos.config(text=f"Residuos óptimos: {residuos}")
                lbl_tiempo.config(text=f"Tiempo: {time.time()-start_time:.1f}s")
            elif tipo == 'fin':
                # Mostrar resultados finales y activar botones
                mejor_individuo_global = valor
                
                # Activar botones de resultado
                btn_guardar_resultado.config(state=tk.NORMAL)
                btn_exportar_imagen.config(state=tk.NORMAL)
                
                # Mostrar resultado
                mostrar_resultado(mejor_individuo_global)
                
                # Cerrar ventana de progreso
                ventana_progreso.destroy()
                
                # Mostrar mensaje de éxito
                messagebox.showinfo("Éxito", 
                                   f"Optimización completada.\n"
                                   f"Láminas utilizadas: {len(mejor_individuo_global.distribuciones)}\n"
                                   f"Aprovechamiento: {mejor_individuo_global.fitness*100:.2f}%\n"
                                   f"Tiempo: {time.time()-start_time:.1f}s")
                
            elif tipo == 'error':
                messagebox.showerror("Error", valor)
                ventana_progreso.destroy()
                
        root.after(100, actualizar_interfaz)
    
    def ejecutar_en_hilo():
        try:
            inicio = time.time()
            
            # Versión mejorada del algoritmo genético
            mejor_individuo, historico = algoritmo_genetico_mejorado(
                piezas_seleccionadas,
                lamina_seleccionada[0],
                lamina_seleccionada[1],
                progreso_queue
            )
            
            # Generar imágenes finales
            generar_imagenes_finales(mejor_individuo)
            
            # Notificar finalización
            progreso_queue.put(('fin', mejor_individuo))
            
        except Exception as e:
            import traceback
            progreso_queue.put(('error', f"{str(e)}\n{traceback.format_exc()}"))
    
    # Iniciar hilos
    threading.Thread(target=ejecutar_en_hilo, daemon=True).start()
    root.after(100, actualizar_interfaz)

def generar_imagenes_intermedias(individuo, generacion):
    carpeta = "resultados_corte"
    
    # Crear subcarpeta para cada generación
    subcarpeta = f"{carpeta}/gen_{generacion}"
    if not os.path.exists(subcarpeta):
        os.makedirs(subcarpeta)
    
    # Generar una imagen por cada lámina
    for idx, distribucion in enumerate(individuo.distribuciones):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(0, individuo.lamina_ancho)
        ax.set_ylim(0, individuo.lamina_alto)
        ax.set_title(f"Lámina {idx+1} - Gen {generacion}")
        
        # Dibujar lámina
        ax.add_patch(patches.Rectangle(
            (0, 0), individuo.lamina_ancho, individuo.lamina_alto,
            edgecolor='black', facecolor='none', linewidth=2
        ))
        
        # Colormap para piezas
        cmap = plt.cm.get_cmap('tab20', len(distribucion['piezas']))
        
        # Dibujar piezas
        for i, pieza in enumerate(distribucion['piezas']):
            color = cmap(i % 20)
            ax.add_patch(patches.Rectangle(
                (pieza.x, pieza.y), pieza.ancho, pieza.alto,
                edgecolor='black', facecolor=color, alpha=0.7
            ))
            
            # Añadir texto con dimensiones
            ax.text(pieza.x + pieza.ancho/2, pieza.y + pieza.alto/2, 
                    f"{pieza.ancho}x{pieza.alto}", 
                    ha='center', va='center', fontsize=8)
        
        # Mostrar áreas residuales importantes
        if 'areas_residuales' in distribucion:
            visitados = np.zeros((individuo.lamina_alto, individuo.lamina_ancho))
            for y in range(individuo.lamina_alto):
                for x in range(individuo.lamina_ancho):
                    if distribucion['matriz'][y, x] == 0 and visitados[y, x] == 0:
                        area, coords = detectar_area_residual(distribucion['matriz'], x, y, visitados)
                        if area >= 25:  # Solo mostrar residuos significativos
                            min_x, min_y, max_x, max_y = coords
                            ancho = max_x - min_x + 1
                            alto = max_y - min_y + 1
                            ax.add_patch(patches.Rectangle(
                                (min_x, min_y), ancho, alto,
                                edgecolor='red', facecolor='none', linestyle='--', linewidth=1
                            ))
                            ax.text(min_x + ancho/2, min_y + alto/2, 
                                    f"R: {area}", 
                                    ha='center', va='center', fontsize=7, color='red')
        
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.savefig(f"{subcarpeta}/lamina_{idx+1}.png", dpi=150)
        plt.close()

def algoritmo_genetico_mejorado(piezas_seleccionadas, lamina_ancho, lamina_alto, progreso_queue):
    # Leer parámetros desde la interfaz
    global POP_SIZE, GENERATIONS, MUTATION_RATE, CROSSOVER_RATE
    POP_SIZE = int(entry_pop_size.get()) if entry_pop_size.get() else 50
    GENERATIONS = int(entry_generations.get()) if entry_generations.get() else 30
    MUTATION_RATE = float(entry_mutation.get()) if entry_mutation.get() else 0.1
    CROSSOVER_RATE = float(entry_crossover.get()) if entry_crossover.get() else 0.8
    
    # Preparar carpeta de resultados
    carpeta_resultados = "resultados_corte"
    if os.path.exists(carpeta_resultados):
        shutil.rmtree(carpeta_resultados)
    os.makedirs(carpeta_resultados)
    
    # Expandir piezas
    todas_piezas = expandir_piezas(piezas_seleccionadas)
    area_total_piezas = sum(p.ancho * p.alto for p in todas_piezas)
    
    # Inicializar población
    poblacion = crear_poblacion_inicial(todas_piezas, lamina_ancho, lamina_alto)
    
    # Iniciar evolución
    mejor_global = None
    historico_fitness = []
    
    for generacion in range(GENERATIONS):
        # Evaluar población
        for individuo in poblacion:
            individuo.calcular_fitness()
        
        # Ordenar por fitness
        poblacion.sort(key=lambda x: x.fitness, reverse=True)
        
        # Obtener mejor individuo de esta generación
        mejor_actual = poblacion[0]
        
        # Actualizar mejor global si es necesario
        if mejor_global is None or mejor_actual.fitness > mejor_global.fitness:
            mejor_global = deepcopy(mejor_actual)
        
        # Guardar historial
        historico_fitness.append(mejor_actual.fitness)
        
        # Calcular estadísticas para reportar progreso
        laminas_usadas = len(mejor_actual.distribuciones)
        area_utilizada = sum(dist.get('area_utilizada', 0) for dist in mejor_actual.distribuciones)
        aprovechamiento = (area_utilizada / (lamina_ancho * lamina_alto * laminas_usadas)) * 100
        residuos_grandes = sum(1 for dist in mejor_actual.distribuciones 
                              for area in dist.get('areas_residuales', []) if area > 25)
        
        # Reportar progreso
        progreso_queue.put(('progreso', (
            generacion+1,
            GENERATIONS,
            mejor_actual.fitness,
            laminas_usadas,
            aprovechamiento,
            residuos_grandes
        )))
        
        # Generar imágenes intermedias cada 5 generaciones
        if generacion % 5 == 0:
            generar_imagenes_intermedias(mejor_actual, generacion)
        
        # Crear siguiente generación (excepto en la última iteración)
        if generacion < GENERATIONS - 1:
            nueva_poblacion = []
            
            # Elitismo: mantener los mejores individuos
            elite_size = max(1, int(POP_SIZE * 0.1))
            nueva_poblacion.extend(poblacion[:elite_size])
            
            # Selección por torneo
            seleccionados = seleccion_torneo(poblacion)
            
            # Cruce y mutación
            for i in range(0, len(seleccionados), 2):
                if i+1 < len(seleccionados):
                    padre1 = seleccionados[i]
                    padre2 = seleccionados[i+1]
                    
                    # Cruce
                    hijo1, hijo2 = cruce_mejorado(padre1, padre2)
                    
                    # Mutación
                    hijo1 = mutacion_mejorada(hijo1)
                    hijo2 = mutacion_mejorada(hijo2)
                    
                    nueva_poblacion.extend([hijo1, hijo2])
            
            # Mantener tamaño de población constante
            poblacion = nueva_poblacion[:POP_SIZE]
    
    # Generar imágenes finales
    generar_imagenes_finales(mejor_global)
    
    return mejor_global, historico_fitness

# Función para mostrar el resultado gráfico
def mostrar_resultado(individuo):
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
    plt.savefig("resultado_optimizacion.png")
    plt.show()

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
            writer.writerow(["Pieza", "Ancho", "Alto", "Posicion X", "Posicion Y", "Rotada"])
            for pieza in individuo.distribucion:
                writer.writerow([pieza.id, pieza.ancho, pieza.alto, pieza.x, pieza.y, pieza.rotada])
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

# Función mejorada para ejecutar el algoritmo genético
def ejecutar_algoritmo_genetico():
    global mejor_individuo_global
    
    if not lamina_seleccionada:
        messagebox.showerror("Error", "Seleccione una lámina")
        return
    if not piezas_seleccionadas:
        messagebox.showerror("Error", "Seleccione al menos una pieza")
        return
    
    # Actualizar parámetros
    try:
        actualizar_parametros_ag()
    except:
        # Si falla la actualización, usar valores predeterminados
        pass
    
    print(f"Ejecutando algoritmo genético con parámetros:")
    print(f"Población: {POP_SIZE}, Generaciones: {GENERATIONS}")
    print(f"Tasa de mutación: {MUTATION_RATE}, Tasa de cruce: {CROSSOVER_RATE}")
    
    # Expandir piezas seleccionadas para el algoritmo
    piezas_expandidas = []
    total_piezas = 0
    for pieza, cantidad in piezas_seleccionadas:
        for i in range(cantidad):
            piezas_expandidas.append(Pieza(pieza[0], pieza[1], id=len(piezas_expandidas)))
            total_piezas += 1
    
    # Obtener dimensiones de la lámina
    lamina_ancho = lamina_seleccionada[0]
    lamina_alto = lamina_seleccionada[1]
    
    # Verificar que las piezas quepan en la lámina
    for pieza in piezas_expandidas:
        if pieza.ancho > lamina_ancho and pieza.alto > lamina_ancho:
            messagebox.showerror("Error", f"La pieza {pieza.ancho}x{pieza.alto} no cabe en la lámina incluso rotada")
            return
        if pieza.alto > lamina_alto and pieza.ancho > lamina_alto:
            messagebox.showerror("Error", f"La pieza {pieza.ancho}x{pieza.alto} no cabe en la lámina incluso rotada")
            return
    
    # Verificar si el área total de las piezas supera el área de la lámina
    area_lamina = lamina_ancho * lamina_alto
    area_total_piezas = sum(pieza.area() for pieza in piezas_expandidas)
    if area_total_piezas > area_lamina:
        resultado = messagebox.askquestion("Advertencia", 
                                          f"El área total de las piezas ({area_total_piezas} cm²) supera el área de la lámina ({area_lamina} cm²). " +
                                          "Es posible que no todas las piezas puedan colocarse. ¿Desea continuar?")
        if resultado != 'yes':
            return
    
    # Mostrar barra de progreso
    ventana_progreso = tk.Toplevel(root)
    ventana_progreso.title("Progreso del Algoritmo Genético")
    ventana_progreso.geometry("400x150")
    
    tk.Label(ventana_progreso, text="Ejecutando algoritmo genético...").pack(pady=10)
    
    barra_progreso = ttk.Progressbar(ventana_progreso, length=300, mode='determinate')
    barra_progreso.pack(pady=10)
    
    lbl_estado = tk.Label(ventana_progreso, text="Inicializando...")
    lbl_estado.pack(pady=10)
    
    # Función para actualizar la barra de progreso
    def actualizar_progreso(generacion, fitness):
        progreso = (generacion / GENERATIONS) * 100
        barra_progreso['value'] = progreso
        lbl_estado.config(text=f"Generación {generacion}/{GENERATIONS} - Fitness: {fitness:.4f}")
        ventana_progreso.update()
    
    # Tiempo de inicio
    import time
    tiempo_inicio = time.time()
    
    # Ejecutar el algoritmo genético
    try:
        # Modificar la función algoritmo_genetico para que actualice la barra de progreso
        def algoritmo_genetico_con_progreso(piezas_seleccionadas, lamina_ancho, lamina_alto):
            # Crear población inicial
            poblacion = crear_poblacion_inicial(piezas_seleccionadas, lamina_ancho, lamina_alto)
            
            # Evolución
            mejor_individuo = max(poblacion, key=lambda x: x.fitness)
            mejores_fitness = [mejor_individuo.fitness]
            
            for generacion in range(GENERATIONS):
                # Selección
                seleccionados = seleccion_torneo(poblacion)
                
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
                for i in range(POP_SIZE):
                    nueva_poblacion[i] = mutacion(nueva_poblacion[i])
                
                # Calcular fitness de la nueva población
                for individuo in nueva_poblacion:
                    individuo.calcular_fitness()
                
                # Elitismo (mantener al mejor individuo)
                mejor_actual = max(nueva_poblacion, key=lambda x: x.fitness)
                if mejor_actual.fitness > mejor_individuo.fitness:
                    mejor_individuo = mejor_actual
                else:
                    # Reemplazar el peor individuo por el mejor de la generación anterior
                    peor_idx = min(range(POP_SIZE), key=lambda i: nueva_poblacion[i].fitness)
                    nueva_poblacion[peor_idx] = mejor_individuo
                
                # Actualizar población
                poblacion = nueva_poblacion
                
                # Registrar el mejor fitness
                mejores_fitness.append(mejor_individuo.fitness)
                
                # Actualizar barra de progreso cada 5 generaciones
                if generacion % 5 == 0 or generacion == GENERATIONS - 1:
                    actualizar_progreso(generacion + 1, mejor_individuo.fitness)
            
            return mejor_individuo, mejores_fitness
        
        # Ejecutar el algoritmo
        mejor_individuo, historico_fitness = algoritmo_genetico_con_progreso(piezas_seleccionadas, lamina_ancho, lamina_alto)
        mejor_individuo_global = mejor_individuo
        
        # Calcular tiempo de ejecución
        tiempo_ejecucion = time.time() - tiempo_inicio
        
        # Cerrar ventana de progreso
        ventana_progreso.destroy()
        
        # Mostrar resultados
        messagebox.showinfo("Info", f"Optimización completada. Aprovechamiento: {mejor_individuo.fitness*100:.2f}%")
        
        # Actualizar estadísticas
        area_lamina = lamina_ancho * lamina_alto
        area_usada = sum(pieza.ancho * pieza.alto for pieza in mejor_individuo.distribucion)
        piezas_colocadas = len(mejor_individuo.distribucion)
        
        lbl_area_total.config(text=f"Área total de lámina: {area_lamina} cm²")
        lbl_area_usada.config(text=f"Área utilizada: {area_usada} cm²")
        lbl_aprovechamiento.config(text=f"Aprovechamiento: {mejor_individuo.fitness*100:.2f}%")
        lbl_piezas_colocadas.config(text=f"Piezas colocadas: {piezas_colocadas} de {total_piezas}")
        lbl_tiempo_ejecucion.config(text=f"Tiempo de ejecución: {tiempo_ejecucion:.2f} segundos")
        
        # Habilitar botones de acción
        btn_guardar_resultado.config(state=tk.NORMAL, command=lambda: guardar_resultado(mejor_individuo))
        btn_exportar_imagen.config(state=tk.NORMAL, command=lambda: exportar_imagen(mejor_individuo))
        
        # Mostrar resultado gráfico
        mostrar_resultado(mejor_individuo)
        
        # Mostrar gráfica de evolución
        mostrar_grafica_evolucion(historico_fitness)
        
        # Cambiar a la pestaña de resultados
        notebook.select(tab_resultados)
        
    except Exception as e:
        ventana_progreso.destroy()
        messagebox.showerror("Error", f"Error durante la ejecución: {str(e)}")
        raise e

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