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
        self.en_lamina = 0  # Nueva propiedad para trackear lámina
    
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
        nueva_pieza.en_lamina = self.en_lamina
        return nueva_pieza

class Individuo:
    def __init__(self, piezas_candidatas, lamina_ancho, lamina_alto):
        self.piezas = piezas_candidatas
        self.lamina_ancho = lamina_ancho
        self.lamina_alto = lamina_alto
        self.fitness = 0.0
        self.distribuciones = []  # Lista de distribuciones por lámina
        
    def calcular_fitness(self):
        self.distribuciones = []
        piezas_restantes = self.piezas.copy()
        area_total = sum(p.area() for p in self.piezas)
        area_utilizada = 0
        
        while piezas_restantes:
            matriz, colocadas, sobrantes = self._colocar_piezas(piezas_restantes)
            self.distribuciones.append({'piezas': colocadas})
            area_utilizada += sum(p.area() for p in colocadas)
            piezas_restantes = sobrantes
        
        # Cálculo de fitness con penalización por múltiples láminas
        utilizacion = area_utilizada / area_total
        self.fitness = utilizacion - (0.1 * len(self.distribuciones))
        return self.fitness
    
    def _colocar_piezas(self, piezas):
        """Algoritmo de colocación mejorado"""
        matriz = np.zeros((self.lamina_alto, self.lamina_ancho))
        colocadas = []
        
        for p in sorted(piezas, key=lambda x: x.area(), reverse=True):
            for y in range(self.lamina_alto - p.alto + 1):
                for x in range(self.lamina_ancho - p.ancho + 1):
                    if self._cabem(matriz, x, y, p.ancho, p.alto):
                        matriz[y:y+p.alto, x:x+p.ancho] = 1
                        p_copy = p.copia()
                        p_copy.x = x
                        p_copy.y = y
                        colocadas.append(p_copy)
                        break
                else:
                    continue
                break
        
        sobrantes = [p for p in piezas if p not in [c.id for c in colocadas]]
        return matriz, colocadas, sobrantes
    
    def _cabem(self, matriz, x, y, ancho, alto):
        """Verificación precisa de espacio"""
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

def algoritmo_genetico(piezas_seleccionadas, lamina_ancho, lamina_alto, progreso_queue):
    # Leer parámetros desde la interfaz
    global POP_SIZE, GENERATIONS, MUTATION_RATE, CROSSOVER_RATE
    POP_SIZE = int(entry_pop_size.get()) if entry_pop_size.get() else 50
    GENERATIONS = int(entry_generations.get()) if entry_generations.get() else 30
    MUTATION_RATE = float(entry_mutation.get()) if entry_mutation.get() else 0.1
    CROSSOVER_RATE = float(entry_crossover.get()) if entry_crossover.get() else 0.8
    
    # Mensajes de inicio en consola
    print("\n=== INICIO ALGORITMO GENÉTICO ===")
    print(f"Población: {POP_SIZE} | Generaciones: {GENERATIONS}")
    print(f"Tamaño lámina: {lamina_ancho}x{lamina_alto}")
    print("="*50 + "\n")
    
    # Configurar carpeta de resultados
    carpeta_resultados = "resultados_lamina"
    if os.path.exists(carpeta_resultados):
        shutil.rmtree(carpeta_resultados)
    os.makedirs(carpeta_resultados)
    
    # Inicializar población
    poblacion = []
    for _ in range(POP_SIZE):
        piezas = expandir_piezas(piezas_seleccionadas)
        random.shuffle(piezas)
        individuo = Individuo(piezas, lamina_ancho, lamina_alto)
        individuo.calcular_fitness()
        poblacion.append(individuo)
    
    mejor_global = None
    historico_fitness = []
    start_time = time.time()
    
    # Evolución
    for generacion in range(GENERATIONS):
        # Evaluación
        for individuo in poblacion:
            individuo.calcular_fitness()
        
        # Selección por torneo
        seleccionados = []
        for _ in range(POP_SIZE):
            participantes = random.sample(poblacion, 3)
            ganador = max(participantes, key=lambda x: x.fitness)
            seleccionados.append(ganador)
        
        # Cruce
        descendencia = []
        for i in range(0, POP_SIZE, 2):
            padre1 = seleccionados[i]
            padre2 = seleccionados[i+1] if i+1 < POP_SIZE else seleccionados[i]
            
            if random.random() < CROSSOVER_RATE:
                punto_cruce = random.randint(1, len(padre1.piezas)-1)
                hijo1_piezas = padre1.piezas[:punto_cruce] + padre2.piezas[punto_cruce:]
                hijo2_piezas = padre2.piezas[:punto_cruce] + padre1.piezas[punto_cruce:]
                descendencia.extend([
                    Individuo(hijo1_piezas, lamina_ancho, lamina_alto),
                    Individuo(hijo2_piezas, lamina_ancho, lamina_alto)
                ])
            else:
                descendencia.extend([padre1, padre2])
        
        # Mutación
        for individuo in descendencia:
            if random.random() < MUTATION_RATE and len(individuo.piezas) > 1:
                i, j = random.sample(range(len(individuo.piezas)), 2)
                individuo.piezas[i], individuo.piezas[j] = individuo.piezas[j], individuo.piezas[i]
        
        # Actualizar población
        poblacion = sorted(descendencia, key=lambda x: x.fitness, reverse=True)[:POP_SIZE]
        
        # Actualizar mejor global
        mejor_actual = max(poblacion, key=lambda x: x.fitness)
        if mejor_global is None or mejor_actual.fitness > mejor_global.fitness:
            mejor_global = deepcopy(mejor_actual)
        
        historico_fitness.append(mejor_actual.fitness)
        
        # Mostrar progreso
        print(f"Gen {generacion+1:03d}/{GENERATIONS} | "
              f"Fitness: {mejor_actual.fitness:.2f} | "
              f"Láminas: {len(mejor_actual.distribuciones)} | "
              f"Tiempo: {time.time()-start_time:.1f}s")
        
        # Generar imágenes
        if generacion % 5 == 0:
            generar_imagenes_finales(mejor_actual, generacion)
    
    # Resultados finales
    print("\n=== RESULTADOS FINALES ===")
    print(f"Mejor fitness: {mejor_global.fitness:.2f}")
    print(f"Láminas usadas: {len(mejor_global.distribuciones)}")
    print(f"Tiempo total: {time.time() - start_time:.1f} segundos")

    progreso_queue.put(('progreso', (
            generacion+1,
            GENERATIONS,
            mejor_actual.fitness,
            len(mejor_actual.distribuciones)
        )))
    
    return mejor_global, historico_fitness


# Expandir las piezas seleccionadas para el algoritmo genético
def expandir_piezas(piezas_seleccionadas):
    todas_piezas = []
    for pieza, cantidad in piezas_seleccionadas:
        for i in range(cantidad):
            todas_piezas.append(Pieza(pieza[0], pieza[1], id=len(todas_piezas)))
    return todas_piezas

def generar_imagenes_finales(individuo, sufijo=""):
    carpeta = "resultados_lamina"
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)
    
    for idx, lamina in enumerate(individuo.distribuciones):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"Lámina {idx+1} - {sufijo}")
        ax.set_xlim(0, individuo.lamina_ancho)
        ax.set_ylim(0, individuo.lamina_alto)
        
        # Dibujar lámina
        ax.add_patch(patches.Rectangle(
            (0, 0), individuo.lamina_ancho, individuo.lamina_alto,
            edgecolor='black', facecolor='none', linewidth=2
        ))
        
        # Dibujar piezas
        for p in lamina['piezas']:
            color = np.random.rand(3,)
            ax.add_patch(patches.Rectangle(
                (p.x, p.y), p.ancho, p.alto,
                edgecolor='black', facecolor=color, alpha=0.7
            ))
            ax.text(p.x + p.ancho/2, p.y + p.alto/2, 
                    f"{p.ancho}x{p.alto}", 
                    ha='center', va='center', fontsize=8)
        
        plt.savefig(f"{carpeta}/lamina_{idx+1}_{sufijo}.png", dpi=150)
        plt.close()

def mostrar_resultado(individuo):
    # Limpiar pestaña de resultados
    for widget in tab_resultados.winfo_children():
        widget.destroy()
    
    # Verificar si hay resultados
    if not individuo.distribuciones:
        messagebox.showwarning("Advertencia", "No hay distribución para mostrar")
        return
    
    # Crear frame para estadísticas
    frame_stats = ttk.Frame(tab_resultados)
    frame_stats.pack(fill=tk.X, padx=10, pady=10)
    
    # Calcular métricas
    area_total = sum(p.area() for dist in individuo.distribuciones for p in dist['piezas'])
    laminas_usadas = len(individuo.distribuciones)
    
    # Mostrar estadísticas
    ttk.Label(frame_stats, text=f"Fitness: {individuo.fitness:.2f}").pack(anchor=tk.W)
    ttk.Label(frame_stats, text=f"Láminas utilizadas: {laminas_usadas}").pack(anchor=tk.W)
    ttk.Label(frame_stats, text=f"Área total utilizada: {area_total} cm²").pack(anchor=tk.W)
    
    # Crear figura de matplotlib
    fig = plt.figure(figsize=(10, 5 * laminas_usadas))
    
    # Generar subplots para cada lámina
    for idx, distribucion in enumerate(individuo.distribuciones):
        ax = fig.add_subplot(laminas_usadas, 1, idx+1)
        ax.set_title(f"Lámina {idx+1} - {len(distribucion['piezas'])} piezas")
        ax.set_xlim(0, individuo.lamina_ancho)
        ax.set_ylim(0, individuo.lamina_alto)
        
        # Dibujar contorno de la lámina
        ax.add_patch(patches.Rectangle(
            (0, 0), individuo.lamina_ancho, individuo.lamina_alto,
            edgecolor='black', facecolor='none', linewidth=2
        ))
        
        # Dibujar piezas
        for i, pieza in enumerate(distribucion['piezas']):
            color = plt.cm.tab20(i % 20)
            ax.add_patch(patches.Rectangle(
                (pieza.x, pieza.y), pieza.ancho, pieza.alto,
                edgecolor='black', facecolor=color, alpha=0.7
            ))
            ax.text(
                pieza.x + pieza.ancho/2, 
                pieza.y + pieza.alto/2,
                f"{pieza.ancho}x{pieza.alto}",
                ha='center', va='center', fontsize=8
            )
        
        # Dibujar áreas residuales
        if 'matriz' in distribucion:
            matriz = distribucion['matriz']
            visitados = np.zeros_like(matriz)
            for y in range(matriz.shape[0]):
                for x in range(matriz.shape[1]):
                    if matriz[y, x] == 0 and visitados[y, x] == 0:
                        # Calcular área residual
                        ancho = 0
                        while x + ancho < matriz.shape[1] and matriz[y, x + ancho] == 0:
                            ancho += 1
                        
                        alto = 0
                        while y + alto < matriz.shape[0] and np.all(matriz[y:y+alto+1, x:x+ancho] == 0):
                            alto += 1
                        
                        ax.add_patch(patches.Rectangle(
                            (x, y), ancho, alto,
                            edgecolor='#888888', facecolor='#eeeeee', alpha=0.5
                        ))
                        visitados[y:y+alto, x:x+ancho] = 1
        
        ax.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    
    # Integrar figura en la GUI
    canvas = FigureCanvasTkAgg(fig, master=tab_resultados)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # Agregar barra de desplazamiento
    scrollbar = ttk.Scrollbar(tab_resultados, orient="vertical", command=canvas.get_tk_widget().yview)
    canvas.get_tk_widget().configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Algoritmo Genético - Ejecución
def ejecutar_algoritmo_genetico():
    global piezas_seleccionadas, lamina_seleccionada
    
    if not lamina_seleccionada:
        messagebox.showerror("Error", "Seleccione una lámina primero")
        return
    
    if not piezas_seleccionadas:
        messagebox.showerror("Error", "Seleccione al menos una pieza")
        return
    
    # Configurar ventana de progreso
    ventana_progreso = tk.Toplevel(root)
    ventana_progreso.title("Progreso")
    ventana_progreso.geometry("300x150")
    
    lbl_progreso = tk.Label(ventana_progreso, text="Inicializando...")
    lbl_progreso.pack(pady=5)
    
    barra_progreso = ttk.Progressbar(ventana_progreso, length=250, mode='determinate')
    barra_progreso.pack(pady=5)
    
    lbl_detalles = tk.Label(ventana_progreso, text="")
    lbl_detalles.pack(pady=5)
    
    # Cola para comunicación entre hilos
    progreso_queue = queue.Queue()
    
    def actualizar_interfaz():
        while not progreso_queue.empty():
            tipo, valor = progreso_queue.get()
            if tipo == 'progreso':
                generacion, total, fitness, laminas = valor
                # Actualizar barra y labels
                barra_progreso['value'] = (generacion / total) * 100
                lbl_detalles.config(text=f"Generación {generacion}/{total} - Fitness: {fitness:.2f}")
                print(f"Gen {generacion:03d}/{total} | Fitness: {fitness:.2f} | Láminas: {laminas}")
            elif tipo == 'fin':
                # Mostrar resultados finales
                messagebox.showinfo("Resultado", f"Optimización completada\n{valor[1]}")
                ventana_progreso.destroy()
            elif tipo == 'error':
                messagebox.showerror("Error", valor)
                ventana_progreso.destroy()
        root.after(100, actualizar_interfaz)
    
    def ejecutar_en_hilo():
        try:
            inicio = time.time()
            mejor, historico = algoritmo_genetico(
                piezas_seleccionadas,
                lamina_seleccionada[0],
                lamina_seleccionada[1],
                progreso_queue  # Pasar la cola como parámetro
            )
            progreso_queue.put(('fin', (mejor, f"Tiempo: {time.time()-inicio:.1f}s")))
        except Exception as e:
            progreso_queue.put(('error', str(e)))
    
    # Iniciar hilos
    threading.Thread(target=ejecutar_en_hilo, daemon=True).start()
    root.after(100, actualizar_interfaz)

def exportar_resultado_csv(individuo):
    with open("resultado_multilamina.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Lámina", "Pieza", "Ancho", "Alto", "X", "Y", "Rotada"])
        for idx, distribucion in enumerate(individuo.distribuciones):
            for pieza in distribucion['piezas']:
                writer.writerow([
                    idx+1,
                    pieza.id,
                    pieza.ancho,
                    pieza.alto,
                    pieza.x,
                    pieza.y,
                    pieza.rotada
                ])

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