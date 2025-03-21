import tkinter as tk
from tkinter import ttk, messagebox
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv
import os
import numpy as np
import math 
from copy import deepcopy
import traceback  # Para el reporte de errores detallado

# modificacion de la funcion fitness esperando arreglar el problema de las laminas

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

    actualizar_tabla_piezas()


# Clases para el algoritmo genético
class Pieza:
    def __init__(self, ancho, alto, id_unico, rotada=False):
        self.ancho = ancho
        self.alto = alto
        self.id = id_unico  # ID único e inmutable
        self.rotada = rotada
        self.x = 0
        self.y = 0
        self.en_lamina = 0  # Lámina asignada
    
    def rotar(self):
        if self.ancho != self.alto:
            self.ancho, self.alto = self.alto, self.ancho
            self.rotada = not self.rotada
        return self
    
    def copia(self):
        return Pieza(self.ancho, self.alto, self.id, self.rotada)
    
    def area(self):
        return self.ancho * self.alto
    
    def __repr__(self):
        return f"Pieza(ID:{self.id} {self.ancho}x{self.alto})"

class Individuo:
    def __init__(self, piezas_originales, lamina_ancho, lamina_alto):
        self.piezas = [p.copia() for p in piezas_originales]  # Copia profunda
        self.lamina_ancho = lamina_ancho
        self.lamina_alto = lamina_alto
        self.fitness = 0.0
        self.distribuciones = []
        self.num_laminas = 1
        self.areas_residuales = []

    def copia(self):
        """Crea una copia profunda del individuo"""
        nuevo = Individuo(
            [p.copia() for p in self.piezas],
            self.lamina_ancho,
            self.lamina_alto
        )
        nuevo.fitness = self.fitness
        nuevo.distribuciones = deepcopy(self.distribuciones)
        nuevo.num_laminas = self.num_laminas
        nuevo.areas_residuales = self.areas_residuales.copy()
        return nuevo
    
    def validar_integridad(self):
        """Garantiza que no se pierdan o dupliquen piezas"""
        ids_originales = {p.id for p in self.piezas}
        ids_actuales = {p.id for p in self.piezas}
        return (
            len(ids_originales) == len(ids_actuales) == len(self.piezas
        ) and (ids_originales == ids_actuales))
    
    def _puede_ubicar(self, matriz, x, y, ancho, alto):
        if y + alto > matriz.shape[0] or x + ancho > matriz.shape[1]:
            return False
        return np.all(matriz[y:y+alto, x:x+ancho] == 0)
    
    def _marcar_matriz(self, matriz, x, y, ancho, alto):
        matriz[y:y+alto, x:x+ancho] = 1
    
    def _calcular_areas_residuales(self, matriz):
        visitados = np.zeros_like(matriz)
        areas = []
        for y in range(matriz.shape[0]):
            for x in range(matriz.shape[1]):
                if matriz[y, x] == 0 and visitados[y, x] == 0:
                    ancho_res = 0
                    alto_res = 0
                    # Buscar en horizontal
                    while x + ancho_res < matriz.shape[1] and matriz[y, x + ancho_res] == 0:
                        ancho_res += 1
                    # Buscar en vertical
                    while y + alto_res < matriz.shape[0] and np.all(matriz[y + alto_res, x:x+ancho_res] == 0):
                        alto_res += 1
                    areas.append(ancho_res * alto_res)
                    visitados[y:y+alto_res, x:x+ancho_res] = 1
        return areas
    
    def _calcular_distribucion_laminar(self, piezas_por_ubicar):
        matriz = np.zeros((self.lamina_alto, self.lamina_ancho))
        colocadas = []
        no_colocadas = []
        
        # Ordenar por área descendente manteniendo IDs
        piezas_ordenadas = sorted(piezas_por_ubicar, key=lambda p: (-p.area(), p.id))
        
        for pieza in piezas_ordenadas:
            ubicada = False
            # Intentar en orientación original
            for y in range(self.lamina_alto - pieza.alto + 1):
                for x in range(self.lamina_ancho - pieza.ancho + 1):
                    if self._puede_ubicar(matriz, x, y, pieza.ancho, pieza.alto):
                        self._marcar_matriz(matriz, x, y, pieza.ancho, pieza.alto)
                        pieza_copia = pieza.copia()
                        pieza_copia.x = x
                        pieza_copia.y = y
                        colocadas.append(pieza_copia)
                        ubicada = True
                        break
                if ubicada: break
            
            # Intentar rotado si no se ubicó
            if not ubicada and pieza.ancho != pieza.alto:
                pieza_rotada = pieza.copia().rotar()
                for y in range(self.lamina_alto - pieza_rotada.alto + 1):
                    for x in range(self.lamina_ancho - pieza_rotada.ancho + 1):
                        if self._puede_ubicar(matriz, x, y, pieza_rotada.ancho, pieza_rotada.alto):
                            self._marcar_matriz(matriz, x, y, pieza_rotada.ancho, pieza_rotada.alto)
                            pieza_rotada.x = x
                            pieza_rotada.y = y
                            colocadas.append(pieza_rotada)
                            ubicada = True
                            break
                    if ubicada: break
            
            if not ubicada:
                no_colocadas.append(pieza.copia())
        
        return {
            'matriz': matriz,
            'piezas': colocadas,
            'piezas_no_ubicadas': no_colocadas,
            'area_utilizada': sum(p.area() for p in colocadas),
            'areas_residuales': self._calcular_areas_residuales(matriz)
        }
    
    def calcular_fitness(self):
        if not self.validar_integridad():
            self.fitness = -999999.0
            return self.fitness
        
        self.distribuciones = []
        piezas_restantes = [p.copia() for p in self.piezas]
        total_area_piezas = sum(p.area() for p in self.piezas)
        total_area_utilizada = 0.0
        areas_residuales = []
        
        while piezas_restantes:
            distribucion = self._calcular_distribucion_laminar(piezas_restantes)
            self.distribuciones.append(distribucion)
            piezas_restantes = distribucion['piezas_no_ubicadas']
            total_area_utilizada += distribucion['area_utilizada']
            areas_residuales.extend(distribucion['areas_residuales'])
        
        # Cálculo de métricas
        utilizacion_promedio = np.mean([
            d['area_utilizada'] / (self.lamina_ancho * self.lamina_alto)
            for d in self.distribuciones
        ]) if self.distribuciones else 0.0
        
        minimo_laminas = max(1, int(np.ceil(total_area_piezas / (self.lamina_ancho * self.lamina_alto))))
        exceso_laminas = max(0, len(self.distribuciones) - minimo_laminas)
        
        # Fitness compuesto
        self.num_laminas = len(self.distribuciones)
        self.fitness = (utilizacion_promedio * 0.8) - (0.05 * exceso_laminas)
        
        return self.fitness

# Funciones para el algoritmo genético
def crear_poblacion_inicial(piezas_maestras, lamina_ancho, lamina_alto):
    poblacion = []
    for _ in range(POP_SIZE):
        # Barajar manteniendo todas las piezas
        individuo_piezas = [p.copia() for p in piezas_maestras]
        random.shuffle(individuo_piezas)
        
        # Aplicar rotaciones iniciales
        for p in individuo_piezas:
            if random.random() < 0.5 and p.ancho != p.alto:
                p.rotar()
        
        individuo = Individuo(individuo_piezas, lamina_ancho, lamina_alto)
        individuo.calcular_fitness()
        poblacion.append(individuo)
    return poblacion

def seleccion_torneo(poblacion, k=3):
    seleccionados = []
    for _ in range(POP_SIZE):
        competidores = random.sample(poblacion, k)
        ganador = max(competidores, key=lambda x: x.fitness)
        seleccionados.append(ganador)
    return seleccionados

def cruce(padre1, padre2):
    if random.random() > CROSSOVER_RATE or len(padre1.piezas) != len(padre2.piezas):
        return padre1, padre2
    
    punto = random.randint(1, len(padre1.piezas)-1)
    hijo1_piezas = [p.copia() for p in padre1.piezas[:punto]] + [p.copia() for p in padre2.piezas[punto:]]
    hijo2_piezas = [p.copia() for p in padre2.piezas[:punto]] + [p.copia() for p in padre1.piezas[punto:]]
    
    hijo1 = Individuo(hijo1_piezas, padre1.lamina_ancho, padre1.lamina_alto)
    hijo2 = Individuo(hijo2_piezas, padre2.lamina_ancho, padre2.lamina_alto)
    
    return hijo1, hijo2

def mutacion(individuo):
    # Crear copia usando el nuevo método
    nuevo = individuo.copia()
    
    # Mutación por intercambio
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(nuevo.piezas)), 2)
        nuevo.piezas[i], nuevo.piezas[j] = nuevo.piezas[j], nuevo.piezas[i]
    
    # Mutación por rotación
    for p in nuevo.piezas:
        if random.random() < MUTATION_RATE/3:
            p.rotar()
    
    nuevo.calcular_fitness()
    return nuevo

def algoritmo_genetico(piezas_maestras, lamina_ancho, lamina_alto):
    poblacion = crear_poblacion_inicial(piezas_maestras, lamina_ancho, lamina_alto)
    mejor_global = max(poblacion, key=lambda x: x.fitness)
    historico = [mejor_global.fitness]
    
    for generacion in range(GENERATIONS):
        # Selección
        seleccionados = seleccion_torneo(poblacion)
        
        # Cruce
        nueva_poblacion = []
        for i in range(0, POP_SIZE, 2):
            if i+1 < POP_SIZE:
                hijo1, hijo2 = cruce(seleccionados[i], seleccionados[i+1])
                nueva_poblacion.extend([hijo1, hijo2])
            else:
                nueva_poblacion.append(seleccionados[i])
        
        # Mutación
        nueva_poblacion = [mutacion(ind) for ind in nueva_poblacion]
        
        # Elitismo
        mejor_actual = max(nueva_poblacion, key=lambda x: x.fitness)
        if mejor_actual.fitness > mejor_global.fitness:
            mejor_global = mejor_actual.copia()  # Usar el nuevo método
        else:
            peor_idx = min(range(POP_SIZE), key=lambda i: nueva_poblacion[i].fitness)
            nueva_poblacion[peor_idx] = mejor_global.copia()
        
        # Actualizar población
        poblacion = nueva_poblacion
        historico.append(mejor_global.fitness)
        
        # Mostrar progreso
        if generacion % 10 == 0:
            print(f"Gen {generacion}: Fitness={mejor_global.fitness:.2f} Láminas={mejor_global.num_laminas}")
    
    return mejor_global, historico

def calcular_minimo_laminas(piezas, lamina_ancho, lamina_alto):
    area_total = sum(p.ancho * p.alto for p in piezas)
    area_lamina = lamina_ancho * lamina_alto
    return max(1, math.ceil(area_total / area_lamina))

# Expandir las piezas seleccionadas para el algoritmo genético
def expandir_piezas(piezas_seleccionadas):
    todas_piezas = []
    contador_id = 0
    for (ancho, alto), cantidad in piezas_seleccionadas:
        for _ in range(cantidad):
            todas_piezas.append(Pieza(ancho, alto, contador_id))
            contador_id += 1
    return todas_piezas

def actualizar_tabla_piezas():
    for row in tabla_piezas.get_children():
        tabla_piezas.delete(row)
    contador_id = 0
    for (ancho, alto), cantidad in piezas_seleccionadas:
        for _ in range(cantidad):
            tabla_piezas.insert("", tk.END, values=(f"{ancho}x{alto}", contador_id))
            contador_id += 1

# Algoritmo Genético - Ejecución
def ejecutar_algoritmo_genetico():
    global mejor_individuo_global
    
    if not lamina_seleccionada:
        messagebox.showerror("Error", "Seleccione una lámina primero")
        return
    
    if not piezas_seleccionadas:
        messagebox.showerror("Error", "Seleccione al menos una pieza")
        return
    
    try:
        # Generar piezas con IDs únicos
        piezas_maestras = []
        contador_id = 0
        for (ancho, alto), cantidad in piezas_seleccionadas:
            for _ in range(cantidad):
                piezas_maestras.append(Pieza(ancho, alto, contador_id))
                contador_id += 1
        
        # Validar dimensiones
        lamina_ancho = lamina_seleccionada[0]
        lamina_alto = lamina_seleccionada[1]
        
        for pieza in piezas_maestras:
            if not (pieza.ancho <= lamina_ancho and pieza.alto <= lamina_alto) and \
               not (pieza.alto <= lamina_ancho and pieza.ancho <= lamina_alto):
                messagebox.showerror(
                    "Error", 
                    f"Pieza {pieza.ancho}x{pieza.alto} (ID:{pieza.id}) no cabe en la lámina"
                )
                return
        
        # Ejecutar AG
        mejor_individuo, historico = algoritmo_genetico(piezas_maestras, lamina_ancho, lamina_alto)
        
        if not mejor_individuo.validar_integridad():
            messagebox.showerror("Error", "Solución inválida: piezas perdidas o duplicadas")
            return
        
        # Mostrar resultados
        messagebox.showinfo(
            "Resultado", 
            f"Láminas usadas: {mejor_individuo.num_laminas}\n"
            f"Piezas colocadas: {len(mejor_individuo.piezas)}/{len(piezas_maestras)}\n"
            f"Fitness: {mejor_individuo.fitness:.2%}"
        )
        
        mejor_individuo_global = mejor_individuo
        mostrar_resultado(mejor_individuo)
        mostrar_grafica_evolucion(historico)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error: {str(e)}")
        print(traceback.format_exc())

# Función para mostrar el resultado gráfico
def mostrar_resultado(individuo):
    if not individuo.distribuciones:
        messagebox.showwarning("Error", "No hay distribución para mostrar")
        return
    
    output_dir = "resultados_corte"
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, dist in enumerate(individuo.distribuciones):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.set_title(f"Lámina {idx+1} - Piezas: {len(dist['piezas'])}")
        ax.set_xlim(0, individuo.lamina_ancho)
        ax.set_ylim(0, individuo.lamina_alto)
        
        # Dibujar lámina
        ax.add_patch(patches.Rectangle(
            (0, 0), individuo.lamina_ancho, individuo.lamina_alto,
            edgecolor='black', facecolor='none', linewidth=2
        ))
        
        # Dibujar piezas
        for p in dist['piezas']:
            ax.add_patch(patches.Rectangle(
                (p.x, p.y), p.ancho, p.alto,
                edgecolor='black', facecolor=np.random.rand(3), alpha=0.7
            ))
            ax.text(
                p.x + p.ancho/2, p.y + p.alto/2,
                f"ID:{p.id}\n{p.ancho}x{p.alto}",
                ha='center', va='center', fontsize=6
            )
        
        plt.savefig(os.path.join(output_dir, f"lamina_{idx+1}.png"))
        plt.close()
    
    messagebox.showinfo("Éxito", f"Gráficos guardados en: {output_dir}")

# Función para mostrar la gráfica de evolución
def mostrar_grafica_evolucion(historico):
    plt.figure(figsize=(10, 6))
    plt.plot(historico, 'b-', linewidth=2)
    plt.title("Evolución del Fitness")
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.grid(True)
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