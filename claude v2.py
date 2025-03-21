import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image
import math
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Pieza:
    def __init__(self, id, ancho, alto, cantidad=1):
        self.id = id
        self.ancho = ancho
        self.alto = alto
        self.cantidad = cantidad
        self.area = ancho * alto

    def __str__(self):
        return f"Pieza {self.id}: {self.ancho}x{self.alto}, Cantidad: {self.cantidad}"

class Lamina:
    def __init__(self, id, ancho, alto):
        self.id = id
        self.ancho = ancho
        self.alto = alto
        self.area = ancho * alto

    def __str__(self):
        return f"Lámina {self.id}: {self.ancho}x{self.alto}"

class Posicion:
    def __init__(self, x, y, pieza_id, lamina_idx):
        self.x = x
        self.y = y
        self.pieza_id = pieza_id
        self.lamina_idx = lamina_idx

    def __str__(self):
        return f"Pos({self.x},{self.y}) - Pieza {self.pieza_id} en Lámina {self.lamina_idx}"

class Individuo:
    def __init__(self, posiciones, piezas, laminas):
        self.posiciones = posiciones  # Lista de objetos Posicion
        self.piezas = piezas  # Diccionario de objetos Pieza por id
        self.laminas = laminas  # Lista de objetos Lamina
        self.fitness = 0
        self.laminas_usadas = set()
        self.calcular_fitness()

    def calcular_fitness(self):
        # Verificar colisiones y salidas de límites
        if self.verificar_restricciones():
            # Calcular áreas
            area_piezas = sum(pieza.area * pieza.cantidad for pieza in self.piezas.values())
            
            # Determinar láminas utilizadas
            self.laminas_usadas = set(pos.lamina_idx for pos in self.posiciones)
            num_laminas_usadas = len(self.laminas_usadas)
            area_laminas_usadas = sum(self.laminas[idx].area for idx in self.laminas_usadas)
            
            # Calcular número teórico mínimo de láminas
            lamina_area = self.laminas[0].area  # Asumimos láminas de igual tamaño
            laminas_minimas_teoricas = math.ceil(area_piezas / lamina_area)
            
            # Calcular fitness base
            self.fitness = area_piezas / area_laminas_usadas
            
            # Penalizar uso excesivo de láminas
            P = 0.1  # Factor de penalización
            self.fitness -= P * max(0, num_laminas_usadas - laminas_minimas_teoricas)
        else:
            self.fitness = 0

        return self.fitness

    def verificar_restricciones(self):
        # Verificar colisiones y salidas de límites
        for i, pos1 in enumerate(self.posiciones):
            pieza1 = self.piezas[pos1.pieza_id]
            lamina = self.laminas[pos1.lamina_idx]
            
            # Verificar si la pieza sale de los límites de la lámina
            if (pos1.x < 0 or pos1.y < 0 or 
                pos1.x + pieza1.ancho > lamina.ancho or 
                pos1.y + pieza1.alto > lamina.alto):
                return False
            
            # Verificar colisiones con otras piezas
            for j, pos2 in enumerate(self.posiciones):
                if i != j and pos1.lamina_idx == pos2.lamina_idx:
                    pieza2 = self.piezas[pos2.pieza_id]
                    
                    # Verificar si hay colisión
                    if (pos1.x < pos2.x + pieza2.ancho and
                        pos1.x + pieza1.ancho > pos2.x and
                        pos1.y < pos2.y + pieza2.alto and
                        pos1.y + pieza1.alto > pos2.y):
                        return False
        
        return True

    def __str__(self):
        return f"Individuo con {len(self.posiciones)} posiciones, Fitness: {self.fitness:.4f}"

class AlgoritmoGenetico:
    def __init__(self, piezas, laminas, tam_poblacion=100, max_generaciones=100, prob_cruce=0.8, prob_mutacion=0.2):
        self.piezas = piezas  # Diccionario de objetos Pieza por id
        self.laminas = laminas  # Lista de objetos Lamina
        self.tam_poblacion = tam_poblacion
        self.max_generaciones = max_generaciones
        self.prob_cruce = prob_cruce
        self.prob_mutacion = prob_mutacion
        self.poblacion = []
        self.mejor_individuo = None
        self.historia_fitness = []
        self.generacion_actual = 0

    def inicializar_poblacion(self):
        self.poblacion = []
        
        # Crear lista expandida de piezas según cantidades
        piezas_expandidas = []
        for pieza_id, pieza in self.piezas.items():
            for i in range(pieza.cantidad):
                piezas_expandidas.append(pieza_id)
        
        for _ in range(self.tam_poblacion):
            posiciones = []
            random.shuffle(piezas_expandidas)
            
            for pieza_id in piezas_expandidas:
                # Elegir una lámina aleatoria
                lamina_idx = random.randint(0, len(self.laminas) - 1)
                lamina = self.laminas[lamina_idx]
                pieza = self.piezas[pieza_id]
                
                # Generar posición aleatoria dentro de la lámina
                x = random.randint(0, lamina.ancho - pieza.ancho)
                y = random.randint(0, lamina.alto - pieza.alto)
                
                posiciones.append(Posicion(x, y, pieza_id, lamina_idx))
            
            individuo = Individuo(posiciones, self.piezas, self.laminas)
            self.poblacion.append(individuo)
        
        # Encontrar el mejor individuo inicial
        self.actualizar_mejor_individuo()

    def actualizar_mejor_individuo(self):
        mejor = max(self.poblacion, key=lambda ind: ind.fitness)
        if self.mejor_individuo is None or mejor.fitness > self.mejor_individuo.fitness:
            self.mejor_individuo = Individuo(mejor.posiciones.copy(), self.piezas, self.laminas)

    def seleccion_torneo(self, k=2):
        """Selección por torneo"""
        seleccionados = []
        for _ in range(len(self.poblacion)):
            competidores = random.sample(self.poblacion, k)
            ganador = max(competidores, key=lambda ind: ind.fitness)
            seleccionados.append(ganador)
        return seleccionados

    def cruce(self, padre1, padre2):
        """Cruza dos individuos intercambiando secciones de sus posiciones"""
        if random.random() > self.prob_cruce:
            return padre1, padre2
        
        punto_corte = random.randint(1, len(padre1.posiciones) - 1)
        
        hijo1_pos = padre1.posiciones[:punto_corte] + padre2.posiciones[punto_corte:]
        hijo2_pos = padre2.posiciones[:punto_corte] + padre1.posiciones[punto_corte:]
        
        hijo1 = Individuo(hijo1_pos, self.piezas, self.laminas)
        hijo2 = Individuo(hijo2_pos, self.piezas, self.laminas)
        
        return hijo1, hijo2

    def mutacion(self, individuo):
        """Muta un individuo cambiando la posición o lámina de una pieza"""
        if random.random() > self.prob_mutacion:
            return individuo
        
        nueva_posiciones = individuo.posiciones.copy()
        idx_a_mutar = random.randint(0, len(nueva_posiciones) - 1)
        pos = nueva_posiciones[idx_a_mutar]
        
        # Decidir qué tipo de mutación aplicar
        tipo_mutacion = random.choice(["posicion", "lamina"])
        
        if tipo_mutacion == "posicion":
            # Cambiar la posición dentro de la misma lámina
            lamina = self.laminas[pos.lamina_idx]
            pieza = self.piezas[pos.pieza_id]
            pos.x = random.randint(0, lamina.ancho - pieza.ancho)
            pos.y = random.randint(0, lamina.alto - pieza.alto)
        else:
            # Cambiar la lámina asignada
            nueva_lamina_idx = random.randint(0, len(self.laminas) - 1)
            lamina = self.laminas[nueva_lamina_idx]
            pieza = self.piezas[pos.pieza_id]
            pos.lamina_idx = nueva_lamina_idx
            pos.x = random.randint(0, lamina.ancho - pieza.ancho)
            pos.y = random.randint(0, lamina.alto - pieza.alto)
        
        return Individuo(nueva_posiciones, self.piezas, self.laminas)

    def evolucionar(self):
        """Realiza una generación de evolución"""
        # Selección
        seleccionados = self.seleccion_torneo()
        
        # Cruce
        nueva_poblacion = []
        for i in range(0, len(seleccionados), 2):
            if i + 1 < len(seleccionados):
                hijo1, hijo2 = self.cruce(seleccionados[i], seleccionados[i+1])
                nueva_poblacion.append(hijo1)
                nueva_poblacion.append(hijo2)
            else:
                nueva_poblacion.append(seleccionados[i])
        
        # Mutación
        for i in range(len(nueva_poblacion)):
            nueva_poblacion[i] = self.mutacion(nueva_poblacion[i])
        
        # Elitismo (mantener al mejor individuo)
        if self.mejor_individuo:
            peor_idx = min(range(len(nueva_poblacion)), key=lambda i: nueva_poblacion[i].fitness)
            nueva_poblacion[peor_idx] = Individuo(self.mejor_individuo.posiciones.copy(), self.piezas, self.laminas)
        
        self.poblacion = nueva_poblacion
        self.actualizar_mejor_individuo()
        self.generacion_actual += 1
        
        # Guardar historia de fitness
        mejor_fitness = max(ind.fitness for ind in self.poblacion)
        promedio_fitness = sum(ind.fitness for ind in self.poblacion) / len(self.poblacion)
        self.historia_fitness.append((mejor_fitness, promedio_fitness))

    def ejecutar(self, callback=None):
        """Ejecuta el algoritmo genético completo"""
        self.inicializar_poblacion()
        
        for generacion in range(self.max_generaciones):
            self.evolucionar()
            
            if callback and generacion % 5 == 0:
                callback(generacion, self.mejor_individuo)
            
            # Criterio de parada (opcional)
            if self.mejor_individuo.fitness > 0.95:
                break
        
        return self.mejor_individuo

    def generar_imagenes(self, carpeta_salida="resultados"):
        """Genera imágenes de las láminas con los cortes realizados"""
        # Crear carpeta si no existe
        if not os.path.exists(carpeta_salida):
            os.makedirs(carpeta_salida)
        
        # Agrupar posiciones por lámina
        laminas_dict = {}
        for pos in self.mejor_individuo.posiciones:
            if pos.lamina_idx not in laminas_dict:
                laminas_dict[pos.lamina_idx] = []
            laminas_dict[pos.lamina_idx].append(pos)
        
        # Generar una imagen por cada lámina utilizada
        for lamina_idx, posiciones in laminas_dict.items():
            lamina = self.laminas[lamina_idx]
            
            # Crear figura
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.set_xlim(0, lamina.ancho)
            ax.set_ylim(0, lamina.alto)
            ax.set_aspect('equal')
            
            # Dibujar lámina
            ax.add_patch(patches.Rectangle((0, 0), lamina.ancho, lamina.alto, 
                                          linewidth=2, edgecolor='black', facecolor='none'))
            
            # Dibujar piezas
            colores = plt.cm.tab20(np.linspace(0, 1, len(self.piezas)))
            for pos in posiciones:
                pieza = self.piezas[pos.pieza_id]
                color_idx = int(pos.pieza_id) % len(colores)
                ax.add_patch(patches.Rectangle((pos.x, pos.y), pieza.ancho, pieza.alto, 
                                              linewidth=1, edgecolor='black', 
                                              facecolor=colores[color_idx]))
                # Añadir texto con el ID de la pieza
                ax.text(pos.x + pieza.ancho/2, pos.y + pieza.alto/2, str(pos.pieza_id),
                        fontsize=8, ha='center', va='center')
            
            # Configurar título y etiquetas
            ax.set_title(f"Lámina {lamina_idx+1}: {lamina.ancho}x{lamina.alto}")
            ax.set_xlabel("Ancho")
            ax.set_ylabel("Alto")
            
            # Guardar imagen
            plt.tight_layout()
            plt.savefig(f"{carpeta_salida}/lamina_{lamina_idx+1}.png", dpi=300)
            plt.close(fig)

class AplicacionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Optimización de Corte de Piezas")
        self.root.geometry("1000x650")
        
        self.piezas = {}  # Diccionario de piezas
        self.laminas = []  # Lista de láminas
        self.ag = None
        
        self.crear_interfaz()
        
    def crear_interfaz(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Notebook para pestañas
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Pestaña 1: Entrada de datos
        tab_datos = ttk.Frame(notebook)
        notebook.add(tab_datos, text="Entrada de Datos")
        
        # Pestaña 2: Resultados
        tab_resultados = ttk.Frame(notebook)
        notebook.add(tab_resultados, text="Resultados")
        
        # Pestaña 3: Visualización
        tab_visualizacion = ttk.Frame(notebook)
        notebook.add(tab_visualizacion, text="Visualización")
        
        # Configurar pestaña de entrada de datos
        self.configurar_tab_datos(tab_datos)
        
        # Configurar pestaña de resultados
        self.configurar_tab_resultados(tab_resultados)
        
        # Configurar pestaña de visualización
        self.configurar_tab_visualizacion(tab_visualizacion)
        
    def configurar_tab_datos(self, tab):
        # Frame para láminas
        frame_laminas = ttk.LabelFrame(tab, text="Gestión de Láminas")
        frame_laminas.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Inputs para láminas
        ttk.Label(frame_laminas, text="Ancho:").grid(row=0, column=0, padx=5, pady=5)
        self.ancho_lamina = ttk.Entry(frame_laminas, width=10)
        self.ancho_lamina.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(frame_laminas, text="Alto:").grid(row=1, column=0, padx=5, pady=5)
        self.alto_lamina = ttk.Entry(frame_laminas, width=10)
        self.alto_lamina.grid(row=1, column=1, padx=5, pady=5)
        
        # Botones para láminas
        ttk.Button(frame_laminas, text="Agregar Lámina", command=self.agregar_lamina).grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        
        # Lista de láminas
        self.lista_laminas = ttk.Treeview(frame_laminas, columns=("ID", "Ancho", "Alto"), show="headings", height=6)
        self.lista_laminas.grid(row=3, column=0, columnspan=2, padx=5, pady=5)
        self.lista_laminas.heading("ID", text="ID")
        self.lista_laminas.heading("Ancho", text="Ancho")
        self.lista_laminas.heading("Alto", text="Alto")
        self.lista_laminas.column("ID", width=50)
        self.lista_laminas.column("Ancho", width=80)
        self.lista_laminas.column("Alto", width=80)
        
        # Frame para piezas
        frame_piezas = ttk.LabelFrame(tab, text="Gestión de Piezas")
        frame_piezas.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # Inputs para piezas
        ttk.Label(frame_piezas, text="ID:").grid(row=0, column=0, padx=5, pady=5)
        self.id_pieza = ttk.Entry(frame_piezas, width=10)
        self.id_pieza.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(frame_piezas, text="Ancho:").grid(row=1, column=0, padx=5, pady=5)
        self.ancho_pieza = ttk.Entry(frame_piezas, width=10)
        self.ancho_pieza.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(frame_piezas, text="Alto:").grid(row=2, column=0, padx=5, pady=5)
        self.alto_pieza = ttk.Entry(frame_piezas, width=10)
        self.alto_pieza.grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(frame_piezas, text="Cantidad:").grid(row=3, column=0, padx=5, pady=5)
        self.cantidad_pieza = ttk.Entry(frame_piezas, width=10)
        self.cantidad_pieza.insert(0, "1")
        self.cantidad_pieza.grid(row=3, column=1, padx=5, pady=5)
        
        # Botones para piezas
        ttk.Button(frame_piezas, text="Agregar Pieza", command=self.agregar_pieza).grid(row=4, column=0, columnspan=2, padx=5, pady=5)
        
        # Lista de piezas
        self.lista_piezas = ttk.Treeview(frame_piezas, columns=("ID", "Ancho", "Alto", "Cantidad"), show="headings", height=6)
        self.lista_piezas.grid(row=5, column=0, columnspan=2, padx=5, pady=5)
        self.lista_piezas.heading("ID", text="ID")
        self.lista_piezas.heading("Ancho", text="Ancho")
        self.lista_piezas.heading("Alto", text="Alto")
        self.lista_piezas.heading("Cantidad", text="Cantidad")
        self.lista_piezas.column("ID", width=50)
        self.lista_piezas.column("Ancho", width=80)
        self.lista_piezas.column("Alto", width=80)
        self.lista_piezas.column("Cantidad", width=80)
        
        # Frame para archivo CSV
        frame_csv = ttk.LabelFrame(tab, text="Gestión de Archivos CSV")
        frame_csv.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        # Botones para CSV
        ttk.Button(frame_csv, text="Guardar Datos", command=self.guardar_datos).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(frame_csv, text="Cargar Datos", command=self.cargar_datos).grid(row=0, column=1, padx=5, pady=5)
        
        # Frame para Algoritmo Genético
        frame_ag = ttk.LabelFrame(tab, text="Configuración Algoritmo Genético")
        frame_ag.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        # Inputs para AG
        ttk.Label(frame_ag, text="Tamaño Población:").grid(row=0, column=0, padx=5, pady=5)
        self.tam_poblacion = ttk.Entry(frame_ag, width=10)
        self.tam_poblacion.insert(0, "100")
        self.tam_poblacion.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(frame_ag, text="Máx. Generaciones:").grid(row=0, column=2, padx=5, pady=5)
        self.max_generaciones = ttk.Entry(frame_ag, width=10)
        self.max_generaciones.insert(0, "100")
        self.max_generaciones.grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Label(frame_ag, text="Prob. Cruce:").grid(row=1, column=0, padx=5, pady=5)
        self.prob_cruce = ttk.Entry(frame_ag, width=10)
        self.prob_cruce.insert(0, "0.8")
        self.prob_cruce.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(frame_ag, text="Prob. Mutación:").grid(row=1, column=2, padx=5, pady=5)
        self.prob_mutacion = ttk.Entry(frame_ag, width=10)
        self.prob_mutacion.insert(0, "0.2")
        self.prob_mutacion.grid(row=1, column=3, padx=5, pady=5)
        
        # Botón para ejecutar AG
        ttk.Button(frame_ag, text="Ejecutar Algoritmo Genético", command=self.ejecutar_ag).grid(row=2, column=0, columnspan=4, padx=5, pady=5)
        
        # Configurar grid weights
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=0)
        tab.grid_rowconfigure(2, weight=0)
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=1)
        
    def configurar_tab_resultados(self, tab):
        # Frame para resultados
        frame_resultados = ttk.LabelFrame(tab, text="Resultados del Algoritmo Genético")
        frame_resultados.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Área de texto para resultados
        self.texto_resultados = tk.Text(frame_resultados, height=15, width=80)
        self.texto_resultados.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Frame para gráfica
        frame_grafica = ttk.LabelFrame(tab, text="Evolución del Fitness")
        frame_grafica.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Figura para la gráfica
        self.fig_fitness, self.ax_fitness = plt.subplots(figsize=(6, 4))
        self.canvas_fitness = FigureCanvasTkAgg(self.fig_fitness, master=frame_grafica)
        self.canvas_fitness.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Botón para guardar resultados
        ttk.Button(tab, text="Guardar Imágenes", command=self.guardar_imagenes).pack(pady=10)
        
    def configurar_tab_visualizacion(self, tab):
        # Frame para visualización
        frame_visualizacion = ttk.LabelFrame(tab, text="Visualización de Láminas")
        frame_visualizacion.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Listbox para seleccionar lámina
        ttk.Label(frame_visualizacion, text="Seleccionar Lámina:").pack(anchor=tk.W, padx=5, pady=5)
        self.lista_laminas_vis = tk.Listbox(frame_visualizacion, height=5)
        self.lista_laminas_vis.pack(fill=tk.X, padx=5, pady=5)
        self.lista_laminas_vis.bind('<<ListboxSelect>>', self.mostrar_lamina)
        
        # Frame para la figura
        frame_figura = ttk.Frame(frame_visualizacion)
        frame_figura.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Figura para la visualización
        self.fig_lamina, self.ax_lamina = plt.subplots(figsize=(6, 5))
        self.canvas_lamina = FigureCanvasTkAgg(self.fig_lamina, master=frame_figura)
        self.canvas_lamina.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def agregar_lamina(self):
        try:
            ancho = float(self.ancho_lamina.get())
            alto = float(self.alto_lamina.get())
            
            if ancho <= 0 or alto <= 0:
                messagebox.showerror("Error", "Las dimensiones deben ser positivas")
                return
            
            lamina_id = len(self.laminas)
            self.laminas.append(Lamina(lamina_id, ancho, alto))
            
            self.lista_laminas.insert("", "end", values=(lamina_id, ancho, alto))
            
            # Limpiar campos
            self.ancho_lamina.delete(0, tk.END)
            self.alto_lamina.delete(0, tk.END)
            
        except ValueError:
            messagebox.showerror("Error", "Por favor, ingrese valores numéricos válidos")
            
    def agregar_pieza(self):
        try:
            pieza_id = self.id_pieza.get()
            ancho = float(self.ancho_pieza.get())
            alto = float(self.alto_pieza.get())
            cantidad = int(self.cantidad_pieza.get())
            
            if ancho <= 0 or alto <= 0 or cantidad <= 0:
                messagebox.showerror("Error", "Las dimensiones y cantidad deben ser positivas")
                return
            
            if not pieza_id:
                pieza_id = str(len(self.piezas))
            
            self.piezas[pieza_id] = Pieza(pieza_id, ancho, alto, cantidad)
            
            self.lista_piezas.insert("", "end", values=(pieza_id, ancho, alto, cantidad))
            
            # Limpiar campos
            self.id_pieza.delete(0, tk.END)
            self.ancho_pieza.delete(0, tk.END)
            self.alto_pieza.delete(0, tk.END)
            self.cantidad_pieza.delete(0, tk.END)
            self.cantidad_pieza.insert(0, "1")
            
        except ValueError:
            messagebox.showerror("Error", "Por favor, ingrese valores numéricos válidos")
            
    def guardar_datos(self):
        try:
            # Guardar láminas
            df_laminas = pd.DataFrame({
                'id': [lamina.id for lamina in self.laminas],
                'ancho': [lamina.ancho for lamina in self.laminas],
                'alto': [lamina.alto for lamina in self.laminas]
            })
            
            # Guardar piezas
            df_piezas = pd.DataFrame({
                'id': [pieza.id for pieza in self.piezas.values()],
                'ancho': [pieza.ancho for pieza in self.piezas.values()],
                'alto': [pieza.alto for pieza in self.piezas.values()],
                'cantidad': [pieza.cantidad for pieza in self.piezas.values()]
            })
            
            # Elegir ubicación para guardar
            nombre_archivo = filedialog.asksaveasfilename(
                defaultextension=".csv", 
                filetypes=[("Archivos CSV", "*.csv")],
                title="Guardar Datos"
            )
            
            if nombre_archivo:
                # Guardar en un solo archivo con un indicador de tipo
                df_laminas['tipo'] = 'lamina'
                df_piezas['tipo'] = 'pieza'
                df_combinado = pd.concat([df_laminas, df_piezas])
                df_combinado.to_csv(nombre_archivo, index=False)
                messagebox.showinfo("Éxito", f"Datos guardados correctamente en {nombre_archivo}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar los datos: {str(e)}")
            
    def cargar_datos(self):
        try:
            # Elegir archivo para cargar
            nombre_archivo = filedialog.askopenfilename(
                filetypes=[("Archivos CSV", "*.csv")],
                title="Cargar Datos"
            )
            
            if nombre_archivo:
                # Cargar datos
                df = pd.read_csv(nombre_archivo)
                
                # Limpiar datos actuales
                self.laminas = []
                self.piezas = {}
                self.lista_laminas.delete(*self.lista_laminas.get_children())
                self.lista_piezas.delete(*self.lista_piezas.get_children())
                
                # Procesar datos cargados
                df_laminas = df[df['tipo'] == 'lamina']
                df_piezas = df[df['tipo'] == 'pieza']
                
                # Cargar láminas
                for _, row in df_laminas.iterrows():
                    lamina = Lamina(row['id'], row['ancho'], row['alto'])
                    self.laminas.append(lamina)
                    self.lista_laminas.insert("", "end", values=(row['id'], row['ancho'], row['alto']))
                
                # Cargar piezas
                for _, row in df_piezas.iterrows():
                    pieza = Pieza(row['id'], row['ancho'], row['alto'], row['cantidad'])
                    self.piezas[row['id']] = pieza
                    self.lista_piezas.insert("", "end", values=(row['id'], row['ancho'], row['alto'], row['cantidad']))
                
                messagebox.showinfo("Éxito", f"Datos cargados correctamente desde {nombre_archivo}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar los datos: {str(e)}")
            
    def ejecutar_ag(self):
        try:
            # Verificar que haya datos
            if not self.laminas:
                messagebox.showerror("Error", "No hay láminas definidas")
                return
                
            if not self.piezas:
                messagebox.showerror("Error", "No hay piezas definidas")
                return
            
            # Obtener parámetros del AG
            tam_poblacion = int(self.tam_poblacion.get())
            max_generaciones = int(self.max_generaciones.get())
            prob_cruce = float(self.prob_cruce.get())
            prob_mutacion = float(self.prob_mutacion.get())
            
            # Crear instancia del AG
            self.ag = AlgoritmoGenetico(
                self.piezas,
                self.laminas,
                tam_poblacion,
                max_generaciones,
                prob_cruce,
                prob_mutacion
            )
            
            # Ejecutar AG
            self.texto_resultados.delete(1.0, tk.END)
            self.texto_resultados.insert(tk.END, "Ejecutando Algoritmo Genético...\n")
            self.root.update()
            
            # Función de callback para actualizar progreso
            def actualizar_progreso(generacion, mejor_individuo):
                self.texto_resultados.insert(tk.END, f"Generación {generacion}: Fitness = {mejor_individuo.fitness:.4f}\n")
                self.texto_resultados.see(tk.END)
                self.root.update()
            
            # Ejecutar AG con callback
            mejor_individuo = self.ag.ejecutar(actualizar_progreso)
            
            # Mostrar resultados
            self.texto_resultados.insert(tk.END, "\n--- Resultados Finales ---\n")
            self.texto_resultados.insert(tk.END, f"Mejor Fitness: {mejor_individuo.fitness:.4f}\n")
            self.texto_resultados.insert(tk.END, f"Láminas utilizadas: {len(mejor_individuo.laminas_usadas)}\n")
            
            # Calcular eficiencia
            area_piezas = sum(pieza.area * pieza.cantidad for pieza in self.piezas.values())
            area_laminas_usadas = sum(self.laminas[idx].area for idx in mejor_individuo.laminas_usadas)
            eficiencia = (area_piezas / area_laminas_usadas) * 100
            self.texto_resultados.insert(tk.END, f"Eficiencia: {eficiencia:.2f}%\n")
            
            # Graficar evolución del fitness
            self.ax_fitness.clear()
            generaciones = range(len(self.ag.historia_fitness))
            mejor_fitness = [f[0] for f in self.ag.historia_fitness]
            promedio_fitness = [f[1] for f in self.ag.historia_fitness]
            
            self.ax_fitness.plot(generaciones, mejor_fitness, 'b-', label='Mejor Fitness')
            self.ax_fitness.plot(generaciones, promedio_fitness, 'r-', label='Fitness Promedio')
            self.ax_fitness.set_xlabel('Generación')
            self.ax_fitness.set_ylabel('Fitness')
            self.ax_fitness.set_title('Evolución del Fitness')
            self.ax_fitness.legend()
            self.ax_fitness.grid(True)
            self.canvas_fitness.draw()
            
            # Actualizar lista de láminas en pestaña de visualización
            self.lista_laminas_vis.delete(0, tk.END)
            for idx in sorted(mejor_individuo.laminas_usadas):
                self.lista_laminas_vis.insert(tk.END, f"Lámina {idx+1}")
            
            messagebox.showinfo("Éxito", "Algoritmo Genético ejecutado correctamente")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al ejecutar el Algoritmo Genético: {str(e)}")
            
    def guardar_imagenes(self):
        try:
            if not self.ag or not self.ag.mejor_individuo:
                messagebox.showerror("Error", "No hay resultados para guardar")
                return
            
            # Elegir carpeta para guardar
            carpeta = filedialog.askdirectory(title="Seleccionar Carpeta para Guardar Imágenes")
            
            if carpeta:
                # Generar imágenes
                self.ag.generar_imagenes(carpeta)
                messagebox.showinfo("Éxito", f"Imágenes guardadas correctamente en {carpeta}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar las imágenes: {str(e)}")
            
    def mostrar_lamina(self, event):
        try:
            if not self.ag or not self.ag.mejor_individuo:
                return
                
            # Obtener índice seleccionado
            seleccion = self.lista_laminas_vis.curselection()
            if not seleccion:
                return
                
            # Obtener índice de la lámina
            idx_lamina = sorted(list(self.ag.mejor_individuo.laminas_usadas))[seleccion[0]]
            lamina = self.laminas[idx_lamina]
            
            # Limpiar figura
            self.ax_lamina.clear()
            
            # Dibujar lámina
            self.ax_lamina.add_patch(patches.Rectangle((0, 0), lamina.ancho, lamina.alto, 
                                      linewidth=2, edgecolor='black', facecolor='none'))
            
            # Dibujar piezas
            colores = plt.cm.tab20(np.linspace(0, 1, len(self.piezas)))
            for pos in self.ag.mejor_individuo.posiciones:
                if pos.lamina_idx == idx_lamina:
                    pieza = self.piezas[pos.pieza_id]
                    color_idx = int(pos.pieza_id) % len(colores)
                    self.ax_lamina.add_patch(patches.Rectangle((pos.x, pos.y), pieza.ancho, pieza.alto, 
                                            linewidth=1, edgecolor='black', 
                                            facecolor=colores[color_idx]))
                    # Añadir texto con el ID de la pieza
                    self.ax_lamina.text(pos.x + pieza.ancho/2, pos.y + pieza.alto/2, str(pos.pieza_id),
                                    fontsize=8, ha='center', va='center')
            
            # Configurar límites y etiquetas
            self.ax_lamina.set_xlim(0, lamina.ancho)
            self.ax_lamina.set_ylim(0, lamina.alto)
            self.ax_lamina.set_aspect('equal')
            self.ax_lamina.set_title(f"Lámina {idx_lamina+1}: {lamina.ancho}x{lamina.alto}")
            self.ax_lamina.set_xlabel("Ancho")
            self.ax_lamina.set_ylabel("Alto")
            
            # Actualizar canvas
            self.canvas_lamina.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al mostrar la lámina: {str(e)}")

# Función principal
def main():
    root = tk.Tk()
    app = AplicacionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()