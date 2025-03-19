import tkinter as tk
from tkinter import ttk, messagebox
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv
import os
import numpy as np
from copy import deepcopy
from Pieza import Pieza
from Individuo import Individuo

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
    global piezas_seleccionadas, tipo_counter
    try:
        with open("piezas.csv", "r") as file:
            reader = csv.reader(file)
            piezas_seleccionadas = []
            for row in reader:
                if row:
                    tipo_id = int(row[0])
                    ancho = int(row[1])
                    alto = int(row[2])
                    cantidad = int(row[3])
                    
                    piezas_seleccionadas.append({
                        'tipo_id': tipo_id,
                        'dimensiones': (ancho, alto),
                        'cantidad': cantidad
                    })
                    tipo_counter = max(tipo_counter, tipo_id + 1)
    except FileNotFoundError:
        piezas_seleccionadas = []

# Guardar datos en CSV
def guardar_datos():
    # Guardar láminas normal
    with open("laminas.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(laminas)
    
    # Guardar piezas con nueva estructura
    with open("piezas.csv", "w", newline='') as file:
        writer = csv.writer(file)
        for p in piezas_seleccionadas:
            writer.writerow([
                p['tipo_id'],
                p['dimensiones'][0],
                p['dimensiones'][1],
                p['cantidad']
            ])

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

# Variable global para llevar el conteo de tipos de piezas
tipo_counter = 1

def agregar_pieza():
    global tipo_counter, piezas_seleccionadas
    
    try:
        ancho = int(entry_ancho_pieza.get())
        alto = int(entry_alto_pieza.get())
        cantidad = int(entry_cantidad_pieza.get())
        
        # Agregar a la lista de piezas con tipo_id
        piezas_seleccionadas.append({
            'tipo_id': tipo_counter,
            'dimensiones': (ancho, alto),
            'cantidad': cantidad
        })
        
        # Actualizar lista en GUI
        lista_piezas.insert(tk.END, f"Tipo {tipo_counter}: {ancho}x{alto} x{cantidad}")
        tipo_counter += 1
        
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

def actualizar_tabla_piezas():
    for row in tabla_piezas.get_children():
        tabla_piezas.delete(row)
    for pieza in piezas_seleccionadas:
        dim = pieza['dimensiones']
        tabla_piezas.insert("", tk.END, values=(
            f"Tipo {pieza['tipo_id']}",
            f"{dim[0]}x{dim[1]}",
            pieza['cantidad']
        ))


# Funciones para el algoritmo genético
def crear_poblacion_inicial(piezas_seleccionadas, lamina_ancho, lamina_alto):
    poblacion = []
    
    # Generar plantilla base de piezas
    todas_piezas = []
    tipo_id = 0
    for (ancho, alto), cantidad in piezas_seleccionadas:
        tipo_id += 1
        for i in range(cantidad):
            todas_piezas.append(Pieza(ancho, alto, tipo_id, id=f"{tipo_id}-{i}"))
    
    # Crear población
    for _ in range(POP_SIZE):
        # Hacer copia profunda y barajar
        piezas_individuo = [p.copia() for p in todas_piezas]
        random.shuffle(piezas_individuo)
        
        # Aplicar rotaciones iniciales
        for p in piezas_individuo:
            if random.random() < 0.5 and p.ancho != p.alto:
                p.rotar()
        
        poblacion.append(Individuo(piezas_individuo, lamina_ancho, lamina_alto))
    
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
    if random.random() > CROSSOVER_RATE:
        return padre1.copia(), padre2.copia()

    # Cruce ordenado que preserva tipos y cantidades
    tipo_counts = {}
    for p in padre1.piezas:
        tipo_counts[p.tipo_id] = tipo_counts.get(p.tipo_id, 0) + 1

    hijo1_piezas = []
    hijo2_piezas = []
    
    # Estrategia de cruce por tipo
    for tipo_id in tipo_counts:
        piezas_p1 = [p for p in padre1.piezas if p.tipo_id == tipo_id]
        piezas_p2 = [p for p in padre2.piezas if p.tipo_id == tipo_id]
        
        # Cruzar por tipo
        punto = random.randint(1, len(piezas_p1)-1)
        hijo1_tipo = piezas_p1[:punto] + piezas_p2[punto:]
        hijo2_tipo = piezas_p2[:punto] + piezas_p1[punto:]
        
        hijo1_piezas.extend(hijo1_tipo)
        hijo2_piezas.extend(hijo2_tipo)

    # Barajar manteniendo integridad de tipos
    random.shuffle(hijo1_piezas)
    random.shuffle(hijo2_piezas)

    return (
        Individuo(hijo1_piezas, padre1.lamina_ancho, padre1.lamina_alto),
        Individuo(hijo2_piezas, padre2.lamina_ancho, padre2.lamina_alto)
    )

def mutacion(individuo):
    # Solo permite intercambiar piezas del mismo tipo
    piezas_por_tipo = {}
    for p in individuo.piezas:
        if p.tipo_id not in piezas_por_tipo:
            piezas_por_tipo[p.tipo_id] = []
        piezas_por_tipo[p.tipo_id].append(p)
    
    # Mutar por tipo
    for tipo_id, piezas in piezas_por_tipo.items():
        if len(piezas) > 1 and random.random() < MUTATION_RATE:
            idx1, idx2 = random.sample(range(len(piezas)), 2)
            # Intercambiar posición
            piezas[idx1], piezas[idx2] = piezas[idx2], piezas[idx1]
            # Rotar aleatoriamente
            if random.random() < 0.5:
                piezas[idx1].rotar()
            if random.random() < 0.5:
                piezas[idx2].rotar()
    
    return individuo

def algoritmo_genetico(piezas_seleccionadas, lamina_ancho, lamina_alto):
    """Algoritmo genético principal"""
    # Crear población inicial
    poblacion = crear_poblacion_inicial(piezas_seleccionadas, lamina_ancho, lamina_alto)
    
    # Evolución
    mejor_individuo = max(poblacion, key=lambda x: x.fitness)
    mejores_fitness = [mejor_individuo.fitness]

    def evaluar_individuo(ind):
        return ind.fitness * (1 - 0.05 * ind.num_laminas)
    
    for generacion in range(GENERATIONS):
        # Selección usando la nueva métrica
        seleccionados = []
        for _ in range(POP_SIZE):
            competidores = random.sample(poblacion, 3)
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
        for i in range(POP_SIZE):
            nueva_poblacion[i] = mutacion(nueva_poblacion[i])

        if mejor_individuo.fitness >= 0.95 and mejor_individuo.num_laminas == 1:
            print("Solución óptima encontrada, terminando temprano")
            break

        
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
    global mejor_individuo_global
    
    if not lamina_seleccionada:
        messagebox.showerror("Error", "Seleccione una lámina")
        return
    if not piezas_seleccionadas:
        messagebox.showerror("Error", "Seleccione al menos una pieza")
        return
    

    # Validar parámetros del AG
    try:
        actualizar_parametros_ag()
    except:
        messagebox.showerror("Error", "Valores de parámetros inválidos")
        return
    
    piezas_para_ag = [
        (p['dimensiones'], p['cantidad'], p['tipo_id'])
        for p in piezas_seleccionadas
    ]


    poblacion = crear_poblacion_inicial(piezas_para_ag, lamina_ancho, lamina_alto)

    # Crear lista de todas las piezas con IDs únicos
    todas_piezas = []
    id_counter = 0
    for (ancho, alto), cantidad in piezas_seleccionadas:
        for _ in range(cantidad):
            todas_piezas.append(Pieza(ancho, alto, id=id_counter))
            id_counter += 1

    # Verificar que las piezas quepan en la lámina
    lamina_ancho, lamina_alto = lamina_seleccionada
    max_lamina_dim = max(lamina_ancho, lamina_alto)
    min_lamina_dim = min(lamina_ancho, lamina_alto)
    
    for pieza in todas_piezas:
        pieza_min_dim = min(pieza.ancho, pieza.alto)
        pieza_max_dim = max(pieza.ancho, pieza.alto)
        
        if pieza_max_dim > max_lamina_dim or pieza_min_dim > min_lamina_dim:
            messagebox.showerror("Error", 
                f"La pieza {pieza.ancho}x{pieza.alto} no cabe en la lámina\n" 
                f"Dimensiones máximas permitidas: {max_lamina_dim}x{min_lamina_dim}")
            return

    # Configurar ventana de progreso
    ventana_progreso = tk.Toplevel(root)
    ventana_progreso.title("Progreso del Algoritmo Genético")
    ventana_progreso.geometry("400x150")
    
    lbl_progreso = tk.Label(ventana_progreso, text="Inicializando algoritmo...")
    lbl_progreso.pack(pady=10)
    
    barra_progreso = ttk.Progressbar(ventana_progreso, length=300, mode='determinate')
    barra_progreso.pack(pady=10)

    # Función para actualizar progreso
    def actualizar_progreso(generacion, fitness, laminas):
        barra_progreso['value'] = (generacion / GENERATIONS) * 100
        lbl_progreso.config(text=f"Generación {generacion}/{GENERATIONS}\n"
                               f"Fitness: {fitness:.2f} - Láminas: {laminas}")
        ventana_progreso.update()

    # Ejecutar en segundo plano para no bloquear la GUI
    def ejecucion_background():
        nonlocal todas_piezas
        try:
            # Crear población inicial corregida
            poblacion = []
            for _ in range(POP_SIZE):
                # Crear copias de las piezas para cada individuo
                piezas_individuo = [pieza.copia() for pieza in todas_piezas]
                
                # Aplicar rotaciones aleatorias
                for p in piezas_individuo:
                    if random.random() < 0.5 and p.ancho != p.alto:
                        p.rotar()
                
                # Barajar orden de colocación
                random.shuffle(piezas_individuo)
                
                # Ordenar por tamaño (área descendente)
                piezas_individuo.sort(key=lambda x: x.area(), reverse=True)
                
                individuo = Individuo(piezas_individuo, lamina_ancho, lamina_alto)
                individuo.calcular_fitness()
                poblacion.append(individuo)

            mejor_individuo = max(poblacion, key=lambda x: x.fitness)
            historico_fitness = [mejor_individuo.fitness]

            # Evolución
            for generacion in range(GENERATIONS):
                # Selección por torneo
                seleccionados = seleccion_torneo(poblacion)
                
                # Cruce
                nueva_poblacion = []
                for i in range(0, POP_SIZE, 2):
                    if i+1 < POP_SIZE:
                        padre1 = seleccionados[i]
                        padre2 = seleccionados[i+1]
                        hijo1, hijo2 = cruce(padre1, padre2)
                        nueva_poblacion.extend([hijo1, hijo2])
                    else:
                        nueva_poblacion.append(seleccionados[i])
                
                # Mutación
                for i in range(POP_SIZE):
                    nueva_poblacion[i] = mutacion(nueva_poblacion[i])
                    nueva_poblacion[i].calcular_fitness()
                
                # Elitismo
                mejor_actual = max(nueva_poblacion, key=lambda x: x.fitness)
                if mejor_actual.fitness < mejor_individuo.fitness:
                    peor_idx = min(range(POP_SIZE), key=lambda i: nueva_poblacion[i].fitness)
                    nueva_poblacion[peor_idx] = mejor_individuo
                
                poblacion = nueva_poblacion
                mejor_individuo = max(poblacion, key=lambda x: x.fitness)
                historico_fitness.append(mejor_individuo.fitness)
                
                # Actualizar interfaz
                actualizar_progreso(generacion, 
                                   mejor_individuo.fitness, 
                                   len(mejor_individuo.distribuciones))

            # Resultados finales
            ventana_progreso.destroy()
            mejor_individuo_global = mejor_individuo
            
            # Mostrar estadísticas
            area_total = sum(lamina_ancho * lamina_alto for _ in mejor_individuo.distribuciones)
            area_usada = sum(p.ancho * p.alto for dist in mejor_individuo.distribuciones for p in dist['piezas'])
            
            messagebox.showinfo("Resultado",
                f"Láminas requeridas: {len(mejor_individuo.distribuciones)}\n"
                f"Aprovechamiento promedio: {(area_usada / area_total) * 100:.1f}%")
            
            # Generar gráficos
            mostrar_resultado(mejor_individuo)
            mostrar_grafica_evolucion(historico_fitness)

        except Exception as e:
            ventana_progreso.destroy()
            messagebox.showerror("Error", f"Error durante la ejecución:\n{str(e)}")
            raise e

    # Ejecutar en un hilo separado
    import threading
    threading.Thread(target=ejecucion_background, daemon=True).start()

# Función para mostrar el resultado gráfico
def mostrar_resultado(individuo):
    if not individuo.distribuciones:
        messagebox.showwarning("Advertencia", "No hay distribución para mostrar")
        return
    
    # Crear carpeta para guardar las imágenes
    output_dir = "resultados_lamina"
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, distribucion in enumerate(individuo.distribuciones):
        # Crear figura individual para cada lámina
        fig = plt.figure(figsize=(12, 6))
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
        
        # Dibujar áreas residuales
        visitados = np.zeros_like(distribucion['matriz'])
        for y in range(distribucion['matriz'].shape[0]):
            for x in range(distribucion['matriz'].shape[1]):
                if distribucion['matriz'][y, x] == 0 and visitados[y, x] == 0:
                    ancho = 0
                    alto = 0
                    for i in range(y, distribucion['matriz'].shape[0]):
                        if distribucion['matriz'][i, x] == 0:
                            alto += 1
                        else:
                            break
                    for j in range(x, distribucion['matriz'].shape[1]):
                        if distribucion['matriz'][y, j] == 0:
                            ancho += 1
                        else:
                            break
                    ax.add_patch(patches.Rectangle(
                        (x, y), ancho, alto,
                        edgecolor='gray', facecolor='lightgray', alpha=0.3
                    ))
                    visitados[y:y+alto, x:x+ancho] = 1
        
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

frame_seleccionar_pieza = tk.LabelFrame(frame_der, text="Seleccionar Pieza")
frame_seleccionar_pieza.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

# Nueva entrada para cantidad
tk.Label(frame_seleccionar_pieza, text="Cantidad:").pack(anchor=tk.W, padx=5)
entry_cantidad_pieza = tk.Entry(frame_seleccionar_pieza)
entry_cantidad_pieza.pack(fill=tk.X, padx=5, pady=5)
entry_cantidad_pieza.insert(0, "1")

# Botón modificado
btn_agregar_pieza = tk.Button(
    frame_seleccionar_pieza, 
    text="Agregar Tipo de Pieza",
    command=agregar_pieza
)
btn_agregar_pieza.pack(fill=tk.X, padx=5, pady=5)

# Lista modificada
lista_piezas = tk.Listbox(
    frame_piezas, 
    height=10,
    selectmode=tk.SINGLE
)
lista_piezas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

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