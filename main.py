import tkinter as tk
from tkinter import ttk, messagebox
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv
import os

# Datos iniciales
laminas = []  # Lista de laminas (ancho, alto)
piezas = []  # Lista de piezas (ancho, alto)
piezas_seleccionadas = []
lamina_seleccionada = None

# Algoritmo Genético Configuración
POP_SIZE = 50  # Tamaño de la población
GENERATIONS = 100  # Número de generaciones
MUTATION_RATE = 0.1  # Probabilidad de mutación

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

# Algoritmo Genético - Simulación
def ejecutar_algoritmo_genetico():
    if not lamina_seleccionada:
        messagebox.showerror("Error", "Seleccione una lámina")
        return
    if not piezas_seleccionadas:
        messagebox.showerror("Error", "Seleccione al menos una pieza")
        return
    
    print("Ejecutando algoritmo genético...")
    
    # Aquí se implementaría el AG para optimizar el corte
    messagebox.showinfo("Info", "Optimización completada")
    
    mostrar_resultado()

# Función para mostrar el resultado gráfico
def mostrar_resultado():
    fig, ax = plt.subplots()
    ax.set_xlim(0, lamina_seleccionada[0])
    ax.set_ylim(0, lamina_seleccionada[1])

    # Dibujar la lámina
    ax.add_patch(patches.Rectangle((0, 0), lamina_seleccionada[0], lamina_seleccionada[1], edgecolor='black', facecolor='none'))

    # Simulación de cortes (random)
    x_offset = 0
    y_offset = 0

    for (pieza, cantidad) in piezas_seleccionadas:
        for _ in range(cantidad):
            if x_offset + pieza[0] > lamina_seleccionada[0]:
                x_offset = 0
                y_offset += pieza[1]
            if y_offset + pieza[1] > lamina_seleccionada[1]:
                break
            ax.add_patch(patches.Rectangle((x_offset, y_offset), pieza[0], pieza[1], edgecolor='blue', facecolor='lightblue'))
            x_offset += pieza[0]

    plt.show()

# Inicializar CSV e interfaz
inicializar_csv()
cargar_datos()

# Interfaz gráfica
root = tk.Tk()
root.title("Optimización de Corte con Algoritmos Genéticos")
root.geometry("850x450")

frame_izq = tk.Frame(root)
frame_izq.pack(side=tk.LEFT, padx=10, pady=10)
frame_der = tk.Frame(root)
frame_der.pack(side=tk.RIGHT, padx=10, pady=10)

# Sección de catálogo
tk.Label(frame_izq, text="Catálogo de Láminas").pack()
lista_laminas = tk.Listbox(frame_izq)
lista_laminas.pack()
tk.Button(frame_izq, text="Seleccionar Lámina", command=seleccionar_lamina).pack()

# Sección de ingreso de nuevas láminas
tk.Label(frame_izq, text="Ancho").pack()
entry_ancho_lamina = tk.Entry(frame_izq)
entry_ancho_lamina.pack()
tk.Label(frame_izq, text="Alto").pack()
entry_alto_lamina = tk.Entry(frame_izq)
entry_alto_lamina.pack()
tk.Button(frame_izq, text="Agregar Lámina", command=agregar_lamina).pack()

# Sección de piezas
tk.Label(frame_der, text="Catálogo de Piezas").pack()
lista_piezas = tk.Listbox(frame_der)
lista_piezas.pack()
tk.Label(frame_der, text="Cantidad").pack()
entry_cantidad_pieza = tk.Entry(frame_der)
entry_cantidad_pieza.pack()
tk.Button(frame_der, text="Seleccionar Pieza", command=seleccionar_pieza).pack()

tabla_piezas = ttk.Treeview(frame_der, columns=("Pieza", "Cantidad"), show="headings")
tabla_piezas.heading("Pieza", text="Pieza")
tabla_piezas.heading("Cantidad", text="Cantidad")
tabla_piezas.pack()

tk.Button(frame_der, text="Ejecutar Algoritmo", command=ejecutar_algoritmo_genetico).pack()

actualizar_listas()
root.mainloop()