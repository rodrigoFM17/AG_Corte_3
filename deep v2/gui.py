# gui.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import matplotlib.pyplot as plt
from genetic_algorithm import GeneticAlgorithm, Individual
import random
import os

class CuttingApp:
    def __init__(self):
        self.root = tk.Tk()
        self.sheets = pd.DataFrame(columns=['Ancho', 'Alto'])
        self.pieces = pd.DataFrame(columns=['Ancho', 'Alto', 'Cantidad'])
        self.setup_ui()
        self.current_results = []
        
    def setup_ui(self):
        self.root.title("Optimización de Corte - Algoritmo Genético")
        self.root.geometry("800x600")
        
        # Panel de configuración
        config_frame = ttk.LabelFrame(self.root, text="Configuración")
        config_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Sección de láminas
        ttk.Label(config_frame, text="Láminas:").grid(row=0, column=0, sticky=tk.W)
        self.sheet_tree = ttk.Treeview(config_frame, columns=('Ancho', 'Alto'), show='headings', height=4)
        self.sheet_tree.grid(row=1, column=0, columnspan=3, padx=5)
        self.sheet_tree.heading('Ancho', text='Ancho (cm)')
        self.sheet_tree.heading('Alto', text='Alto (cm)')
        
        ttk.Label(config_frame, text="Ancho:").grid(row=2, column=0)
        self.sheet_width = ttk.Entry(config_frame, width=8)
        self.sheet_width.grid(row=2, column=1)
        
        ttk.Label(config_frame, text="Alto:").grid(row=2, column=2)
        self.sheet_height = ttk.Entry(config_frame, width=8)
        self.sheet_height.grid(row=2, column=3)
        
        ttk.Button(config_frame, text="Agregar lámina", command=self.add_sheet).grid(row=2, column=4, padx=5)
        ttk.Button(config_frame, text="Eliminar selección", command=self.remove_sheet).grid(row=2, column=5)
        
        # Sección de piezas
        ttk.Label(config_frame, text="Piezas:").grid(row=3, column=0, sticky=tk.W, pady=(10,0))
        self.piece_tree = ttk.Treeview(config_frame, columns=('Ancho', 'Alto', 'Cantidad'), show='headings', height=4)
        self.piece_tree.grid(row=4, column=0, columnspan=3, padx=5)
        self.piece_tree.heading('Ancho', text='Ancho (cm)')
        self.piece_tree.heading('Alto', text='Alto (cm)')
        self.piece_tree.heading('Cantidad', text='Cantidad')
        
        ttk.Label(config_frame, text="Ancho:").grid(row=5, column=0)
        self.piece_width = ttk.Entry(config_frame, width=8)
        self.piece_width.grid(row=5, column=1)
        
        ttk.Label(config_frame, text="Alto:").grid(row=5, column=2)
        self.piece_height = ttk.Entry(config_frame, width=8)
        self.piece_height.grid(row=5, column=3)
        
        ttk.Label(config_frame, text="Cantidad:").grid(row=5, column=4)
        self.piece_qty = ttk.Entry(config_frame, width=8)
        self.piece_qty.grid(row=5, column=5)
        
        ttk.Button(config_frame, text="Agregar pieza", command=self.add_piece).grid(row=5, column=6, padx=5)
        ttk.Button(config_frame, text="Eliminar selección", command=self.remove_piece).grid(row=5, column=7)
        
        # Controles de archivo
        file_frame = ttk.Frame(config_frame)
        file_frame.grid(row=6, column=0, columnspan=8, pady=10)
        ttk.Button(file_frame, text="Guardar datos", command=self.save_data).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_frame, text="Cargar datos", command=self.load_data).pack(side=tk.LEFT, padx=2)
        
        # Parámetros del AG
        params_frame = ttk.LabelFrame(self.root, text="Parámetros del Algoritmo Genético")
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(params_frame, text="Tamaño población:").grid(row=0, column=0)
        self.pop_size = ttk.Entry(params_frame, width=8)
        self.pop_size.grid(row=0, column=1)
        self.pop_size.insert(0, "50")
        
        ttk.Label(params_frame, text="Generaciones:").grid(row=0, column=2)
        self.generations = ttk.Entry(params_frame, width=8)
        self.generations.grid(row=0, column=3)
        self.generations.insert(0, "100")
        
        ttk.Label(params_frame, text="Penalización:").grid(row=0, column=4)
        self.penalty = ttk.Entry(params_frame, width=8)
        self.penalty.grid(row=0, column=5)
        self.penalty.insert(0, "0.2")
        
        # Botón de ejecución
        ttk.Button(self.root, text="Ejecutar Optimización", command=self.run_optimization).pack(pady=10)
        
        # Consola de resultados
        self.results_text = tk.Text(self.root, height=8)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def add_sheet(self):
        try:
            width = int(self.sheet_width.get())
            height = int(self.sheet_height.get())
            new_sheet = pd.DataFrame([[width, height]], columns=['Ancho', 'Alto'])
            self.sheets = pd.concat([self.sheets, new_sheet], ignore_index=True)
            self.update_sheet_list()
        except ValueError:
            messagebox.showerror("Error", "Valores inválidos para lámina")
            
    def remove_sheet(self):
        selected = self.sheet_tree.selection()
        if selected:
            index = int(selected[0].split('_')[-1])
            self.sheets = self.sheets.drop(index).reset_index(drop=True)
            self.update_sheet_list()
            
    def add_piece(self):
        try:
            width = int(self.piece_width.get())
            height = int(self.piece_height.get())
            qty = int(self.piece_qty.get())
            new_piece = pd.DataFrame([[width, height, qty]], columns=['Ancho', 'Alto', 'Cantidad'])
            self.pieces = pd.concat([self.pieces, new_piece], ignore_index=True)
            self.update_piece_list()
        except ValueError:
            messagebox.showerror("Error", "Valores inválidos para pieza")
            
    def remove_piece(self):
        selected = self.piece_tree.selection()
        if selected:
            index = int(selected[0].split('_')[-1])
            self.pieces = self.pieces.drop(index).reset_index(drop=True)
            self.update_piece_list()
    
    def update_sheet_list(self):
        self.sheet_tree.delete(*self.sheet_tree.get_children())
        for i, row in self.sheets.iterrows():
            self.sheet_tree.insert('', 'end', f'sheet_{i}', values=(row['Ancho'], row['Alto']))
            
    def update_piece_list(self):
        self.piece_tree.delete(*self.piece_tree.get_children())
        for i, row in self.pieces.iterrows():
            self.piece_tree.insert('', 'end', f'piece_{i}', values=(row['Ancho'], row['Alto'], row['Cantidad']))
    
    def save_data(self):
        try:
            folder = filedialog.askdirectory()
            if folder:
                self.sheets.to_csv(os.path.join(folder, 'sheets.csv'), index=False)
                self.pieces.to_csv(os.path.join(folder, 'pieces.csv'), index=False)
                messagebox.showinfo("Guardado", "Datos guardados exitosamente")
        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar: {str(e)}")
            
    def load_data(self):
        try:
            folder = filedialog.askdirectory()
            if folder:
                self.sheets = pd.read_csv(os.path.join(folder, 'sheets.csv'))
                self.pieces = pd.read_csv(os.path.join(folder, 'pieces.csv'))
                self.update_sheet_list()
                self.update_piece_list()
                messagebox.showinfo("Carga", "Datos cargados exitosamente")
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar: {str(e)}")
    
    def run_optimization(self):
        if self.sheets.empty or self.pieces.empty:
            messagebox.showerror("Error", "Debe ingresar láminas y piezas primero")
            return
            
        # Preparar datos
        all_pieces = []
        for _, row in self.pieces.iterrows():
            all_pieces.extend([(row['Ancho'], row['Alto'])] * row['Cantidad'])
        
        sheet_size = (self.sheets.iloc[0]['Ancho'], self.sheets.iloc[0]['Alto'])
        
        # Configurar y ejecutar AG
        ga = GeneticAlgorithm(
            pieces=all_pieces,
            sheet_size=sheet_size,
            penalty=float(self.penalty.get()),
            pop_size=int(self.pop_size.get())
        )
        
        population = ga.initialize_population()
        best_fitness = []
        
        for gen in range(int(self.generations.get())):
            for individual in population:
                if not individual.fitness:
                    ga.calculate_fitness(individual)
            
            population = ga.evolve(population)
            best = max(population, key=lambda x: x.fitness)
            best_fitness.append(best.fitness)
            
            # Actualizar progreso
            self.results_text.insert(tk.END, f"Generación {gen+1}: Mejor Fitness = {best.fitness:.2f}\n")
            self.results_text.see(tk.END)
            self.root.update()
        
        # Mostrar y guardar resultados
        best = max(population, key=lambda x: x.fitness)
        self.generate_results(best, sheet_size)
        messagebox.showinfo("Completado", "Optimización finalizada. Resultados guardados en /resultados")
        
    def generate_results(self, individual: Individual, sheet_size):
        if not os.path.exists('resultados'):
            os.makedirs('resultados')
            
        sheets = {}
        for layout in individual.layout:
            sheet_idx, x, y, l, w = layout
            if sheet_idx not in sheets:
                sheets[sheet_idx] = []
            sheets[sheet_idx].append((x, y, l, w))
        
        for sheet_idx, pieces in sheets.items():
            fig, ax = plt.subplots()
            ax.set_xlim(0, sheet_size[0])
            ax.set_ylim(0, sheet_size[1])
            ax.set_title(f"Lámina {sheet_idx + 1}")
            
            for x, y, l, w in pieces:
                rect = plt.Rectangle((x, y), l, w, edgecolor='black', facecolor=random.choice(['#FF0000', '#00FF00', '#0000FF', '#FFFF00']))
                ax.add_patch(rect)
                ax.text(x + l/2, y + w/2, f"{l}x{w}", ha='center', va='center', color='white')
            
            plt.savefig(f'resultados/lamina_{sheet_idx + 1}.png')
            plt.close()

if __name__ == "__main__":
    app = CuttingApp()
    app.root.mainloop()