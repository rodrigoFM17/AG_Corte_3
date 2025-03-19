from collections import Counter

class Individuo:
    def __init__(self, piezas_candidatas, lamina_ancho, lamina_alto):
        self.piezas = sorted(piezas_candidatas, 
                           key=lambda x: x.area(), 
                           reverse=True)
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
        
        for pieza in piezas_por_ubicar:
            ubicada = False
            # Intentar colocar en todas las posiciones posibles
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
                    ancho = 0
                    alto = 0
                    # Buscar límites del área residual
                    for i in range(y, matriz.shape[0]):
                        if matriz[i, x] == 0:
                            alto += 1
                        else:
                            break
                    for j in range(x, matriz.shape[1]):
                        if matriz[y, j] == 0:
                            ancho += 1
                        else:
                            break
                    areas.append(ancho * alto)
                    visitados[y:y+alto, x:x+ancho] = 1
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

        tipos_originales = Counter(p.tipo_id for p in self.piezas)
        tipos_actuales = Counter(p.tipo_id for p in self.piezas)
        
        if tipos_originales != tipos_actuales:
            self.fitness = -999999  # Penalización fuerte
            return self.fitness
        
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
        
        # Cálculo de fitness compuesto
        utilization = total_area_utilizada / total_area_piezas
        residual_score = (max_residual + (total_residual / 10)) / (self.lamina_ancho * self.lamina_alto)
        
        self.fitness = (utilization * 0.7) + (residual_score * 0.3) - penalizacion_laminas
        self.num_laminas = len(self.distribuciones)
        return self.fitness