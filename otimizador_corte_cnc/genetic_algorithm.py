import random
import copy
from common.layout_display import LayoutDisplayMixin
import math
import time
from typing import List, Dict, Any, Tuple

class GeneticAlgorithm(LayoutDisplayMixin):
    def __init__(self, TAM_POP, recortes_disponiveis, sheet_width, sheet_height, 
                 numero_geracoes=100, max_otimizacoes=30):
        print("Algoritmo Genético para Otimização do Corte de Chapa.")
        self.TAM_POP = TAM_POP
        self.recortes_disponiveis = recortes_disponiveis  # Armazena os recortes para uso posterior
        # Ordena por área decrescente
        self.initial_layout = sorted(
            recortes_disponiveis,
            key=lambda r: r.get('largura', r.get('r', 0)) * r.get('altura', r.get('r', 0)),
            reverse=True
        )
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.numero_geracoes = numero_geracoes
        self.max_otimizacoes = max_otimizacoes
        self.POP = []
        self.optimized_layout = None
        self.initialize_population()
    
    def initialize_population(self):
        # Cada indivíduo é uma cópia da lista inicial com parte embaralhada para diversidade
        for _ in range(self.TAM_POP):
            individuo = self.initial_layout[:]  # Cópia rasa
            random.shuffle(individuo[:len(individuo)//2])
            self.POP.append(individuo)
    
    def get_dims(self, rec: Dict[str,Any], rot: int) -> Tuple[float,float]:
        """Retorna (largura, altura) considerando o tipo da peça e a rotação (0 ou 90)."""
        tipo = rec["tipo"]
        if tipo == "circular":
            d = 2 * rec["r"]
            return (d, d)
        elif tipo in ("retangular", "diamante"):
            if rot == 90:
                return (rec["altura"], rec["largura"])
            else:
                return (rec["largura"], rec["altura"])
        elif tipo == "triangular":
            return (rec["b"], rec["h"])
        else:
            return (rec.get("largura", 10), rec.get("altura", 10))
    
    def decode_layout(self, individual: List[Dict[str,Any]]) -> Tuple[List[Dict[str,Any]], int]:
    
        layout_result: List[Dict[str,Any]] = []
        discarded = 0
        # Lista de retângulos livres: (x, y, largura, altura)
        free_rects: List[Tuple[float,float,float,float]] = [(0, 0, self.sheet_width, self.sheet_height)]
        
        for rec in individual:
            possible_configs = []
            if rec["tipo"] in ("retangular", "diamante"):
                for rot in [0, 90]:
                    w, h = self.get_dims(rec, rot)
                    possible_configs.append((rot, w, h))
            else:
                w, h = self.get_dims(rec, 0)
                possible_configs.append((0, w, h))
            
            placed = False
            for (rot, w, h) in possible_configs:
                best_index = -1
                for i, (rx, ry, rw, rh) in enumerate(free_rects):
                    if w <= rw and h <= rh:
                        best_index = i
                        break
                if best_index != -1:
                    placed = True
                    r_final = copy.deepcopy(rec)
                    r_final["rotacao"] = rot
                    (rx, ry, rw, rh) = free_rects[best_index]
                    r_final["x"] = rx
                    r_final["y"] = ry
                    layout_result.append(r_final)
                    del free_rects[best_index]
                    # Retângulo à direita
                    if w < rw:
                        newW = rw - w
                        free_rects.append((rx + w, ry, newW, rh))
                    # Retângulo abaixo
                    if h < rh:
                        newH = rh - h
                        free_rects.append((rx, ry + h, w, newH))
                    break
            if not placed:
                discarded += 1
        
        return layout_result, discarded

    def evaluate(self, individual: List[Dict[str,Any]]) -> float:
    
        layout, discarded = self.decode_layout(individual)
        if not layout:
            return self.sheet_width * self.sheet_height * 2 + discarded * 10000

        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')
        for rec in layout:
            rot = rec.get("rotacao", 0)
            w, h = self.get_dims(rec, rot)
            x0, y0 = rec["x"], rec["y"]
            x1, y1 = x0 + w, y0 + h
            x_min = min(x_min, x0)
            y_min = min(y_min, y0)
            x_max = max(x_max, x1)
            y_max = max(y_max, y1)
        area_layout = (x_max - x_min) * (y_max - y_min)
        penalty = discarded * 10000
        return area_layout + penalty

    def selection(self) -> List[List[Dict[str,Any]]]:
        self.POP.sort(key=self.evaluate)
        return self.POP[:self.TAM_POP // 2] + [random.choice(self.POP) for _ in range(self.TAM_POP // 2)]
    
    def crossover(self, parent1: List[Dict[str,Any]], parent2: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        cut = random.randint(1, len(parent1) - 1)
        child = parent1[:cut] + [p for p in parent2 if p not in parent1[:cut]]
        return child
    
    def mutate(self, individual: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        if random.random() < 0.2:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        if random.random() < 0.1:
            for rec in individual:
                if 'rotacao' in rec:
                    rec['rotacao'] = random.choice([0, 90])
        return individual
    
    def genetic_operators(self):
        new_population = []
        selected = self.selection()
        while len(new_population) < self.TAM_POP:
            parent1, parent2 = random.sample(selected, 2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        self.POP = new_population

    def run(self) -> List[Dict[str,Any]]:
        best_individual = None
        best_fitness = float('inf')
        for opt in range(self.max_otimizacoes):
            for gen in range(self.numero_geracoes):
                self.genetic_operators()
                current_best = min([self.evaluate(ind) for ind in self.POP])
                if gen % 10 == 0:
                    print(f"Otimização {opt}, Geração {gen}")
            for individual in self.POP:
                fit = self.evaluate(individual)
                if fit < best_fitness:
                    best_fitness = fit
                    best_individual = individual[:]
        layout, _ = self.decode_layout(best_individual)
        self.optimized_layout = layout
        print(f"Melhor Fitness Final: {best_fitness}")
        return self.optimized_layout
    
    def optimize_and_display(self):
        start_time = time.time()
        self.display_layout(self.recortes_disponiveis, title="Initial Layout - Genetic Algorithm")
        self.run()
        self.display_layout(self.optimized_layout, title="Optimized Layout - Genetic Algorithm")
        processing_time = time.time() - start_time
        print(f"Tempo de Processamento: {processing_time:.2f} segundos.", flush=True)

        return self.optimized_layout
