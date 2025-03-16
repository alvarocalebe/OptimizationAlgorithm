import random
import copy
from common.layout_display import LayoutDisplayMixin

class GeneticAlgorithm(LayoutDisplayMixin):
    def __init__(self, TAM_POP, recortes_disponiveis, sheet_width, sheet_height, numero_geracoes=1000, max_otimizacoes=100):
        print("Algoritmo Genético para Otimização do Corte de Chapa.")
        self.TAM_POP = TAM_POP
        self.initial_layout = sorted(recortes_disponiveis, key=lambda r: r.get('largura', r.get('r', 0)) * r.get('altura', r.get('r', 0)), reverse=True)
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.numero_geracoes = numero_geracoes
        self.max_otimizacoes = max_otimizacoes
        self.POP = []
        self.optimized_layout = None
        self.initialize_population()
    
    def initialize_population(self):
        for _ in range(self.TAM_POP):
            individuo = sorted(self.initial_layout, key=lambda r: (r.get('largura', r.get('r', 0)), r.get('altura', r.get('r', 0))), reverse=True)
            random.shuffle(individuo[:len(individuo)//2])
            self.POP.append(individuo)
    
    def lgfi_heuristic(self, individuo):
        layout = []
        x_offset, y_offset = 0, 0
        for recorte in individuo:
            for _ in range(50):
                recorte['x'] = x_offset
                recorte['y'] = y_offset
                if 'rotacao' in recorte and random.random() < 0.3:
                    recorte['rotacao'] = random.choice([0, 90])
                if not self.has_overlap(recorte, layout):
                    break
            layout.append(copy.deepcopy(recorte))
            x_offset += recorte.get('largura', recorte.get('r', 0)) + 1
            if x_offset >= self.sheet_width:
                x_offset = 0
                y_offset += recorte.get('altura', recorte.get('r', 0)) + 1
        return layout
    
    def has_overlap(self, new_recorte, layout):
        for recorte in layout:
            if not (new_recorte['x'] + new_recorte.get('largura', new_recorte.get('r', 0)) <= recorte['x'] or
                    recorte['x'] + recorte.get('largura', recorte.get('r', 0)) <= new_recorte['x'] or
                    new_recorte['y'] + new_recorte.get('altura', recorte.get('r', 0)) <= recorte['y'] or
                    recorte['y'] + recorte.get('altura', recorte.get('r', 0)) <= new_recorte['y']):
                return True
        return False
    
    def evaluate(self, individuo):
        layout = self.lgfi_heuristic(individuo)
        area_utilizada = sum(r.get('largura', r.get('r', 0)) * r.get('altura', r.get('r', 0)) for r in layout)
        desperdicio = (self.sheet_width * self.sheet_height) - area_utilizada
        return -desperdicio
    
    def selection(self):
        self.POP.sort(key=self.evaluate, reverse=True)
        return self.POP[:self.TAM_POP // 2] + [random.choice(self.POP) for _ in range(self.TAM_POP // 2)]
    
    def crossover(self, parent1, parent2):
        cut = random.randint(1, len(parent1) - 1)
        child = parent1[:cut] + [p for p in parent2 if p not in parent1[:cut]]
        return child
    
    def mutate(self, individuo):
        if random.random() < 0.2:
            idx1, idx2 = random.sample(range(len(individuo)), 2)
            individuo[idx1], individuo[idx2] = individuo[idx2], individuo[idx1]
        if random.random() < 0.1:
            for recorte in individuo:
                if 'rotacao' in recorte:
                    recorte['rotacao'] = random.choice([0, 90])
        return individuo
    
    def genetic_operators(self):
        new_population = []
        selected = self.selection()
        while len(new_population) < self.TAM_POP:
            parent1, parent2 = random.sample(selected, 2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        self.POP = new_population
    
    def run(self):
        melhor_layout = None
        melhor_aptidao = float('-inf')
        for _ in range(self.max_otimizacoes):
            for _ in range(self.numero_geracoes):
                self.POP.sort(key=self.evaluate, reverse=True)
                self.genetic_operators()
            melhor_candidato = self.lgfi_heuristic(self.POP[0])
            aptidao_atual = self.evaluate(melhor_candidato)
            if aptidao_atual > melhor_aptidao:
                melhor_aptidao = aptidao_atual
                melhor_layout = melhor_candidato
        self.optimized_layout = melhor_layout
        return self.optimized_layout
    
    def optimize_and_display(self):
        self.display_layout(self.initial_layout, title="Initial Layout - Genetic Algorithm")
        self.optimized_layout = self.run()
        self.display_layout(self.optimized_layout, title="Optimized Layout - Genetic Algorithm")
        return self.optimized_layout