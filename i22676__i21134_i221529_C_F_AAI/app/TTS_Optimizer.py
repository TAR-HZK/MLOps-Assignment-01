import random
import re
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)

class TTSGeneticAlgorithm:
    # Genetic Algorithm for TTS parameter optimization
    #  Applying genetic algorithm
    #  get the best possible path
    #  genes, mutation concepts
    
    def __init__(self, population_size=50, generations=100, mutation_rate=0.1, crossover_rate=0.8):
        self.parameter_ranges = {
            'pitch': (75, 300),
            'speed': (0.5, 2.0),
            'volume': (0.1, 2.0),
            'voice_clarity': (0.1, 1.0),
            'pause_duration': (0.1, 1.5)
        }
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.best_params = None

    def evolve(self, text, target_params=None):
        """Run evolution process"""
        population = self._initialize_population()
        for _ in range(self.generations):
            fitnesses = [self._fitness(ind, text, target_params) for ind in population]
            population = self._next_generation(population, fitnesses)
        return self.best_params

    def _fitness(self, individual, text, target_params):
        """Calculate individual fitness"""
        if target_params:
            return sum(1 - abs(individual[k]-v)/(max_-min_) 
                      for (k, (min_, max_)), v in zip(self.parameter_ranges.items(), target_params.values()))/len(target_params)
        else:
            complexity = self._text_complexity(text)
            score = 100 - abs(individual['speed']-(1+complexity*0.5))*20
            score -= abs(individual['pause_duration']-(0.5+complexity*0.7))*15
            return max(0, score)

    def _text_complexity(self, text):
        """Estimate text complexity"""
        words = word_tokenize(text.lower())
        avg_word_len = sum(len(w) for w in words)/len(words) if words else 0
        sentences = re.split(r'[.!?]+', text)
        avg_sent_len = sum(len(s.split()) for s in sentences)/len(sentences) if sentences else 0
        return min(1.0, (avg_word_len/10 + avg_sent_len/30)/2)

    def _initialize_population(self):
        return [self._create_individual() for _ in range(self.population_size)]

    def _create_individual(self):
        return {k: min_ + random.random()*(max_-min_) for k, (min_, max_) in self.parameter_ranges.items()}

    def _next_generation(self, population, fitnesses):
        new_pop = sorted(zip(population, fitnesses), key=lambda x: -x[1])[:2]
        while len(new_pop) < self.population_size:
            parents = random.choices(population, weights=fitnesses, k=2)
            child = self._crossover(parents[0], parents[1])
            new_pop.append(self._mutate(child))
        self.best_params = new_pop[0][0]
        return [ind for ind, _ in new_pop]

    def _crossover(self, parent1, parent2):
        return {k: parent1[k] if random.random() < 0.5 else parent2[k] for k in self.parameter_ranges}

    def _mutate(self, individual):
        return {k: max(min_, min(max_, v + (random.random()-0.5)*(max_-min_)*0.2))
                if random.random() < self.mutation_rate else v
                for (k, (min_, max_)), v in zip(self.parameter_ranges.items(), individual.values())}