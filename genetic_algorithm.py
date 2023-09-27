import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
distance_matrix = pd.read_csv('dataset/Distance_Matrix.csv').drop('Unnamed: 0', axis=1)
biomass_history = pd.read_csv('dataset/Biomass_History.csv')
# CONSTANTS
num_facilities = 25 # Number of facilities
num_customers = 2417 # Number of customers
# CLASSES
class Facility:
    _id=0
    C = 2000
    def __init__(self, x, y):
        self.id = Facility._id
        self.name = f"Facility{self.id}"
        Facility._id += 1
        self.capacity = Facility.C

        self.x = x
        self.y = y

        self.biomass = 0

class Customer:
    _id=0
    def __init__(self, x, y, biomass):
        self.id = Customer._id
        self.name = f"Customer{self.id}"
        Customer._id += 1
        self.biomass = biomass

        self.x = x
        self.y = y

class Chromosome:
    _id=0
    def __init__(self, genes):
        self.id = Chromosome._id
        self.name = f"Chromosome{self.id}"
        Chromosome._id += 1
        self.genes = genes
        self.fitness = 0

    def __repr__(self):
        return f"Chromosome: {self.genes} Fitness: {self.fitness}"
    
    def __str__(self):
        return f"Chromosome: {self.genes} Fitness: {self.fitness}"
    
    def __eq__(self, other):
        return self.genes == other.genes
    
    def __hash__(self):
        return hash(self.genes)
    
    def __getitem__(self, key):
        return self.genes[key]
    
    def __setitem__(self, key, value):
        self.genes[key] = value

class Population:
    def __init__(self, chromosomes):
        self.chromosomes = chromosomes
        self.fitness = 0
        self.best_chromosome = None

    # enxute print
    def __repr__(self):
        return f"Population size: {len(self.chromosomes)} Fitness: {self.fitness} Best Chromosome: {self.best_chromosome.name}"
    
    def __str__(self):
        return f"Population size: {len(self.chromosomes)} Fitness: {self.fitness} Best Chromosome: {self.best_chromosome.name}"
    
    def __getitem__(self, key):
        return self.chromosomes[key]
    
    def __setitem__(self, key, value):
        self.chromosomes[key] = value

# FUNCTIONS
def generate_facilities():
    facilities = []
    for i in range(num_facilities):
        facilities.append(Facility(random.randint(0, 100), random.randint(0, 100)))
    return facilities

def generate_customers():
    customers = []
    for i in range(num_customers):
        customers.append(Customer(random.randint(0, 100), random.randint(0, 100), random.randint(0, 100)))
    return customers

def generate_chromosome():
    np.random.seed(0)
    # define medians
    medians = np.random.randint(0, num_customers, (num_facilities, 1))
    # define chromosome vector choosing randomly from medians for each vertex 
    chromosome_vector = medians[np.random.randint(0, len(medians), (num_customers, 1)).squeeze()]
    # set medians to -1
    chromosome_vector[medians.squeeze()] = -1

    chromosome = Chromosome(chromosome_vector)
    return chromosome

def generate_population():
    chromosomes = []
    for i in range(100):
        chromosomes.append(Chromosome(generate_chromosome()))
    return Population(chromosomes)

def calculate_fitness(population, facilities, customers):
    for chromosome in population.chromosomes:
        for facility in facilities:
            facility.biomass = 0
        for i in range(num_customers):
            facilities[chromosome[i]].biomass += customers[i].biomass
        for facility in facilities:
            if facility.biomass > facility.capacity:
                chromosome.fitness = 0
                break
            else:
                chromosome.fitness += facility.biomass
    population.fitness = sum([chromosome.fitness for chromosome in population.chromosomes])
    population.best_chromosome = max(population.chromosomes, key=lambda chromosome: chromosome.fitness)
    return population

def crossover(population):
    new_chromosomes = []
    for i in range(100):
        parent1 = random.choice(population.chromosomes)
        parent2 = random.choice(population.chromosomes)
        child1 = Chromosome(parent1.genes[:])
        child2 = Chromosome(parent2.genes[:])
        for j in range(num_customers):
            if random.random() < 0.5:   
                child1[j], child2[j] = child2[j], child1[j]
        new_chromosomes.append(child1)
        new_chromosomes.append(child2)
    return Population(new_chromosomes)

def mutate(population):
    for chromosome in population.chromosomes:
        for i in range(num_customers):
            if random.random() < 0.01:
                chromosome[i] = random.randint(0, num_facilities-1)
    return population

def main():
    print(generate_chromosome())

if __name__ == "__main__":
    main()
