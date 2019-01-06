import  random
import numpy as np
import matplotlib.pyplot as plt


# Target Function

def f(x):
    return np.sin(10*x)*x + np.cos(20*x)*x

chromosome_length = 10
population_size = 100
cross_rate = 0.8
mutation_rate = 0.02
interval = [0,10]

def fitness(population):
    population = abs(population)
    return population/population.sum()

def select(population, fit):#Roulette Wheel
    indexes = np.random.choice(np.arange(population_size),size=population_size,replace=True,p=fit)
    return population[indexes]

def crossover(chromosome,population):
    if np.random.rand() < cross_rate:
        _ = np.random.randint(0, population_size)
        return (population[_]+chromosome)/2
    return chromosome

def mutate(chromosome):
    if np.random.rand() < mutation_rate:
        if chromosome <= 1 and chromosome >= 0:
            chromosome += random.uniform(-mutation_rate,mutation_rate)
        elif chromosome > 1:
            chromosome -= random.uniform(0,mutation_rate)
        elif chromosome < 0:
            chromosome += random.uniform(0,mutation_rate)
            
    return chromosome
    

population = np.random.random((population_size,))

plt.ion()     
x = np.linspace(0,1, 200)
plt.plot(x, f(x))

for _ in range(100):
    F_ = f(population)
    fit = fitness(F_)
    sca = plt.scatter(population, F_, s=200, lw=0, alpha=0.5); plt.pause(0.05)
    population = select(population, fit)
   
    for chromosome in population:
        child_chr = crossover(chromosome, population)
        child_chr = mutate(child_chr)
        population[ : ] = child_chr
    #print(min(population),f(min(population)))

x = population.sum()/len(population)

#plt.plot(x,f(x),'o')
plt.ioff()
plt.show()
