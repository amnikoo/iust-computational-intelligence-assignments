import numpy
import random

def cal_pop_fitness(equation_weights, pop):
    fitness = []
    for j in range(len(pop)):
        fitness_member = 0
        for i in range(len(equation_weights)):
            fitness_member += equation_weights[i]*(pop[j]**i)
        fitness.append(abs(int(fitness_member*100)/100))
    return(fitness)

def select_mating_pool(pop, fitness, num_parents):
    f = fitness.copy()
    parents = []
    for j in range(num_parents):
        parents.append(pop[f.index(min(f))])
        f.remove(f[f.index(min(f))])
        pop.remove(pop[f.index(min(f))])
    return(parents, pop)

def crossover(parents, crossover_count):
    for i in range(crossover_count):
        n = random.randint(0, len(parents)//2)
        m = n + len(parents)//2
        l = random.randint(n, len(parents)//2)
        temp = parents[n:l]
        parents[n:l] = parents[m:len(parents)//2+l]
        parents[m:len(parents)//2+l] = temp
    return(parents)

def mutation(offspring_crossover, num_mutations=1):
    i = random.randint(0, len(offspring_crossover)-1)
    for mutation_num in range(num_mutations):
        offspring_crossover[i] = offspring_crossover[i] + int(random.uniform(-4, 4)*100)/100
    return(offspring_crossover)


equation_weights = [-13257.2,15501.2,-7227.94,1680.1,-194.7,9]

sol_per_pop = 20
num_parents_mating = 10

pop_size = sol_per_pop
new_population = []
for i in range(pop_size//2):
    new_population.append(round(random.uniform(-9, 9), 2))
    new_population.append(random.randint(-9, 9))
print(new_population)

best_outputs = []
num_generations = 100
for generation in range(1,num_generations+1):
    print("Generation : ", generation)
    fitness = cal_pop_fitness(equation_weights, new_population)
    print("Fitness")
    print(fitness)

    best_outputs.append(numpy.min(fitness))
    print("Best result : ", numpy.min(fitness))
    
    parents, f = select_mating_pool(new_population, fitness, num_parents_mating)
    print("Parents")
    print(parents)

    offspring_crossover = crossover(parents, crossover_count=5)
    print("Crossover")
    print(offspring_crossover)

    offspring_mutation = mutation(offspring_crossover, num_mutations=2)
    print("Mutation")
    print(offspring_mutation)

    new_population[0:len(parents)] = parents
    new_population[len(parents):] = f
    print(new_population)
    
fitness = cal_pop_fitness(equation_weights, new_population)
best_match_idx = f.index(min(f))

print("Best solution : ", new_population[best_match_idx])
print("Best solution fitness : ", fitness[best_match_idx])
