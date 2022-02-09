import numpy as np
import random as rd
from random import randint
import matplotlib.pyplot as plt

# initially list an evenly space (table-like) number of items. we set it to start at 1 and stop at not more than or equal to 11
item_number = np.arange(1, 11)

# randomly assign a weight per item with a minimum weight of 1 and maximum of 15
weight = np.random.randint(1, 15, size=10)

# randomly assign a value per item with a minimum value of 10 and maximum of 750
value = np.random.randint(10, 750, size=10)

# weight limit of the knapsack
knapsack_weight_limit = 35

# list and print the items with its respective number, weight and value
print("The list as follows:")
print('Item No.   Weight   Value')
for i in range(item_number.shape[0]):
    print('{0}          {1}         {2}\n'.format(item_number[i], weight[i], value[i]))

# number of solutions in an entire population
solutions_per_pop = 8

# size of the population with an eight solution(chromosome and a solution has a 10 gene.
# gene is the composition of a solution(chromosome) with a binary representation (representation may varie on the the problem that is needed to be solved.)
pop_size = (solutions_per_pop, item_number.shape[0])
print('Population size = {}'.format(pop_size))

# now we set a 10 random numbers per solution from 0 - 1 (binary representation) to generate solution and a population
initial_population = np.random.randint(2, size=pop_size)
initial_population = initial_population.astype(int)
num_generations = 50
print('Initial population: \n{}'.format(initial_population))


# calculation of fitness per solution in the population
def cal_fitness(weight, value, population, weight_limit):
    fitness = np.empty(population.shape[0])
    for i in range(population.shape[0]):
        # total value in a solution
        S1 = np.sum(population[i] * value)
        # total weight in a solution
        S2 = np.sum(population[i] * weight)

        # if the total weight of a solution is not exceeding the defined weight limit, fitness of the solution will be equal to the total value of the solution (S1)
        if S2 <= weight_limit:
            fitness[i] = S1
        # if the total weight exceeds fitness of the solution will be equal to 0
        else:
            fitness[i] = 0
    return fitness.astype(int)


# selection of the fittest solution in the population to undergo crossover
def selection(fitness, num_parents, population):
    fitness = list(fitness)
    parents = np.empty((num_parents, population.shape[1]))

    # loop through population who had fit for the problem and finding num_parent value of  highest fitness in the population to be parent and undergo the crossover function.
    # once the value is found to be highest it will be redefined it value to -99999 for the loop not to be included again as one of the highest value in the population
    for i in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        parents[i, :] = population[max_fitness_idx[0][0], :]
        fitness[max_fitness_idx[0][0]] = -999999
    return parents


# from the selection function we get the parents (highest value or fittest from the population)
# which will undergo the crossover function
def crossover(parents, num_offsprings):
    # initially these are the parents but as the code goes on this will change to new set of genes
    offsprings = np.empty((num_offsprings, parents.shape[1]))

    # here we're defining the crossover point where the number of gene is divided by 2. so the crossover point is in half of the solution.
    crossover_point = int(parents.shape[1] / 2)

    # we define a fixed crossover rate to be used to know the chance to undergo crossover
    crossover_rate = 0.8
    i = 0

    # a loop for the occurance of the crossover
    while (parents.shape[0] < num_offsprings):

        parent1_index = i % parents.shape[0]
        parent_index = (i + 1) % parents.shape[0]

        # generate a random float number between 0 - 1
        x = rd.random()

        # a condition where we will use the crossover rate earlier. if we define a higher float from 0 - 1 there is a higher chance of the crossover to happen
        if x > crossover_rate:
            continue
        parent1_index = i % parents.shape[0]
        parent2_index = (i + 1) % parent_index[0]
        offsprings[i, 0:crossover_point] = parents[parent1_index, 0:crossover_point]
        offsprings[i, crossover_point:] = parents[parent2_index, crossover_point:]
        i = +1
    return offsprings


#from the crossovered solutions
#we will generate another set of solution by the process of mutating it.
#mutation is replacing the the value of a gene in a solution, in this case a 0 can be change to 1 or vice versa
def mutation(offsprings):
    #initially mutants are the offsprings or result solutions from the crossover process
    mutants = np.empty((offsprings.shape))
    #here we defined a fixed mutation rate same as the crossover rate the higher the number the higher the chance to mutate
    mutation_rate = 0.4
    #a loop for the process of mutating
    for i in range(mutants.shape[0]):
        #generate a random float number from 0 -1
        random_value = rd.random()
        mutants[i, :] = offsprings[i, :]
        #same as the crossover if the random value is higher than the mutation rate it will undergo mutation
        if random_value > mutation_rate:
            continue
        #randomly selects a gene to be mutated or in here the index of an array of the solution
        int_random_value = randint(0, offsprings.shape[1] - 1)
        #a condition that if the gene is equal to 0 it will change to 1 else 1 will change to 0
        if mutants[i, int_random_value] == 0:
            mutants[i, int_random_value] = 1
        else:
            mutants[i, int_random_value] = 0
    return mutants


# calling of functions defined above in its order of definition based from the flow of genetic algorithm.
def optimize(weight, value, population, pop_size, num_generations, weight_limit):
    parameters, fitness_history = [], []
    num_parents = int(pop_size[0] / 2)
    num_offsprings = pop_size[0] - num_parents
    for i in range(num_generations):
        fitness = cal_fitness(weight, value, population, weight_limit)
        fitness_history.append(fitness)
        parents = selection(fitness, num_parents, population)
        offsprings = crossover(parents, num_offsprings)
        mutants = mutation(offsprings)
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = mutants

    #displaying the last generation -> this has the most number of fittest solutions in all the generations generated
    print('Last generation: \n{}\n'.format(population))
    fitness_last_gen = cal_fitness(weight, value, population, weight_limit)
    print('Fitness of the last generation: \n{}\n'.format(fitness_last_gen))
    max_fitness = np.where(fitness_last_gen == np.max(fitness_last_gen))
    parameters.append(population[max_fitness[0][0], :])
    return parameters, fitness_history

#showing the fittest solution with its item number
parameters, fitness_history = optimize(weight, value, initial_population, pop_size, num_generations,
                                       knapsack_weight_limit)
print('The optimized parameters for the given inputs are: \n{}'.format(parameters))
selected_items = item_number * parameters
print('\nSelected items that will maximize the knapsack without breaking it:')
for i in range(selected_items.shape[1]):
    if selected_items[0][i] != 0:
        print('{}\n'.format(selected_items[0][i]))

#showing a graph for the whole orocess of until it gets the best solution to the problem
fitness_history_mean = [np.mean(fitness) for fitness in fitness_history]
fitness_history_max = [np.max(fitness) for fitness in fitness_history]
plt.plot(list(range(num_generations)), fitness_history_mean, label='Mean Fitness')
plt.plot(list(range(num_generations)), fitness_history_max, label='Max Fitness')
plt.legend()
plt.title('Fitness through the generations')
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.show()
print(np.asarray(fitness_history).shape)
