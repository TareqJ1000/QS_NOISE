# -*- coding: utf-8 -*-
"""
This code runs the genetic optimization on the Kraus operator decomposition
"""
import numpy as np
from scipy import integrate
import pygad 
import yaml 
from yaml import Loader


# Load configuration file
stream = open(f"configs/ga0.yaml", 'r')
cnfg = yaml.load(stream, Loader=Loader)

# Load parameters
n_coeffs = cnfg['n_coeffs']

# Parameters for the genetic optimization

# prepare other parameters 
num_generations = cnfg['num_generations']
parents_mating =  cnfg['parents_mating']

sol_per_pop = cnfg['sol_per_pop'] # number of parents in the population?? 

#lower and upper-bound ranges of the parameterization. 
init_range_low = 0
init_range_high = 2*np.pi

parent_selection_type = "tournament"
K_tournament = cnfg['num_contestants'] # number of contestants, essentially
keep_elitism  = cnfg['num_elitists']

#instance name 
instance_name = cnfg['instance_name']

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = cnfg['percentage_mutation'] # probability of mutation 

# Functions used to compute the loss function
 
def compute_fourier_expansions(q,init_E, init_nz, ECos, ESin, nzCos, nzSin):
    en_gen = init_E 
    # This computes the ... phase? inside the cos term of the integrand 
    for i in np.arange(len(ECos)): # Ecos and Esin should be of the same length
        en_gen += ECos[i]*np.cos(i*q) + ESin[i]*np.sin(i*q)

    nz_gen = init_nz
    #Same deal as before
    for i in np.arange(len(ECos)): # Ecos and Esin should be of the same length
        nz_gen += nzCos[i]*np.cos(i*q) + nzSin[i]*np.sin(i*q)
    
    return en_gen, nz_gen

def compute_en(q, init_E, ECos, ESin):
    en_gen = init_E 
    # This computes the ... phase? inside the cos term of the integrand 
    for i in np.arange(len(ECos)): # Ecos and Esin should be of the same length
        en_gen += ECos[i]*np.cos(i*q) + ESin[i]*np.sin(i*q)
        
    return en_gen

def compute_nz(q, init_nz, nzCos, nzSin):
    nz_gen = init_nz
    # This computes the ... phase? inside the cos term of the integrand 
    for i in np.arange(len(nzCos)): # Ecos and Esin should be of the same length
        nz_gen += nzCos[i]*np.cos(i*q) + nzSin[i]*np.sin(i*q)
    return nz_gen 

def first_integrand(q, init_E, ECos, ESin):
    en_gen = compute_en(q, init_E, ECos, ESin)
    return np.cos(en_gen)**2

def second_integrand(q, init_E, init_nz,  ECos, ESin, nzCos, nzSin):
    en_gen = compute_en(q, init_E, ECos, ESin)
    nz_gen = compute_nz(q, init_nz, nzCos, nzSin)
    
    return (nz_gen*np.sin(en_gen))**2

def compute_loss_func(init_E, init_nz, Ecos, Esin, nzCos, nzSin):
    integral_one = integrate.quad(first_integrand, -np.pi, np.pi, args=(init_E, Ecos, Esin))[0]
    integral_two = integrate.quad(second_integrand, -np.pi, np.pi, args=(init_E, init_nz, Ecos, Esin, nzCos, nzSin))[0]
    
    return np.abs(1 - (integral_one + integral_two))

# There are n_coeffs coefficients for Ecos, n_coeffs coefficients for Esin, same for both nzCos and nzSin, and 2 for the initial En and nz values. 
# In all, we have 4*n_coeff + 2 coeffcients we have to consider 

def assign_coeffs(n_coeffs, coeffs):
    
    init_E = coeffs[0]
    init_nz = coeffs[1]
    Ecos = coeffs[1:n_coeffs+1]
    Esin = coeffs[n_coeffs+1:2*n_coeffs+1]
    nzCos = coeffs[2*n_coeffs+1: 3*n_coeffs+1]
    nzSin = coeffs[3*n_coeffs+1:4*n_coeffs+1]
    
    return init_E, init_nz, Ecos, Esin, nzCos, nzSin 


# Made for the PyGAD class
# ga_instance - this is the instance of the pygad.GA class that stores all of our hyperparameters (?)
# solution - these would be the parameters for which we are optimizing for. Change the fitness_batch_size to accept multiple solutions 
# solution_idx - indices of solution in the population. 

def the_fitness_function(ga_instance, solution, solution_idx):
    init_E, init_nz, Ecos, Esin, nzCos, nzSin = assign_coeffs(n_coeffs, solution)
    fitness = compute_loss_func(init_E, init_nz, Ecos, Esin, nzCos, nzSin)
    return -fitness


#This function keeps track of the generation number + best fitness
def on_gen(ga_instance):
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])


fitness_function = the_fitness_function
num_genes = 4*n_coeffs + 2


ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       on_generation=on_gen, 
                        K_tournament=K_tournament, 
                        keep_elitism=keep_elitism)


ga_instance.run()

ga_instance.save(filename=f'genetic_instances/{instance_name}')

# save instance of the genetic algorthm for later analysis

















