// fitness_evaluator.h: Header file for fitness evaluation in TSPJ using flattened arrays

#ifndef FITNESS_EVALUATOR_H
#define FITNESS_EVALUATOR_H

#include "genome.h"
#include <vector>

// Flatten the population into a single array for GPU processing
void flattenPopulation(const std::vector<Genome>& population, std::vector<size_t>& flatArray,
                       std::vector<float>& fitnessArray, int mode);

// Unflatten the fitness results back into the population
void unflattenFitness(const std::vector<float>& fitnessArray, std::vector<Genome>& population);

// Evaluate fitness for the entire population
void evaluatePopulationFitness(std::vector<Genome>& population,
                                const std::vector<std::vector<float>>& travelTimes,
                                const std::vector<std::vector<float>>& jobTimes, int mode);

#endif // FITNESS_EVALUATOR_H
