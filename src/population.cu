// population.cu: Manage population for TSPJ genetic algorithm

#include "population.h"
#include "genome.h"
#include "fitness_evaluator.h"
#include <vector>
#include <algorithm>
#include <iostream>
#include <random>

// Initialize a population with random genomes
void initializePopulation(std::vector<Genome>& population, size_t populationSize,
                          size_t numCities, size_t numJobs, int mode) {
    population.clear();
    population.reserve(populationSize);

    for (size_t i = 0; i < populationSize; ++i) {
        population.emplace_back(numCities, numJobs, mode);
    }
}

// Evaluate fitness of the entire population using fitness_evaluator
void evaluatePopulation(std::vector<Genome>& population,
                        const std::vector<std::vector<float>>& travelTimes,
                        const std::vector<std::vector<float>>& jobTimes, int mode) {
    evaluatePopulationFitness(population, travelTimes, jobTimes, mode); // Pass mode to fitness evaluator
}

// Sort population by fitness (ascending order)
void sortPopulationByFitness(std::vector<Genome>& population) {
    std::sort(population.begin(), population.end(), [](const Genome& a, const Genome& b) {
        return a.fitness < b.fitness;
    });
}

// Get the best genome in the population
Genome getBestGenome(const std::vector<Genome>& population) {
    return *std::min_element(population.begin(), population.end(), [](const Genome& a, const Genome& b) {
        return a.fitness < b.fitness;
    });
}

// Replace the worst genomes with offspring
void replaceWorst(std::vector<Genome>& population, const std::vector<Genome>& offspring) {
    size_t numOffspring = offspring.size();
    size_t populationSize = population.size();

    // Replace the worst genomes
    for (size_t i = 0; i < numOffspring; ++i) {
        population[populationSize - numOffspring + i] = offspring[i];
    }

    // Re-sort population
    sortPopulationByFitness(population);
}

// Print the population for debugging
void printPopulation(const std::vector<Genome>& population, int mode) {
    std::cout << "Population:\n";
    for (size_t i = 0; i < population.size(); ++i) {
        std::cout << "Genome " << i << ":\n";
        population[i].print(mode); // Pass mode to Genome's print function
    }
}
