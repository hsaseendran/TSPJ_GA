// mutation.h: Header file for GPU-optimized 2-opt and swap mutations in TSPJ

#ifndef MUTATION_H
#define MUTATION_H

#include "genome.h"
#include <vector>

/**
 * Initialize the global cost matrix for 2-opt mutations
 * Should be called once at the start of the GA
 * 
 * @param travelTimes The travel time/cost matrix
 */
void initializeMutationCostMatrix(const std::vector<std::vector<float>>& travelTimes);

/**
 * Cleanup the global cost matrix
 * Should be called at the end of the GA
 */
void cleanupMutationCostMatrix();

/**
 * Perform mutation on a single genome
 * - Uses GPU-optimized 2-opt mutation for TSP sequences (city and pickup)
 * - Uses swap mutation for job sequence
 * 
 * @param genome The genome to mutate
 * @param mutationRate Probability of applying mutation
 * @param mode 0=no pickup, 1=with pickup sequence
 * @param stagnationCount Number of generations without improvement (for escape probability)
 */
void performMutation(Genome& genome, float mutationRate, int mode, size_t stagnationCount = 0);

/**
 * Perform batch mutation on multiple genomes (optimized version)
 * Processes multiple genomes in parallel on GPU
 * 
 * @param genomes Vector of genomes to mutate
 * @param mutationRate Probability of applying mutation
 * @param mode 0=no pickup, 1=with pickup sequence
 * @param travelTimes Travel time matrix for cost-aware 2-opt
 * @param stagnationCount Number of generations without improvement
 */
void performBatchMutation(std::vector<Genome>& genomes, float mutationRate, int mode,
                         const std::vector<std::vector<float>>& travelTimes,
                         size_t stagnationCount = 0);

#endif // MUTATION_H