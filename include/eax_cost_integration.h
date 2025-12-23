// eax_cost_integration.h: Integration of cost matrix with EAX crossover

#ifndef EAX_COST_INTEGRATION_H
#define EAX_COST_INTEGRATION_H

#include "genome.h"
#include "crossover.h"  // Include crossover.h to get Cycle definition
#include <vector>

// Initialize EAX with proper cost matrices
void initializeEAXCostMatrices(const std::vector<std::vector<float>>& travelTimes,
                               const std::vector<std::vector<float>>& jobTimes);

// Enhanced EAX crossover with cost-aware evaluation
Genome performCostAwareEAXCrossover(const Genome& parent1, const Genome& parent2, 
                                    int mode);

// Batch version with cost awareness
std::vector<Genome> performBatchCostAwareEAXCrossover(const std::vector<Genome>& parents1,
                                                      const std::vector<Genome>& parents2, 
                                                      int mode);

// Cleanup function
void cleanupEAXCostMatrices();

// GPU kernels for cost-aware evaluation
__global__ void calculateTourCostKernel(const size_t* tour, const float* travelCosts, 
                                        size_t tourLength, float* totalCost);

__global__ void evaluateAssembliesWithCostsKernel(const Cycle* cycles, const uint16_t* numCycles,
                                                   const float* travelCosts, const float* jobCosts,
                                                   float* assemblyCosts, uint32_t* bestAssemblies,
                                                   uint16_t tourLength, uint32_t numPairs);

#endif // EAX_COST_INTEGRATION_H