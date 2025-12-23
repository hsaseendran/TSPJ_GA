// crossover.h: Header file for GPU-based Edge Assembly Crossover (EAX) in TSPJ

#ifndef CROSSOVER_H
#define CROSSOVER_H

#include "genome.h"
#include <vector>

// Maximum problem size constraints for GPU memory allocation
#define MAX_CITIES 256
#define MAX_CYCLES 64
#define MAX_CYCLE_LENGTH 64

// Edge structure for EAX
struct Edge {
    uint16_t from, to;
    __device__ __host__ Edge() : from(0), to(0) {}
    __device__ __host__ Edge(uint16_t f, uint16_t t) : from(f), to(t) {}
    __device__ __host__ bool operator==(const Edge& other) const {
        return (from == other.from && to == other.to) || 
               (from == other.to && to == other.from);
    }
};

// Cycle structure for EAX
struct Cycle {
    uint16_t length;
    uint16_t cities[MAX_CYCLE_LENGTH];
    bool usesParent1[MAX_CYCLE_LENGTH]; // Which parent each edge comes from
    
    __device__ __host__ Cycle() : length(0) {}
};

// Host function to perform EAX crossover for TSPJ (returns single high-quality offspring)
Genome performCrossover(const Genome& parent1, const Genome& parent2, int mode);

// Batch processing version for multiple parent pairs
std::vector<Genome> performBatchEAXCrossover(const std::vector<Genome>& parents1,
                                            const std::vector<Genome>& parents2, 
                                            int mode);

// Internal EAX implementation
Genome performEAXCrossover(const Genome& parent1, const Genome& parent2, int mode);

// GPU kernels (internal use)
__global__ void extractEdgesKernel(const size_t* tour1, const size_t* tour2,
                                   Edge* edges1, Edge* edges2, 
                                   uint16_t tourLength, uint32_t numPairs);

__global__ void buildUnionGraphKernel(const Edge* edges1, const Edge* edges2,
                                      uint8_t* adjacencyMatrix, uint16_t* degrees,
                                      uint16_t tourLength, uint32_t numPairs);

__global__ void findAlternatingCyclesKernel(const uint8_t* adjacencyMatrix,
                                            const uint16_t* degrees,
                                            Cycle* cycles, uint16_t* numCycles,
                                            uint16_t tourLength, uint32_t numPairs);

__global__ void evaluateAssembliesKernel(const Cycle* cycles, const uint16_t* numCycles,
                                         const float* costMatrix, float* assemblyCosts,
                                         uint32_t* bestAssemblies, uint16_t tourLength,
                                         uint32_t numPairs);

__global__ void constructOffspringKernel(const Cycle* cycles, const uint16_t* numCycles,
                                         const uint32_t* bestAssemblies,
                                         const size_t* parent1Tours, const size_t* parent2Tours,
                                         size_t* childTours, uint16_t tourLength,
                                         uint32_t numPairs);

#endif // CROSSOVER_H