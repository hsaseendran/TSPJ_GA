// eax_crossover.h: Header file for GPU-based Edge Assembly Crossover (EAX)

#ifndef EAX_CROSSOVER_H
#define EAX_CROSSOVER_H

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

// Host function to perform EAX crossover for TSPJ
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

#endif // EAX_CROSSOVER_H

// eax_crossover.cu: GPU-based Edge Assembly Crossover implementation

#include "eax_crossover.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/extrema.h>
#include <cub/cub.cuh>

// Device utility functions
__device__ inline uint16_t getEdgeKey(uint16_t from, uint16_t to) {
    return (from < to) ? (from * MAX_CITIES + to) : (to * MAX_CITIES + from);
}

__device__ inline void addEdgeToMatrix(uint8_t* adjacencyMatrix, uint16_t from, uint16_t to, 
                                       uint16_t tourLength, uint8_t parentId) {
    if (from < tourLength && to < tourLength) {
        uint32_t idx = from * tourLength + to;
        atomicOr(&adjacencyMatrix[idx], (1 << parentId));
        if (from != to) { // Symmetric
            uint32_t idx_sym = to * tourLength + from;
            atomicOr(&adjacencyMatrix[idx_sym], (1 << parentId));
        }
    }
}

__device__ inline bool hasEdgeFromParent(const uint8_t* adjacencyMatrix, 
                                         uint16_t from, uint16_t to, 
                                         uint16_t tourLength, uint8_t parentId) {
    if (from >= tourLength || to >= tourLength) return false;
    uint32_t idx = from * tourLength + to;
    return (adjacencyMatrix[idx] & (1 << parentId)) != 0;
}

// Kernel 1: Extract edges from tours in parallel
__global__ void extractEdgesKernel(const size_t* tour1, const size_t* tour2,
                                   Edge* edges1, Edge* edges2, 
                                   uint16_t tourLength, uint32_t numPairs) {
    uint32_t pairIdx = blockIdx.x;
    uint32_t edgeIdx = threadIdx.x;
    
    if (pairIdx < numPairs && edgeIdx < tourLength) {
        uint32_t baseIdx = pairIdx * tourLength;
        
        // Extract edge from tour1
        uint16_t from1 = tour1[baseIdx + edgeIdx];
        uint16_t to1 = tour1[baseIdx + ((edgeIdx + 1) % tourLength)];
        edges1[baseIdx + edgeIdx] = Edge(from1, to1);
        
        // Extract edge from tour2
        uint16_t from2 = tour2[baseIdx + edgeIdx];
        uint16_t to2 = tour2[baseIdx + ((edgeIdx + 1) % tourLength)];
        edges2[baseIdx + edgeIdx] = Edge(from2, to2);
    }
}

// Kernel 2: Build union graph with adjacency matrix
__global__ void buildUnionGraphKernel(const Edge* edges1, const Edge* edges2,
                                      uint8_t* adjacencyMatrix, uint16_t* degrees,
                                      uint16_t tourLength, uint32_t numPairs) {
    uint32_t pairIdx = blockIdx.x;
    uint32_t edgeIdx = threadIdx.x;
    
    if (pairIdx < numPairs && edgeIdx < tourLength) {
        uint32_t baseIdx = pairIdx * tourLength;
        uint32_t matrixOffset = pairIdx * tourLength * tourLength;
        
        // Add edges from both parents to adjacency matrix
        Edge e1 = edges1[baseIdx + edgeIdx];
        Edge e2 = edges2[baseIdx + edgeIdx];
        
        addEdgeToMatrix(adjacencyMatrix + matrixOffset, e1.from, e1.to, tourLength, 0);
        addEdgeToMatrix(adjacencyMatrix + matrixOffset, e2.from, e2.to, tourLength, 1);
        
        // Calculate degree for this edge's endpoints (atomic to handle concurrent updates)
        if (edgeIdx == 0) { // Single thread per pair calculates degrees
            uint32_t degreeOffset = pairIdx * tourLength;
            for (uint16_t city = 0; city < tourLength; ++city) {
                uint16_t degree = 0;
                for (uint16_t neighbor = 0; neighbor < tourLength; ++neighbor) {
                    if (adjacencyMatrix[matrixOffset + city * tourLength + neighbor] != 0) {
                        degree++;
                    }
                }
                degrees[degreeOffset + city] = degree;
            }
        }
    }
}

// Kernel 3: Find alternating cycles using parallel DFS
__global__ void findAlternatingCyclesKernel(const uint8_t* adjacencyMatrix,
                                            const uint16_t* degrees,
                                            Cycle* cycles, uint16_t* numCycles,
                                            uint16_t tourLength, uint32_t numPairs) {
    uint32_t pairIdx = blockIdx.x;
    uint32_t startCity = threadIdx.x;
    
    if (pairIdx >= numPairs || startCity >= tourLength) return;
    
    uint32_t matrixOffset = pairIdx * tourLength * tourLength;
    uint32_t degreeOffset = pairIdx * tourLength;
    uint32_t cycleOffset = pairIdx * MAX_CYCLES;
    
    __shared__ bool visited[MAX_CITIES];
    __shared__ uint16_t currentCycle[MAX_CYCLE_LENGTH];
    __shared__ bool currentParents[MAX_CYCLE_LENGTH];
    __shared__ uint16_t sharedNumCycles;
    
    // Initialize shared memory
    if (threadIdx.x == 0) {
        sharedNumCycles = 0;
        for (uint16_t i = 0; i < tourLength; i++) {
            visited[i] = false;
        }
    }
    __syncthreads();
    
    // Each thread tries to find a cycle starting from a different city
    if (!visited[startCity] && degrees[degreeOffset + startCity] > 0) {
        uint16_t cycleLength = 0;
        uint16_t currentCity = startCity;
        uint8_t lastParentUsed = 2; // 0=parent1, 1=parent2, 2=none
        
        do {
            if (cycleLength >= MAX_CYCLE_LENGTH) break;
            
            visited[currentCity] = true;
            currentCycle[cycleLength] = currentCity;
            
            // Find next city using alternating parent
            uint16_t nextCity = tourLength; // Invalid
            uint8_t nextParent = (lastParentUsed == 0) ? 1 : 0;
            
            for (uint16_t candidate = 0; candidate < tourLength; candidate++) {
                if (hasEdgeFromParent(adjacencyMatrix + matrixOffset, currentCity, candidate, 
                                      tourLength, nextParent) && 
                    (candidate == startCity || !visited[candidate])) {
                    nextCity = candidate;
                    break;
                }
            }
            
            if (nextCity >= tourLength) break; // No valid next city
            
            currentParents[cycleLength] = (nextParent == 0);
            cycleLength++;
            currentCity = nextCity;
            lastParentUsed = nextParent;
            
        } while (currentCity != startCity && cycleLength < MAX_CYCLE_LENGTH);
        
        // Store cycle if valid and complete
        if (currentCity == startCity && cycleLength >= 3) {
            uint16_t cycleIdx = atomicAdd(&sharedNumCycles, 1);
            if (cycleIdx < MAX_CYCLES) {
                Cycle& cycle = cycles[cycleOffset + cycleIdx];
                cycle.length = cycleLength;
                for (uint16_t i = 0; i < cycleLength; i++) {
                    cycle.cities[i] = currentCycle[i];
                    cycle.usesParent1[i] = currentParents[i];
                }
            }
        }
    }
    
    __syncthreads();
    if (threadIdx.x == 0) {
        numCycles[pairIdx] = sharedNumCycles;
    }
}

// Kernel 4: Evaluate all possible assemblies and find best
__global__ void evaluateAssembliesKernel(const Cycle* cycles, const uint16_t* numCycles,
                                         const float* costMatrix, float* assemblyCosts,
                                         uint32_t* bestAssemblies, uint16_t tourLength,
                                         uint32_t numPairs) {
    uint32_t pairIdx = blockIdx.x;
    uint32_t assemblyIdx = threadIdx.x;
    
    if (pairIdx >= numPairs) return;
    
    uint16_t cycleCount = numCycles[pairIdx];
    uint32_t maxAssemblies = (1 << cycleCount); // 2^cycleCount
    
    __shared__ float sharedCosts[256]; // Max threads per block
    __shared__ uint32_t sharedAssemblies[256];
    __shared__ uint32_t bestIdx;
    __shared__ float bestCost;
    
    if (threadIdx.x == 0) {
        bestCost = INFINITY;
        bestIdx = 0;
    }
    __syncthreads();
    
    float cost = INFINITY;
    if (assemblyIdx < maxAssemblies) {
        // Evaluate this assembly
        cost = 0.0f;
        bool validTour = true;
        
        // Calculate cost based on which cycles are included
        uint32_t cycleOffset = pairIdx * MAX_CYCLES;
        for (uint16_t cycleIdx = 0; cycleIdx < cycleCount && validTour; cycleIdx++) {
            bool includeCycle = (assemblyIdx >> cycleIdx) & 1;
            const Cycle& cycle = cycles[cycleOffset + cycleIdx];
            
            for (uint16_t i = 0; i < cycle.length && validTour; i++) {
                uint16_t from = cycle.cities[i];
                uint16_t to = cycle.cities[(i + 1) % cycle.length];
                
                // Use appropriate cost based on inclusion and parent selection
                bool useThisEdge = (includeCycle && !cycle.usesParent1[i]) || 
                                   (!includeCycle && cycle.usesParent1[i]);
                
                if (useThisEdge && from < tourLength && to < tourLength) {
                    cost += costMatrix[from * tourLength + to];
                } else {
                    validTour = false;
                }
            }
        }
        
        if (!validTour) cost = INFINITY;
    }
    
    sharedCosts[threadIdx.x] = cost;
    sharedAssemblies[threadIdx.x] = assemblyIdx;
    __syncthreads();
    
    // Parallel reduction to find minimum cost
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && threadIdx.x + stride < blockDim.x) {
            if (sharedCosts[threadIdx.x + stride] < sharedCosts[threadIdx.x]) {
                sharedCosts[threadIdx.x] = sharedCosts[threadIdx.x + stride];
                sharedAssemblies[threadIdx.x] = sharedAssemblies[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        assemblyCosts[pairIdx] = sharedCosts[0];
        bestAssemblies[pairIdx] = sharedAssemblies[0];
    }
}

// Kernel 5: Construct offspring tours from best assemblies
__global__ void constructOffspringKernel(const Cycle* cycles, const uint16_t* numCycles,
                                         const uint32_t* bestAssemblies,
                                         const size_t* parent1Tours, const size_t* parent2Tours,
                                         size_t* childTours, uint16_t tourLength,
                                         uint32_t numPairs) {
    uint32_t pairIdx = blockIdx.x;
    uint32_t cityIdx = threadIdx.x;
    
    if (pairIdx >= numPairs || cityIdx >= tourLength) return;
    
    uint32_t assembly = bestAssemblies[pairIdx];
    uint16_t cycleCount = numCycles[pairIdx];
    uint32_t parentOffset = pairIdx * tourLength;
    uint32_t childOffset = pairIdx * tourLength;
    uint32_t cycleOffset = pairIdx * MAX_CYCLES;
    
    __shared__ size_t sharedChild[MAX_CITIES];
    __shared__ bool processed[MAX_CITIES];
    
    // Initialize
    if (threadIdx.x == 0) {
        for (uint16_t i = 0; i < tourLength; i++) {
            processed[i] = false;
            sharedChild[i] = parent1Tours[parentOffset + i]; // Default to parent1
        }
    }
    __syncthreads();
    
    // Apply selected cycles
    if (cityIdx == 0) { // Single thread processes cycles
        for (uint16_t cycleIdx = 0; cycleIdx < cycleCount; cycleIdx++) {
            bool includeCycle = (assembly >> cycleIdx) & 1;
            if (includeCycle) {
                const Cycle& cycle = cycles[cycleOffset + cycleIdx];
                // Apply this cycle's edges from parent2
                for (uint16_t i = 0; i < cycle.length; i++) {
                    if (!cycle.usesParent1[i]) {
                        uint16_t pos = cycle.cities[i];
                        if (pos < tourLength) {
                            sharedChild[pos] = parent2Tours[parentOffset + pos];
                        }
                    }
                }
            }
        }
    }
    __syncthreads();
    
    // Copy result to global memory
    if (cityIdx < tourLength) {
        childTours[childOffset + cityIdx] = sharedChild[cityIdx];
    }
}

// Host function for single crossover (interface compatibility)
Genome performCrossover(const Genome& parent1, const Genome& parent2, int mode) {
    return performEAXCrossover(parent1, parent2, mode);
}

// Optimized batch processing version
std::vector<Genome> performBatchEAXCrossover(const std::vector<Genome>& parents1,
                                            const std::vector<Genome>& parents2, 
                                            int mode) {
    if (parents1.size() != parents2.size() || parents1.empty()) {
        return {};
    }
    
    uint32_t numPairs = parents1.size();
    uint16_t tourLength = parents1[0].citySequence.size();
    
    // Flatten all parent data for batch processing
    thrust::device_vector<size_t> d_allParents1City(numPairs * tourLength);
    thrust::device_vector<size_t> d_allParents2City(numPairs * tourLength);
    thrust::device_vector<size_t> d_allChildrenCity(numPairs * tourLength);
    
    // Copy data to device
    for (uint32_t i = 0; i < numPairs; i++) {
        thrust::copy(parents1[i].citySequence.begin(), parents1[i].citySequence.end(),
                    d_allParents1City.begin() + i * tourLength);
        thrust::copy(parents2[i].citySequence.begin(), parents2[i].citySequence.end(),
                    d_allParents2City.begin() + i * tourLength);
    }
    
    // Allocate batch processing memory
    thrust::device_vector<Edge> d_edges1(numPairs * tourLength);
    thrust::device_vector<Edge> d_edges2(numPairs * tourLength);
    thrust::device_vector<uint8_t> d_adjacencyMatrix(numPairs * tourLength * tourLength);
    thrust::device_vector<uint16_t> d_degrees(numPairs * tourLength);
    thrust::device_vector<Cycle> d_cycles(numPairs * MAX_CYCLES);
    thrust::device_vector<uint16_t> d_numCycles(numPairs);
    thrust::device_vector<float> d_assemblyCosts(numPairs);
    thrust::device_vector<uint32_t> d_bestAssemblies(numPairs);
    thrust::device_vector<float> d_costMatrix(tourLength * tourLength);
    
    // Initialize cost matrix (placeholder - should use actual travel costs)
    thrust::fill(d_costMatrix.begin(), d_costMatrix.end(), 1.0f);
    
    // Launch batch kernels
    dim3 gridDim(numPairs);
    dim3 blockDim(tourLength);
    
    extractEdgesKernel<<<gridDim, blockDim>>>(
        thrust::raw_pointer_cast(d_allParents1City.data()),
        thrust::raw_pointer_cast(d_allParents2City.data()),
        thrust::raw_pointer_cast(d_edges1.data()),
        thrust::raw_pointer_cast(d_edges2.data()),
        tourLength, numPairs);
    
    buildUnionGraphKernel<<<gridDim, blockDim>>>(
        thrust::raw_pointer_cast(d_edges1.data()),
        thrust::raw_pointer_cast(d_edges2.data()),
        thrust::raw_pointer_cast(d_adjacencyMatrix.data()),
        thrust::raw_pointer_cast(d_degrees.data()),
        tourLength, numPairs);
    
    findAlternatingCyclesKernel<<<gridDim, blockDim>>>(
        thrust::raw_pointer_cast(d_adjacencyMatrix.data()),
        thrust::raw_pointer_cast(d_degrees.data()),
        thrust::raw_pointer_cast(d_cycles.data()),
        thrust::raw_pointer_cast(d_numCycles.data()),
        tourLength, numPairs);
    
    dim3 evalBlockDim(min(256, 1 << min(16, (int)tourLength)));
    evaluateAssembliesKernel<<<gridDim, evalBlockDim>>>(
        thrust::raw_pointer_cast(d_cycles.data()),
        thrust::raw_pointer_cast(d_numCycles.data()),
        thrust::raw_pointer_cast(d_costMatrix.data()),
        thrust::raw_pointer_cast(d_assemblyCosts.data()),
        thrust::raw_pointer_cast(d_bestAssemblies.data()),
        tourLength, numPairs);
    
    constructOffspringKernel<<<gridDim, blockDim>>>(
        thrust::raw_pointer_cast(d_cycles.data()),
        thrust::raw_pointer_cast(d_numCycles.data()),
        thrust::raw_pointer_cast(d_bestAssemblies.data()),
        thrust::raw_pointer_cast(d_allParents1City.data()),
        thrust::raw_pointer_cast(d_allParents2City.data()),
        thrust::raw_pointer_cast(d_allChildrenCity.data()),
        tourLength, numPairs);
    
    // Process job sequences with same assembly decisions
    thrust::device_vector<size_t> d_allParents1Job(numPairs * tourLength);
    thrust::device_vector<size_t> d_allParents2Job(numPairs * tourLength);
    thrust::device_vector<size_
    uint16_t tourLength = parent1.citySequence.size();
    uint32_t numPairs = 1; // Single pair for now
    
    // Device memory allocation
    thrust::device_vector<size_t> d_parent1City(parent1.citySequence);
    thrust::device_vector<size_t> d_parent2City(parent2.citySequence);
    thrust::device_vector<size_t> d_childCity(tourLength);
    
    thrust::device_vector<Edge> d_edges1(tourLength);
    thrust::device_vector<Edge> d_edges2(tourLength);
    thrust::device_vector<uint8_t> d_adjacencyMatrix(tourLength * tourLength);
    thrust::device_vector<uint16_t> d_degrees(tourLength);
    thrust::device_vector<Cycle> d_cycles(MAX_CYCLES);
    thrust::device_vector<uint16_t> d_numCycles(1);
    thrust::device_vector<float> d_assemblyCosts(1);
    thrust::device_vector<uint32_t> d_bestAssemblies(1);
    
    // Create cost matrix (simplified - use Euclidean distance)
    thrust::device_vector<float> d_costMatrix(tourLength * tourLength);
    thrust::fill(d_costMatrix.begin(), d_costMatrix.end(), 1.0f); // Placeholder
    
    // Launch kernels in sequence
    dim3 gridDim(numPairs);
    dim3 blockDim(tourLength);
    
    // Step 1: Extract edges
    extractEdgesKernel<<<gridDim, blockDim>>>(
        thrust::raw_pointer_cast(d_parent1City.data()),
        thrust::raw_pointer_cast(d_parent2City.data()),
        thrust::raw_pointer_cast(d_edges1.data()),
        thrust::raw_pointer_cast(d_edges2.data()),
        tourLength, numPairs);
    
    // Step 2: Build union graph
    buildUnionGraphKernel<<<gridDim, blockDim>>>(
        thrust::raw_pointer_cast(d_edges1.data()),
        thrust::raw_pointer_cast(d_edges2.data()),
        thrust::raw_pointer_cast(d_adjacencyMatrix.data()),
        thrust::raw_pointer_cast(d_degrees.data()),
        tourLength, numPairs);
    
    // Step 3: Find alternating cycles
    findAlternatingCyclesKernel<<<gridDim, blockDim>>>(
        thrust::raw_pointer_cast(d_adjacencyMatrix.data()),
        thrust::raw_pointer_cast(d_degrees.data()),
        thrust::raw_pointer_cast(d_cycles.data()),
        thrust::raw_pointer_cast(d_numCycles.data()),
        tourLength, numPairs);
    
    // Step 4: Evaluate assemblies
    dim3 evalBlockDim(min(256, 1 << min(16, (int)tourLength))); // Max 2^16 assemblies
    evaluateAssembliesKernel<<<gridDim, evalBlockDim>>>(
        thrust::raw_pointer_cast(d_cycles.data()),
        thrust::raw_pointer_cast(d_numCycles.data()),
        thrust::raw_pointer_cast(d_costMatrix.data()),
        thrust::raw_pointer_cast(d_assemblyCosts.data()),
        thrust::raw_pointer_cast(d_bestAssemblies.data()),
        tourLength, numPairs);
    
    // Step 5: Construct offspring
    constructOffspringKernel<<<gridDim, blockDim>>>(
        thrust::raw_pointer_cast(d_cycles.data()),
        thrust::raw_pointer_cast(d_numCycles.data()),
        thrust::raw_pointer_cast(d_bestAssemblies.data()),
        thrust::raw_pointer_cast(d_parent1City.data()),
        thrust::raw_pointer_cast(d_parent2City.data()),
        thrust::raw_pointer_cast(d_childCity.data()),
        tourLength, numPairs);
    
    // Create result genome
    Genome child(tourLength, tourLength, mode);
    thrust::copy(d_childCity.begin(), d_childCity.end(), child.citySequence.begin());
    
    // Apply same process to job sequence
    thrust::device_vector<size_t> d_parent1Job(parent1.jobSequence);
    thrust::device_vector<size_t> d_parent2Job(parent2.jobSequence);
    thrust::device_vector<size_t> d_childJob(tourLength);
    
    // Reuse the same assembly decision for job sequence
    constructOffspringKernel<<<gridDim, blockDim>>>(
        thrust::raw_pointer_cast(d_cycles.data()),
        thrust::raw_pointer_cast(d_numCycles.data()),
        thrust::raw_pointer_cast(d_bestAssemblies.data()),
        thrust::raw_pointer_cast(d_parent1Job.data()),
        thrust::raw_pointer_cast(d_parent2Job.data()),
        thrust::raw_pointer_cast(d_childJob.data()),
        tourLength, numPairs);
    
    thrust::copy(d_childJob.begin(), d_childJob.end(), child.jobSequence.begin());
    
    // Handle pickup sequence if mode == 1
    if (mode == 1) {
        thrust::device_vector<size_t> d_parent1Pickup(parent1.pickupSequence);
        thrust::device_vector<size_t> d_parent2Pickup(parent2.pickupSequence);
        thrust::device_vector<size_t> d_childPickup(tourLength);
        
        constructOffspringKernel<<<gridDim, blockDim>>>(
            thrust::raw_pointer_cast(d_cycles.data()),
            thrust::raw_pointer_cast(d_numCycles.data()),
            thrust::raw_pointer_cast(d_bestAssemblies.data()),
            thrust::raw_pointer_cast(d_parent1Pickup.data()),
            thrust::raw_pointer_cast(d_parent2Pickup.data()),
            thrust::raw_pointer_cast(d_childPickup.data()),
            tourLength, numPairs);
        
        thrust::copy(d_childPickup.begin(), d_childPickup.end(), child.pickupSequence.begin());
    }
    
    return child;
}