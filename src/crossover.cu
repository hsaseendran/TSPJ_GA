// crossover.cu: GPU-based Edge Assembly Crossover implementation (Fixed - No Atomics)

#include "crossover.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/extrema.h>
#include <algorithm>
#include <iostream>
#include <cstdio>

// Device utility functions
__device__ inline uint16_t getEdgeKey(uint16_t from, uint16_t to) {
    return (from < to) ? (from * MAX_CITIES + to) : (to * MAX_CITIES + from);
}

__device__ inline void addEdgeToMatrix(uint8_t* adjacencyMatrix, uint16_t from, uint16_t to, 
                                       uint16_t tourLength, uint8_t parentId) {
    if (from < tourLength && to < tourLength) {
        uint32_t idx = from * tourLength + to;
        // Use bitwise OR instead of atomic - safe for setting bits
        adjacencyMatrix[idx] |= (1 << parentId);
        if (from != to) { // Symmetric
            uint32_t idx_sym = to * tourLength + from;
            adjacencyMatrix[idx_sym] |= (1 << parentId);
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
        uint16_t from1 = (uint16_t)tour1[baseIdx + edgeIdx];
        uint16_t to1 = (uint16_t)tour1[baseIdx + ((edgeIdx + 1) % tourLength)];
        edges1[baseIdx + edgeIdx] = Edge(from1, to1);
        
        // Extract edge from tour2
        uint16_t from2 = (uint16_t)tour2[baseIdx + edgeIdx];
        uint16_t to2 = (uint16_t)tour2[baseIdx + ((edgeIdx + 1) % tourLength)];
        edges2[baseIdx + edgeIdx] = Edge(from2, to2);
    }
}

// Kernel 2: Build union graph with adjacency matrix
__global__ void buildUnionGraphKernel(const Edge* edges1, const Edge* edges2,
                                      uint8_t* adjacencyMatrix, uint16_t* degrees,
                                      uint16_t tourLength, uint32_t numPairs) {
    uint32_t pairIdx = blockIdx.x;
    uint32_t edgeIdx = threadIdx.x;
    
    if (pairIdx < numPairs) {
        uint32_t matrixOffset = pairIdx * tourLength * tourLength;
        
        // Single thread per block handles all edges for this pair
        if (edgeIdx == 0) {
            // Process all edges sequentially within this thread
            for (uint32_t i = 0; i < tourLength; i++) {
                uint32_t baseIdx = pairIdx * tourLength;
                Edge e1 = edges1[baseIdx + i];
                Edge e2 = edges2[baseIdx + i];
                
                addEdgeToMatrix(adjacencyMatrix + matrixOffset, e1.from, e1.to, tourLength, 0);
                addEdgeToMatrix(adjacencyMatrix + matrixOffset, e2.from, e2.to, tourLength, 1);
            }
            
            // Calculate degrees
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

// Kernel 3: Find alternating cycles using parallel DFS (No atomics)
__global__ void findAlternatingCyclesKernel(const uint8_t* adjacencyMatrix,
                                            const uint16_t* degrees,
                                            Cycle* cycles, uint16_t* numCycles,
                                            uint16_t tourLength, uint32_t numPairs) {
    uint32_t pairIdx = blockIdx.x;
    uint32_t threadId = threadIdx.x;
    
    if (pairIdx >= numPairs) return;
    
    uint32_t matrixOffset = pairIdx * tourLength * tourLength;
    uint32_t degreeOffset = pairIdx * tourLength;
    uint32_t cycleOffset = pairIdx * MAX_CYCLES;
    
    __shared__ bool visited[MAX_CITIES];
    __shared__ uint16_t foundCycles[MAX_CYCLES][MAX_CYCLE_LENGTH];
    __shared__ bool cycleParents[MAX_CYCLES][MAX_CYCLE_LENGTH];
    __shared__ uint16_t cycleLengths[MAX_CYCLES];
    __shared__ uint16_t cycleCount;
    
    // Initialize shared memory
    if (threadId == 0) {
        cycleCount = 0;
        for (uint16_t i = 0; i < min(tourLength, (uint16_t)MAX_CITIES); i++) {
            visited[i] = false;
        }
        for (uint16_t i = 0; i < MAX_CYCLES; i++) {
            cycleLengths[i] = 0;
        }
    }
    __syncthreads();
    
    // Each thread tries to find cycles starting from different cities
    uint16_t startCity = threadId;
    while (startCity < tourLength && cycleCount < MAX_CYCLES) {
        if (!visited[startCity] && degrees[degreeOffset + startCity] > 0) {
            uint16_t currentCycle[MAX_CYCLE_LENGTH];
            bool currentParents[MAX_CYCLE_LENGTH];
            uint16_t cycleLength = 0;
            uint16_t currentCity = startCity;
            uint8_t lastParentUsed = 2; // 0=parent1, 1=parent2, 2=none
            
            do {
                if (cycleLength >= MAX_CYCLE_LENGTH) break;
                
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
                // Use thread ID to determine cycle slot (avoid atomics)
                uint16_t cycleSlot = threadId % MAX_CYCLES;
                if (cycleLengths[cycleSlot] == 0) { // Slot is empty
                    cycleLengths[cycleSlot] = cycleLength;
                    for (uint16_t i = 0; i < cycleLength; i++) {
                        foundCycles[cycleSlot][i] = currentCycle[i];
                        cycleParents[cycleSlot][i] = currentParents[i];
                    }
                    if (cycleSlot >= cycleCount) {
                        cycleCount = cycleSlot + 1;
                    }
                    
                    // Mark cities as visited
                    for (uint16_t i = 0; i < cycleLength; i++) {
                        visited[currentCycle[i]] = true;
                    }
                }
            }
        }
        
        startCity += blockDim.x; // Move to next potential start city
        __syncthreads();
    }
    
    // Copy cycles to global memory
    if (threadId == 0) {
        uint16_t actualCycles = 0;
        for (uint16_t i = 0; i < cycleCount && i < MAX_CYCLES; i++) {
            if (cycleLengths[i] > 0) {
                Cycle& cycle = cycles[cycleOffset + actualCycles];
                cycle.length = cycleLengths[i];
                for (uint16_t j = 0; j < cycleLengths[i]; j++) {
                    cycle.cities[j] = foundCycles[i][j];
                    cycle.usesParent1[j] = cycleParents[i][j];
                }
                actualCycles++;
            }
        }
        numCycles[pairIdx] = actualCycles;
    }
}

// Kernel 4: Evaluate all possible assemblies and find best (No atomics)
__global__ void evaluateAssembliesKernel(const Cycle* cycles, const uint16_t* numCycles,
                                         const float* costMatrix, float* assemblyCosts,
                                         uint32_t* bestAssemblies, uint16_t tourLength,
                                         uint32_t numPairs) {
    uint32_t pairIdx = blockIdx.x;
    uint32_t assemblyIdx = threadIdx.x;
    
    if (pairIdx >= numPairs) return;
    
    uint16_t cycleCount = numCycles[pairIdx];
    uint32_t maxAssemblies = min(256U, 1U << min(cycleCount, (uint16_t)16));
    
    __shared__ float sharedCosts[256]; 
    __shared__ uint32_t sharedAssemblies[256];
    
    // Initialize shared memory
    float cost = INFINITY;
    if (assemblyIdx < maxAssemblies) {
        // Evaluate this assembly
        cost = 0.0f;
        uint32_t cycleOffset = pairIdx * MAX_CYCLES;
        bool validAssembly = true;
        
        for (uint16_t cycleIdx = 0; cycleIdx < cycleCount && validAssembly; cycleIdx++) {
            bool includeCycle = (assemblyIdx >> cycleIdx) & 1;
            const Cycle& cycle = cycles[cycleOffset + cycleIdx];
            
            if (includeCycle) {
                for (uint16_t i = 0; i < cycle.length; i++) {
                    uint16_t from = cycle.cities[i];
                    uint16_t to = cycle.cities[(i + 1) % cycle.length];
                    if (from < tourLength && to < tourLength) {
                        cost += costMatrix[from * tourLength + to];
                    } else {
                        validAssembly = false;
                        break;
                    }
                }
            } else {
                cost += 1.0f; // Small penalty for not using cycle
            }
        }
        
        cost += assemblyIdx * 0.001f; // Tie breaking
        
        if (!validAssembly) cost = INFINITY;
    }
    
    sharedCosts[threadIdx.x] = cost;
    sharedAssemblies[threadIdx.x] = assemblyIdx;
    __syncthreads();
    
    // Parallel reduction to find minimum cost (no atomics)
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
    
    if (pairIdx >= numPairs) return;
    
    uint32_t assembly = bestAssemblies[pairIdx];
    uint16_t cycleCount = numCycles[pairIdx];
    uint32_t parentOffset = pairIdx * tourLength;
    uint32_t childOffset = pairIdx * tourLength;
    uint32_t cycleOffset = pairIdx * MAX_CYCLES;
    
    __shared__ size_t sharedChild[MAX_CITIES];
    __shared__ bool useParent2[MAX_CITIES];
    
    // Initialize shared memory
    if (threadIdx.x == 0) {
        for (uint16_t i = 0; i < min(tourLength, (uint16_t)MAX_CITIES); i++) {
            sharedChild[i] = parent1Tours[parentOffset + i]; // Default to parent1
            useParent2[i] = false;
        }
        
        // Determine which positions should use parent2 based on assembly
        for (uint16_t cycleIdx = 0; cycleIdx < cycleCount; cycleIdx++) {
            bool includeCycle = (assembly >> cycleIdx) & 1;
            if (includeCycle) {
                const Cycle& cycle = cycles[cycleOffset + cycleIdx];
                for (uint16_t i = 0; i < cycle.length; i++) {
                    if (!cycle.usesParent1[i] && cycle.cities[i] < tourLength) {
                        useParent2[cycle.cities[i]] = true;
                    }
                }
            }
        }
        
        // Apply parent2 values where needed
        for (uint16_t i = 0; i < tourLength; i++) {
            if (useParent2[i]) {
                sharedChild[i] = parent2Tours[parentOffset + i];
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

// Host function implementation for single pair
Genome performEAXCrossover(const Genome& parent1, const Genome& parent2, int mode) {
    uint16_t tourLength = static_cast<uint16_t>(parent1.citySequence.size());
    uint32_t numPairs = 1;
    
    // Device memory allocation
    thrust::device_vector<size_t> d_parent1City(parent1.citySequence);
    thrust::device_vector<size_t> d_parent2City(parent2.citySequence);
    thrust::device_vector<size_t> d_childCity(tourLength);
    
    thrust::device_vector<Edge> d_edges1(tourLength);
    thrust::device_vector<Edge> d_edges2(tourLength);
    thrust::device_vector<uint8_t> d_adjacencyMatrix(tourLength * tourLength, 0);
    thrust::device_vector<uint16_t> d_degrees(tourLength, 0);
    thrust::device_vector<Cycle> d_cycles(MAX_CYCLES);
    thrust::device_vector<uint16_t> d_numCycles(1, 0);
    thrust::device_vector<float> d_assemblyCosts(1);
    thrust::device_vector<uint32_t> d_bestAssemblies(1);
    
    // Create simple cost matrix
    thrust::device_vector<float> d_costMatrix(tourLength * tourLength);
    thrust::fill(d_costMatrix.begin(), d_costMatrix.end(), 1.0f);
    
    // Launch kernels
    dim3 gridDim(numPairs);
    dim3 blockDim(std::min(static_cast<int>(tourLength), 256));
    
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
    int maxThreads = std::min(256, 1 << std::min(16, static_cast<int>(tourLength)));
    dim3 evalBlockDim(maxThreads);
    
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
    
    // Wait for completion
    cudaDeviceSynchronize();
    
    // Create result genome
    Genome child(tourLength, tourLength, mode);
    thrust::copy(d_childCity.begin(), d_childCity.end(), child.citySequence.begin());
    
    // Apply same process to job sequence
    thrust::device_vector<size_t> d_parent1Job(parent1.jobSequence);
    thrust::device_vector<size_t> d_parent2Job(parent2.jobSequence);
    thrust::device_vector<size_t> d_childJob(tourLength);
    
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

// Optimized batch processing version
std::vector<Genome> performBatchEAXCrossover(const std::vector<Genome>& parents1,
                                            const std::vector<Genome>& parents2, 
                                            int mode) {
    if (parents1.size() != parents2.size() || parents1.empty()) {
        return {};
    }
    
    uint32_t numPairs = static_cast<uint32_t>(parents1.size());
    uint16_t tourLength = static_cast<uint16_t>(parents1[0].citySequence.size());
    
    // Flatten all parent data
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
    thrust::device_vector<uint8_t> d_adjacencyMatrix(numPairs * tourLength * tourLength, 0);
    thrust::device_vector<uint16_t> d_degrees(numPairs * tourLength, 0);
    thrust::device_vector<Cycle> d_cycles(numPairs * MAX_CYCLES);
    thrust::device_vector<uint16_t> d_numCycles(numPairs, 0);
    thrust::device_vector<float> d_assemblyCosts(numPairs);
    thrust::device_vector<uint32_t> d_bestAssemblies(numPairs);
    thrust::device_vector<float> d_costMatrix(tourLength * tourLength);
    
    thrust::fill(d_costMatrix.begin(), d_costMatrix.end(), 1.0f);
    
    // Launch batch kernels
    dim3 gridDim(numPairs);
    dim3 blockDim(std::min(static_cast<int>(tourLength), 256));
    
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
    
    int maxThreads = std::min(256, 1 << std::min(16, static_cast<int>(tourLength)));
    dim3 evalBlockDim(maxThreads);
    
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
    
    // Process job sequences
    thrust::device_vector<size_t> d_allParents1Job(numPairs * tourLength);
    thrust::device_vector<size_t> d_allParents2Job(numPairs * tourLength);
    thrust::device_vector<size_t> d_allChildrenJob(numPairs * tourLength);
    
    for (uint32_t i = 0; i < numPairs; i++) {
        thrust::copy(parents1[i].jobSequence.begin(), parents1[i].jobSequence.end(),
                    d_allParents1Job.begin() + i * tourLength);
        thrust::copy(parents2[i].jobSequence.begin(), parents2[i].jobSequence.end(),
                    d_allParents2Job.begin() + i * tourLength);
    }
    
    constructOffspringKernel<<<gridDim, blockDim>>>(
        thrust::raw_pointer_cast(d_cycles.data()),
        thrust::raw_pointer_cast(d_numCycles.data()),
        thrust::raw_pointer_cast(d_bestAssemblies.data()),
        thrust::raw_pointer_cast(d_allParents1Job.data()),
        thrust::raw_pointer_cast(d_allParents2Job.data()),
        thrust::raw_pointer_cast(d_allChildrenJob.data()),
        tourLength, numPairs);
    
    // Handle pickup sequences if mode == 1
    thrust::device_vector<size_t> d_allParents1Pickup, d_allParents2Pickup, d_allChildrenPickup;
    if (mode == 1) {
        d_allParents1Pickup.resize(numPairs * tourLength);
        d_allParents2Pickup.resize(numPairs * tourLength);
        d_allChildrenPickup.resize(numPairs * tourLength);
        
        for (uint32_t i = 0; i < numPairs; i++) {
            thrust::copy(parents1[i].pickupSequence.begin(), parents1[i].pickupSequence.end(),
                        d_allParents1Pickup.begin() + i * tourLength);
            thrust::copy(parents2[i].pickupSequence.begin(), parents2[i].pickupSequence.end(),
                        d_allParents2Pickup.begin() + i * tourLength);
        }
        
        constructOffspringKernel<<<gridDim, blockDim>>>(
            thrust::raw_pointer_cast(d_cycles.data()),
            thrust::raw_pointer_cast(d_numCycles.data()),
            thrust::raw_pointer_cast(d_bestAssemblies.data()),
            thrust::raw_pointer_cast(d_allParents1Pickup.data()),
            thrust::raw_pointer_cast(d_allParents2Pickup.data()),
            thrust::raw_pointer_cast(d_allChildrenPickup.data()),
            tourLength, numPairs);
    }
    
    // Wait for completion
    cudaDeviceSynchronize();
    
    // Copy results back to host
    std::vector<Genome> children;
    children.reserve(numPairs);
    
    for (uint32_t i = 0; i < numPairs; i++) {
        Genome child(tourLength, tourLength, mode);
        
        // Copy city sequence
        thrust::copy(d_allChildrenCity.begin() + i * tourLength,
                    d_allChildrenCity.begin() + (i + 1) * tourLength,
                    child.citySequence.begin());
        
        // Copy job sequence
        thrust::copy(d_allChildrenJob.begin() + i * tourLength,
                    d_allChildrenJob.begin() + (i + 1) * tourLength,
                    child.jobSequence.begin());
        
        // Copy pickup sequence if mode == 1
        if (mode == 1) {
            thrust::copy(d_allChildrenPickup.begin() + i * tourLength,
                        d_allChildrenPickup.begin() + (i + 1) * tourLength,
                        child.pickupSequence.begin());
        }
        
        children.push_back(std::move(child));
    }
    
    return children;
}