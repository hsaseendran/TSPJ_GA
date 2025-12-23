// eax_cost_integration.cu: Implementation of cost-aware EAX crossover

#include "eax_cost_integration.h"
#include "crossover.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <iostream>

// Global device cost matrices (managed internally)
static thrust::device_vector<float>* g_d_travelCosts = nullptr;
static thrust::device_vector<float>* g_d_jobCosts = nullptr;
static size_t g_numCities = 0;
static size_t g_numJobs = 0;

void initializeEAXCostMatrices(const std::vector<std::vector<float>>& travelTimes,
                               const std::vector<std::vector<float>>& jobTimes) {
    // Clean up existing matrices
    cleanupEAXCostMatrices();
    
    g_numCities = travelTimes.size();
    g_numJobs = jobTimes[0].size();
    
    // Flatten travel times matrix
    std::vector<float> flatTravelTimes(g_numCities * g_numCities);
    for (size_t i = 0; i < g_numCities; i++) {
        for (size_t j = 0; j < g_numCities; j++) {
            flatTravelTimes[i * g_numCities + j] = travelTimes[i][j];
        }
    }
    
    // Flatten job times matrix
    std::vector<float> flatJobTimes(g_numCities * g_numJobs);
    for (size_t i = 0; i < g_numCities; i++) {
        for (size_t j = 0; j < g_numJobs; j++) {
            flatJobTimes[i * g_numJobs + j] = jobTimes[i][j];
        }
    }
    
    // Allocate and copy to device
    g_d_travelCosts = new thrust::device_vector<float>(flatTravelTimes);
    g_d_jobCosts = new thrust::device_vector<float>(flatJobTimes);
    
    std::cout << "EAX cost matrices initialized: " 
              << g_numCities << " cities, " << g_numJobs << " jobs" << std::endl;
}

void cleanupEAXCostMatrices() {
    delete g_d_travelCosts;
    delete g_d_jobCosts;
    g_d_travelCosts = nullptr;
    g_d_jobCosts = nullptr;
}

// Enhanced cost calculation kernel for EAX evaluation
__global__ void calculateTourCostKernel(const size_t* tour, const float* travelCosts, 
                                        size_t tourLength, float* totalCost) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float sharedCosts[256];
    
    float localCost = 0.0f;
    if (idx < tourLength) {
        size_t from = tour[idx];
        size_t to = tour[(idx + 1) % tourLength];
        if (from < tourLength && to < tourLength) {
            localCost = travelCosts[from * tourLength + to];
        }
    }
    
    sharedCosts[threadIdx.x] = localCost;
    __syncthreads();
    
    // Parallel reduction to sum costs
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && threadIdx.x + stride < blockDim.x) {
            sharedCosts[threadIdx.x] += sharedCosts[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(totalCost, sharedCosts[0]);
    }
}

// Modified EAX assembly evaluation with actual costs
__global__ void evaluateAssembliesWithCostsKernel(const Cycle* cycles, const uint16_t* numCycles,
                                                   const float* travelCosts, const float* jobCosts,
                                                   float* assemblyCosts, uint32_t* bestAssemblies,
                                                   uint16_t tourLength, uint32_t numPairs) {
    uint32_t pairIdx = blockIdx.x;
    uint32_t assemblyIdx = threadIdx.x;
    
    if (pairIdx >= numPairs) return;
    
    uint16_t cycleCount = numCycles[pairIdx];
    uint32_t maxAssemblies = min(256U, 1U << cycleCount); // Limit to block size
    
    __shared__ float sharedCosts[256];
    __shared__ uint32_t sharedAssemblies[256];
    __shared__ size_t reconstructedTour[MAX_CITIES];
    
    // Initialize shared memory
    if (threadIdx.x == 0) {
        for (int i = 0; i < 256; i++) {
            sharedCosts[i] = INFINITY;
            sharedAssemblies[i] = 0;
        }
        // Initialize default tour
        for (uint16_t i = 0; i < tourLength; i++) {
            reconstructedTour[i] = i; // Default sequence 0,1,2,...
        }
    }
    __syncthreads();
    
    float cost = INFINITY;
    if (assemblyIdx < maxAssemblies) {
        // Reconstruct tour based on assembly decision
        uint32_t cycleOffset = pairIdx * MAX_CYCLES;
        
        // Start with a valid base tour
        bool validTour = true;
        
        // Simple cost calculation based on cycles
        cost = 0.0f;
        for (uint16_t cycleIdx = 0; cycleIdx < cycleCount && validTour; cycleIdx++) {
            bool includeCycle = (assemblyIdx >> cycleIdx) & 1;
            const Cycle& cycle = cycles[cycleOffset + cycleIdx];
            
            if (includeCycle) {
                // Add cost for including this cycle
                for (uint16_t i = 0; i < cycle.length; i++) {
                    uint16_t from = cycle.cities[i];
                    uint16_t to = cycle.cities[(i + 1) % cycle.length];
                    if (from < tourLength && to < tourLength) {
                        cost += travelCosts[from * tourLength + to];
                    } else {
                        validTour = false;
                        break;
                    }
                }
            } else {
                // Add penalty for not including cycle (encouraging diversity)
                cost += 10.0f;
            }
        }
        
        // Add small randomization to break ties
        cost += assemblyIdx * 0.001f;
        
        if (!validTour) cost = INFINITY;
    }
    
    sharedCosts[threadIdx.x] = cost;
    sharedAssemblies[threadIdx.x] = assemblyIdx;
    __syncthreads();
    
    // Find minimum cost assembly using parallel reduction
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

Genome performCostAwareEAXCrossover(const Genome& parent1, const Genome& parent2, int mode) {
    if (!g_d_travelCosts || !g_d_jobCosts) {
        // Fallback to simple EAX if cost matrices not initialized
        std::cerr << "Warning: Cost matrices not initialized, using simple EAX" << std::endl;
        return performEAXCrossover(parent1, parent2, mode);
    }
    
    // Use the enhanced EAX with actual cost evaluation
    uint16_t tourLength = parent1.citySequence.size();
    uint32_t numPairs = 1;
    
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
    
    // Launch kernels (same as before except for assembly evaluation)
    dim3 gridDim(numPairs);
    dim3 blockDim(static_cast<int>(std::min(static_cast<int>(tourLength), 256)));
    
    extractEdgesKernel<<<gridDim, blockDim>>>(
        thrust::raw_pointer_cast(d_parent1City.data()),
        thrust::raw_pointer_cast(d_parent2City.data()),
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
    
    // Use cost-aware assembly evaluation
    int maxThreads = std::min(256, 1 << std::min(16, static_cast<int>(tourLength)));
    dim3 evalBlockDim(maxThreads);
    evaluateAssembliesWithCostsKernel<<<gridDim, evalBlockDim>>>(
        thrust::raw_pointer_cast(d_cycles.data()),
        thrust::raw_pointer_cast(d_numCycles.data()),
        thrust::raw_pointer_cast(g_d_travelCosts->data()),
        thrust::raw_pointer_cast(g_d_jobCosts->data()),
        thrust::raw_pointer_cast(d_assemblyCosts.data()),
        thrust::raw_pointer_cast(d_bestAssemblies.data()),
        tourLength, numPairs);
    
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

std::vector<Genome> performBatchCostAwareEAXCrossover(const std::vector<Genome>& parents1,
                                                      const std::vector<Genome>& parents2, 
                                                      int mode) {
    if (!g_d_travelCosts || !g_d_jobCosts) {
        // Fallback to simple batch EAX if cost matrices not initialized
        std::cerr << "Warning: Cost matrices not initialized, using simple batch EAX" << std::endl;
        return performBatchEAXCrossover(parents1, parents2, mode);
    }
    
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
    
    // Launch batch kernels
    dim3 gridDim(numPairs);
    dim3 blockDim(static_cast<int>(std::min(static_cast<int>(tourLength), 256)));
    
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
    
    // Use cost-aware assembly evaluation
    int maxThreads = std::min(256, 1 << std::min(16, static_cast<int>(tourLength)));
    dim3 evalBlockDim(maxThreads);
    evaluateAssembliesWithCostsKernel<<<gridDim, evalBlockDim>>>(
        thrust::raw_pointer_cast(d_cycles.data()),
        thrust::raw_pointer_cast(d_numCycles.data()),
        thrust::raw_pointer_cast(g_d_travelCosts->data()),
        thrust::raw_pointer_cast(g_d_jobCosts->data()),
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
    
    // Copy results back to host
    std::vector<Genome> children;
    children.reserve(numPairs);
    
    for (uint32_t i = 0; i < numPairs; i++) {
        Genome child(tourLength, tourLength, mode);
        
        thrust::copy(d_allChildrenCity.begin() + i * tourLength,
                    d_allChildrenCity.begin() + (i + 1) * tourLength,
                    child.citySequence.begin());
        
        thrust::copy(d_allChildrenJob.begin() + i * tourLength,
                    d_allChildrenJob.begin() + (i + 1) * tourLength,
                    child.jobSequence.begin());
        
        if (mode == 1) {
            thrust::copy(d_allChildrenPickup.begin() + i * tourLength,
                        d_allChildrenPickup.begin() + (i + 1) * tourLength,
                        child.pickupSequence.begin());
        }
        
        children.push_back(std::move(child));
    }
    
    return children;
}