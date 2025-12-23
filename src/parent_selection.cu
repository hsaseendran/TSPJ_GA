// parent_selection.cu: Parent selection for TSPJ problem using GPU

#include "parent_selection.h"
#include "genome.h"
#include <vector>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <iostream>

// Kernel for GPU-based tournament selection
__global__ void tournamentSelectionKernel(const float* fitness, size_t populationSize,
                                          size_t numParents, size_t tournamentSize,
                                          size_t* selectedIndices, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numParents) {
        thrust::default_random_engine rng(seed + idx);
        thrust::uniform_int_distribution<size_t> dist(0, populationSize - 1);

        size_t bestIndex = dist(rng);
        float bestFitness = fitness[bestIndex];

        for (size_t i = 1; i < tournamentSize; ++i) {
            size_t candidateIndex = dist(rng);
            float candidateFitness = fitness[candidateIndex];

            if (candidateFitness < bestFitness) {
                bestIndex = candidateIndex;
                bestFitness = candidateFitness;
            }
        }

        selectedIndices[idx] = bestIndex;
    }
}

// Host function for GPU-based parent selection
std::vector<Genome> selectParents(const std::vector<Genome>& population,
                                  size_t numParents, size_t tournamentSize) {
    size_t populationSize = population.size();

    // Extract fitness values
    std::vector<float> fitness(populationSize);
    for (size_t i = 0; i < populationSize; ++i) {
        fitness[i] = population[i].fitness;
    }

    // Allocate device memory
    float* d_fitness;
    size_t* d_selectedIndices;
    cudaMalloc(&d_fitness, sizeof(float) * populationSize);
    cudaMalloc(&d_selectedIndices, sizeof(size_t) * numParents);

    // Copy fitness values to device
    cudaMemcpy(d_fitness, fitness.data(), sizeof(float) * populationSize, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numParents + threadsPerBlock - 1) / threadsPerBlock;
    size_t* h_selectedIndices = new size_t[numParents];

    tournamentSelectionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_fitness, populationSize, numParents, tournamentSize, d_selectedIndices, time(NULL));

    // Copy selected indices back to host
    cudaMemcpy(h_selectedIndices, d_selectedIndices, sizeof(size_t) * numParents, cudaMemcpyDeviceToHost);

    // Collect selected parents
    std::vector<Genome> selectedParents;
    selectedParents.reserve(numParents);
    for (size_t i = 0; i < numParents; ++i) {
        selectedParents.push_back(population[h_selectedIndices[i]]);
    }

    // Free device memory
    cudaFree(d_fitness);
    cudaFree(d_selectedIndices);
    delete[] h_selectedIndices;

    return selectedParents;
}
