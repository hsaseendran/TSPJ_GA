#include "cost_matrix.h"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <cassert>

// Kernel to compute the cost matrix
__global__ void computeCostMatrixKernel(float* costMatrix, const float* coordinates, int numCities) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numCities && col < numCities) {
        if (row == col) {
            costMatrix[row * numCities + col] = 0.0f; // Distance to self is 0
        } else {
            float x1 = coordinates[row * 2];
            float y1 = coordinates[row * 2 + 1];
            float x2 = coordinates[col * 2];
            float y2 = coordinates[col * 2 + 1];
            costMatrix[row * numCities + col] = sqrtf((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
        }
    }
}

// Constructor
CostMatrix::CostMatrix(int numCities) : numCities(numCities), d_costMatrix(nullptr) {
    size_t size = numCities * numCities * sizeof(float);
    cudaMalloc(&d_costMatrix, size);
}

// Destructor
CostMatrix::~CostMatrix() {
    cudaFree(d_costMatrix);
}

// Initialize the cost matrix on the GPU
void CostMatrix::initialize(const std::vector<std::pair<float, float>>& coordinates) {
    assert(coordinates.size() == numCities);
    
    // Flatten coordinates into a single array
    std::vector<float> flattenedCoordinates(numCities * 2);
    for (int i = 0; i < numCities; ++i) {
        flattenedCoordinates[i * 2] = coordinates[i].first;
        flattenedCoordinates[i * 2 + 1] = coordinates[i].second;
    }

    // Allocate GPU memory for coordinates
    float* d_coordinates;
    size_t coordSize = numCities * 2 * sizeof(float);
    cudaMalloc(&d_coordinates, coordSize);
    cudaMemcpy(d_coordinates, flattenedCoordinates.data(), coordSize, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((numCities + blockDim.x - 1) / blockDim.x,
                 (numCities + blockDim.y - 1) / blockDim.y);
    computeCostMatrixKernel<<<gridDim, blockDim>>>(d_costMatrix, d_coordinates, numCities);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Launch Error1: " << cudaGetErrorString(err) << std::endl;
    }

    cudaFree(d_coordinates);
}

// Overload for Initialize cost matrix
void CostMatrix::initialize(const std::vector<std::vector<float>>& predefinedMatrix) {
    assert(predefinedMatrix.size() == numCities);

    // Flatten the 2D matrix into a 1D array
    std::vector<float> flatMatrix;
    for (const auto& row : predefinedMatrix) {
        assert(row.size() == numCities); // Ensure square matrix
        flatMatrix.insert(flatMatrix.end(), row.begin(), row.end());
    }

    // Copy to the GPU memory
    cudaMemcpy(d_costMatrix, flatMatrix.data(), numCities * numCities * sizeof(float), cudaMemcpyHostToDevice);
}

// Retrieve the cost matrix from GPU
std::vector<std::vector<float>> CostMatrix::getCostMatrix() const {
    std::vector<std::vector<float>> hostMatrix(numCities, std::vector<float>(numCities));
    std::vector<float> flatMatrix(numCities * numCities);

    cudaMemcpy(flatMatrix.data(), d_costMatrix, numCities * numCities * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < numCities; ++i) {
        std::copy(flatMatrix.begin() + i * numCities, flatMatrix.begin() + (i + 1) * numCities, hostMatrix[i].begin());
    }

    for (int i = 0; i < numCities; ++i) {
        std::cout << "Row " << i << ": ";
        for (float value : hostMatrix[i]) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    return hostMatrix;
}

// Get a pointer to the GPU memory for the cost matrix
const float* CostMatrix::getDeviceCostMatrix() const {
    return d_costMatrix;
}
