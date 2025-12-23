#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include "cost_matrix.h"

// Function to generate the expected cost matrix for validation
std::vector<std::vector<float>> generateExpectedCostMatrix(const std::vector<std::pair<float, float>>& coordinates) {
    int numCities = coordinates.size();
    std::vector<std::vector<float>> expectedMatrix(numCities, std::vector<float>(numCities));

    for (int i = 0; i < numCities; ++i) {
        for (int j = 0; j < numCities; ++j) {
            if (i == j) {
                expectedMatrix[i][j] = 0.0f; // Distance to self
            } else {
                float dx = coordinates[i].first - coordinates[j].first;
                float dy = coordinates[i].second - coordinates[j].second;
                expectedMatrix[i][j] = sqrtf(dx * dx + dy * dy);
            }
        }
    }
    return expectedMatrix;
}

void testCostMatrix() {
    // Define a small dataset of TSP coordinates
    std::vector<std::pair<float, float>> coordinates = {
        {0.0, 0.0}, // City 0
        {3.0, 4.0}, // City 1
        {6.0, 8.0}, // City 2
    };
    int numCities = coordinates.size();

    // Initialize the CostMatrix
    CostMatrix costMatrix(numCities);
    costMatrix.initialize(coordinates);

    // Retrieve the generated cost matrix
    auto generatedMatrix = costMatrix.getCostMatrix();

    // Generate the expected cost matrix
    auto expectedMatrix = generateExpectedCostMatrix(coordinates);

    // Compare the generated matrix with the expected matrix
    for (int i = 0; i < numCities; ++i) {
        for (int j = 0; j < numCities; ++j) {
            assert(std::abs(generatedMatrix[i][j] - expectedMatrix[i][j]) < 1e-5);
        }
    }

    std::cout << "Cost Matrix Test Passed!" << std::endl;
}

int main() {
    testCostMatrix();
    return 0;
}
