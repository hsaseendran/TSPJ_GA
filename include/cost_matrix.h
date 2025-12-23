#ifndef COST_MATRIX_H
#define COST_MATRIX_H

#include <vector>

/**
 * Class to compute and manage the cost matrix for TSP.
 */
class CostMatrix {
public:
    /**
     * Constructor
     * @param numCities Number of cities in the problem.
     */
    CostMatrix(int numCities);

    /**
     * Destructor
     */
    ~CostMatrix();

    /**
     * Initialize the cost matrix on GPU.
     * @param coordinates Vector of city coordinates in (x, y) format.
     */
    void initialize(const std::vector<std::pair<float, float>>& coordinates);
    void initialize(const std::vector<std::vector<float>>& predefinedMatrix); // overload

    /**
     * Retrieve the cost matrix from GPU.
     * @return A 2D vector representing the cost matrix.
     */
    std::vector<std::vector<float>> getCostMatrix() const;

    /**
     * Get a pointer to the GPU memory for the cost matrix.
     * @return Device pointer to the cost matrix.
     */
    const float* getDeviceCostMatrix() const;

private:
    int numCities; // Number of cities
    float* d_costMatrix; // Device memory for the cost matrix
};

#endif // COST_MATRIX_H
