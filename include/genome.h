// genome.h: Header file for TSPJ Genome representation

#ifndef GENOME_H
#define GENOME_H

#include <vector>
#include <iostream>
#include <limits>

class Genome {
public:
    // Sequences representing cities and jobs
    std::vector<size_t> citySequence;
    std::vector<size_t> jobSequence;
    std::vector<size_t> pickupSequence;

    // Fitness value of the genome
    float fitness;

    // Constructor
    Genome() : fitness(std::numeric_limits<float>::max()) {}
    Genome(size_t numCities, size_t numJobs, int mode);

    // Evaluate fitness of the genome
    void evaluateFitness(const std::vector<std::vector<float>>& travelTimes,
                         const std::vector<std::vector<float>>& jobTimes);

    float getFitness();

    // Debugging utility to print the genome
    void print(int mode) const;
};

#endif // GENOME_H
