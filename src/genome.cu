// genome.cu: Representation of chromosomes for TSPJ problem

#include "genome.h"
#include <vector>
#include <algorithm>
#include <random>

// Constructor for Genome
Genome::Genome(size_t numCities, size_t numJobs, int mode)
    : citySequence(numCities), jobSequence(numJobs)  {
    // Initialize city sequence as a permutation of [1, numCities]
    std::iota(citySequence.begin(), citySequence.end(), 1); // City indices [1, numCities]
    std::iota(jobSequence.begin(), jobSequence.end(), 1);   // Job indices [1, numJobs]
    if(mode == 1){
        pickupSequence.resize(numCities);
        std::iota(pickupSequence.begin(), pickupSequence.end(), 1); // Pickup sequence [1, numCities]
    }
    

    // Randomize city and job sequences
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(citySequence.begin(), citySequence.end(), g);
    std::shuffle(jobSequence.begin(), jobSequence.end(), g);
    std::shuffle(pickupSequence.begin(), pickupSequence.end(), g);
    if(mode == 1){
        std::shuffle(pickupSequence.begin(), pickupSequence.end(), g);
    }

    fitness = std::numeric_limits<float>::max(); // Default high fitness
}

// // Evaluate fitness of the genome
// void Genome::evaluateFitness(const std::vector<std::vector<float>>& travelTimes,
//                               const std::vector<std::vector<float>>& jobTimes) {
//     float completionTime = 0.0f;
//     float currentTime = 0.0f;
//     size_t prevCity = 0; // Start from depot

//     for (size_t i = 0; i < citySequence.size(); ++i) {
//         size_t city = citySequence[i];
//         size_t job = jobSequence[i];

//         // Travel to the next city
//         currentTime += travelTimes[prevCity][city];
        
//         // Start the job at the city
//         completionTime = std::max(completionTime, currentTime + jobTimes[city][job]);
        
//         // Move to the next city
//         prevCity = city;
//     }

//     // Return to depot
//     currentTime += travelTimes[prevCity][0];

//     // Update fitness to reflect makespan
//     fitness = std::max(completionTime, currentTime);
// }

float Genome::getFitness(){
    return fitness;
}

// Print genome for debugging
void Genome::print(int mode) const {
    std::cout << "City Sequence: ";
    for (const auto& city : citySequence) {
        std::cout << city << " ";
    }
    std::cout << "\nJob Sequence: ";
    for (const auto& job : jobSequence) {
        std::cout << job << " ";
    }
    if (mode == 1) { // Only print pickup sequence in TSP-J with Pickup
        std::cout << "\nPickup Sequence: ";
        for (const auto& pickup : pickupSequence) {
            std::cout << pickup << " ";
        }
    }
    std::cout << "\nFitness: " << fitness << "\n";
}
