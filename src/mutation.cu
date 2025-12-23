// mutation.cu: Stagnation-aware mutation with escape mechanisms

#include "mutation.h"
#include "genome.h"
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

// Simple 2-opt with escape probability based on stagnation
void apply2OptWithEscape(std::vector<size_t>& sequence, float mutationRate, size_t stagnationCount) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> probDist(0.0f, 1.0f);
    
    if (probDist(gen) < mutationRate) {
        std::uniform_int_distribution<size_t> indexDist(0, sequence.size() - 1);
        
        // Calculate escape probability based on stagnation
        float escapeProb = std::min(0.5f, 0.05f + (stagnationCount / 500.0f));
        
        // Try to find an improving move first (simple heuristic)
        bool moveApplied = false;
        int attempts = 5; // Quick attempts to find a good move
        
        for (int attempt = 0; attempt < attempts && !moveApplied; attempt++) {
            size_t i = indexDist(gen);
            size_t j = indexDist(gen);
            
            if (i > j) std::swap(i, j);
            if (j <= i + 1) continue;
            
            // Simple heuristic: prefer moves that change more of the tour
            float moveSize = static_cast<float>(j - i) / sequence.size();
            float applyProb = (moveSize > 0.3f) ? 0.8f : 0.3f; // Prefer larger moves
            
            // Apply move based on probability or escape mechanism
            if (probDist(gen) < applyProb || probDist(gen) < escapeProb) {
                std::reverse(sequence.begin() + i, sequence.begin() + j + 1);
                moveApplied = true;
            }
        }
        
        // If no move applied yet and stagnation is high, force a random move
        if (!moveApplied && stagnationCount > 200) {
            size_t i = indexDist(gen);
            size_t j = indexDist(gen);
            if (i > j) std::swap(i, j);
            if (j > i + 1) {
                std::reverse(sequence.begin() + i, sequence.begin() + j + 1);
            }
        }
    }
}

// 3-opt style mutation for higher disruption
void apply3OptStyle(std::vector<size_t>& sequence, float mutationRate) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> probDist(0.0f, 1.0f);
    
    if (probDist(gen) < mutationRate) {
        std::uniform_int_distribution<size_t> indexDist(0, sequence.size() - 1);
        
        // Select three points
        std::vector<size_t> points = {indexDist(gen), indexDist(gen), indexDist(gen)};
        std::sort(points.begin(), points.end());
        
        if (points[0] != points[1] && points[1] != points[2]) {
            // Randomly choose a 3-opt move type
            int moveType = gen() % 4;
            
            switch(moveType) {
                case 0: // Reverse first segment
                    std::reverse(sequence.begin() + points[0], sequence.begin() + points[1]);
                    break;
                case 1: // Reverse second segment
                    std::reverse(sequence.begin() + points[1], sequence.begin() + points[2]);
                    break;
                case 2: // Reverse both segments
                    std::reverse(sequence.begin() + points[0], sequence.begin() + points[1]);
                    std::reverse(sequence.begin() + points[1], sequence.begin() + points[2]);
                    break;
                case 3: // Swap segments
                    std::rotate(sequence.begin() + points[0], 
                               sequence.begin() + points[1], 
                               sequence.begin() + points[2]);
                    break;
            }
        }
    }
}

// Double-bridge move for escaping local optima
void applyDoubleBridge(std::vector<size_t>& sequence, float mutationRate) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> probDist(0.0f, 1.0f);
    
    if (probDist(gen) < mutationRate && sequence.size() >= 8) {
        std::uniform_int_distribution<size_t> indexDist(1, sequence.size() / 4);
        
        // Select 4 random positions
        size_t pos1 = indexDist(gen);
        size_t pos2 = pos1 + indexDist(gen);
        size_t pos3 = pos2 + indexDist(gen);
        size_t pos4 = std::min(pos3 + indexDist(gen), sequence.size() - 1);
        
        // Perform double-bridge move
        std::vector<size_t> newSeq;
        newSeq.insert(newSeq.end(), sequence.begin(), sequence.begin() + pos1);
        newSeq.insert(newSeq.end(), sequence.begin() + pos3, sequence.begin() + pos4);
        newSeq.insert(newSeq.end(), sequence.begin() + pos2, sequence.begin() + pos3);
        newSeq.insert(newSeq.end(), sequence.begin() + pos1, sequence.begin() + pos2);
        newSeq.insert(newSeq.end(), sequence.begin() + pos4, sequence.end());
        
        sequence = newSeq;
    }
}

// Main mutation function with stagnation awareness
void performMutation(Genome& genome, float mutationRate, int mode, size_t stagnationCount) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> probDist(0.0f, 1.0f);
    
    // Choose mutation strategy based on stagnation level
    if (stagnationCount < 100) {
        // Low stagnation: standard 2-opt
        apply2OptWithEscape(genome.citySequence, mutationRate, stagnationCount);
        if (mode == 1 && !genome.pickupSequence.empty()) {
            apply2OptWithEscape(genome.pickupSequence, mutationRate, stagnationCount);
        }
    } else if (stagnationCount < 300) {
        // Medium stagnation: mix of 2-opt and 3-opt
        if (probDist(gen) < 0.7f) {
            apply2OptWithEscape(genome.citySequence, mutationRate, stagnationCount);
        } else {
            apply3OptStyle(genome.citySequence, mutationRate);
        }
        
        if (mode == 1 && !genome.pickupSequence.empty()) {
            apply2OptWithEscape(genome.pickupSequence, mutationRate, stagnationCount);
        }
    } else {
        // High stagnation: use more disruptive mutations
        float mutationType = probDist(gen);
        if (mutationType < 0.4f) {
            apply3OptStyle(genome.citySequence, mutationRate * 1.5f); // Higher rate
        } else if (mutationType < 0.7f) {
            applyDoubleBridge(genome.citySequence, mutationRate);
        } else {
            // Apply multiple 2-opts
            apply2OptWithEscape(genome.citySequence, mutationRate, stagnationCount);
            apply2OptWithEscape(genome.citySequence, mutationRate / 2, stagnationCount);
        }
        
        if (mode == 1 && !genome.pickupSequence.empty()) {
            if (probDist(gen) < 0.5f) {
                apply3OptStyle(genome.pickupSequence, mutationRate);
            } else {
                apply2OptWithEscape(genome.pickupSequence, mutationRate, stagnationCount);
            }
        }
    }
    
    // Job sequence mutation - increase rate with stagnation
    float jobMutationRate = mutationRate * (1.0f + stagnationCount / 500.0f);
    if (probDist(gen) < jobMutationRate && genome.jobSequence.size() > 1) {
        std::uniform_int_distribution<size_t> indexDist(0, genome.jobSequence.size() - 1);
        
        // Multiple swaps for high stagnation
        int numSwaps = (stagnationCount > 200) ? 2 : 1;
        for (int s = 0; s < numSwaps; s++) {
            size_t idx1 = indexDist(gen);
            size_t idx2 = indexDist(gen);
            if (idx1 != idx2) {
                std::swap(genome.jobSequence[idx1], genome.jobSequence[idx2]);
            }
        }
    }
}

// Batch mutation function
void performBatchMutation(std::vector<Genome>& genomes, float mutationRate, int mode,
                         const std::vector<std::vector<float>>& travelTimes,
                         size_t stagnationCount) {
    // Process each genome with stagnation-aware mutation
    for (auto& genome : genomes) {
        performMutation(genome, mutationRate, mode, stagnationCount);
    }
}

// Initialize cost matrix (stub for compatibility)
void initializeMutationCostMatrix(const std::vector<std::vector<float>>& travelTimes) {
    // Not used in this simplified version
}

void cleanupMutationCostMatrix() {
    // Not used in this simplified version
}