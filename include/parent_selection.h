// parent_selection.h: Header file for GPU-based parent selection in TSPJ

#ifndef PARENT_SELECTION_H
#define PARENT_SELECTION_H

#include "genome.h"
#include <vector>

// Host function for GPU-based parent selection
std::vector<Genome> selectParents(const std::vector<Genome>& population,
                                  size_t numParents, size_t tournamentSize);

#endif // PARENT_SELECTION_H
