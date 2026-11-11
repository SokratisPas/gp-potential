#pragma once

#include <array>
#include "treeClass.h"


// --------------------------------------------------------
// individual type
enum class IndType {
	A,
	B
};


// --------------------------------------------------------
// Individual class
class Individual {
public:
// Each individual has three trees: A-A, A-B, and B-B interactions.
	
	std::array<Tree, 3> trees;	// {treeAA, treeAB, treeBB}

	double fitness = std::numeric_limits<double>::infinity();

	double energyLoss = 0.0;

	bool evaluated = false;

	// in "GeneticProgram::ReplaceInd" we use "std::find"
	// so we need the == and != operators
	// so we define them for Individual, Tree, Node
	bool operator==(const Individual& other) const;
	bool operator!=(const Individual& other) const;
};


// --------------------------------------------------------
bool Individual::operator==(const Individual& other) const
{
	return trees == other.trees
		&& fitness == other.fitness
		&& energyLoss == other.energyLoss
		&& evaluated == other.evaluated;
}

// --------------------------------------------------------
bool Individual::operator!=(const Individual& other) const
{
	return !(*this == other);
}