#pragma once

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
	// when an atom is W, we use tree_W etc...
	// so we capture the eam idea, the atomic environment depends on the atom type
	// and Vtotal = V_W + V_Mo
	Tree tree_W;
	Tree tree_Mo;

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
	return tree_W == other.tree_W
		&& tree_Mo == other.tree_Mo
		&& fitness == other.fitness
		&& energyLoss == other.energyLoss
		&& evaluated == other.evaluated;
}

// --------------------------------------------------------
bool Individual::operator!=(const Individual& other) const
{
	return !(*this == other);
}