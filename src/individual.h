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
// each individual has 4 trees:
// tree AA, AB, BB -> interactions between atoms A-A, A-B, B-B
// tree General    -> general interaction term between neighbors
	
	std::array<Tree,4> trees;	// {treeAA, treeAB, treeBB, treeGeneral}

	IndType indType;	// A or B for the two element material

	double fitness = std::numeric_limits<double>::infinity();

	double energyLoss = 0.0;

	bool evaluated = false;
};