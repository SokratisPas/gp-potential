#pragma once

#include "treeClass.h"

// --------------------------------------------------------
// Individual class
class Individual {
public:
	Tree tree;

	double fitness = 1e6;

	double energyLoss = 0.0;
	double forceLoss = 0.0;

	bool evaluated = false;
};