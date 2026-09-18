#pragma once

#include <vector>
#include "treeClass.h"
#include "POSCARdata.h"
#include "xyz-parser.h"

constexpr double compPar = 0.3;

// -----------------------------------------------
double fitnessFunction(const std::array<Tree,4>& trees, 
	const std::vector<std::vector<PairDistanceData>>& data)
{
	// Note : the fittnes is wrong, need to create a neighbor for each atom
	// and to choose which tree to use + the general tree.
	
	double totalLossEnergySqr = 0.0;
	int numSnapShots = 0;

	for (const auto& simulation : data)
	{
		double lossSimSqr = 0.0;

		for (const auto& snapshot : simulation)
		{
			double energyCand = 0.0;

            // loop over precomputed distances
            for (double r : snapshot.distances)
            {
                double V = trees[0].evaluate(r);

                energyCand += V;
            }

            // Energy error square
            lossSimSqr += (energyCand - snapshot.energyReal) *
                          (energyCand - snapshot.energyReal);

			numSnapShots++;
		}

		// add loss to total loss
		totalLossEnergySqr += lossSimSqr;
	}

	double lossSize = compPar * trees[0].Size();

	double totalLoss =
		totalLossEnergySqr / static_cast<double>(numSnapShots)
		+ lossSize;

	return totalLoss;
}

// -----------------------------------------------
double fitnessFun_2elements(const std::array<Tree,4>& trees, 
	const std::vector<Snapshot>& data)
{
	// add fitness function
	return 0.0;
}