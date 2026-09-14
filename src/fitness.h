#pragma once

#include <vector>
#include "treeClass.h"
#include "POSCARdata.h"

constexpr double compPar = 0.3;

// -----------------------------------------------
double fitnessFunction(const Tree& tree, 
	const std::vector<std::vector<PairDistanceData>>& data)
{
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
                double V = tree.evaluate(r);

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

	double lossSize = compPar * tree.Size();

	double totalLoss =
		totalLossEnergySqr / static_cast<double>(numSnapShots)
		+ lossSize;

	return totalLoss;
}