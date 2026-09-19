#pragma once

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>
#include "treeClass.h"
#include "POSCARdata.h"
#include "xyz-parser.h"


// -----------------------------------------------
double fitnessFun_2elements(const std::array<Tree,4>& trees,
	const std::vector<Snapshot>& data,
	int dataSample,
	std::mt19937& rng)
{
	if (data.empty() || dataSample <= 0)
	{
		std::cerr << "Error: No data provided or invalid data sample size." << std::endl;
		return 0.0;
	}

	// take dataSample random snapshots from data
	const int sampleCount = std::min(dataSample, static_cast<int>(data.size()));

	std::vector<int> indices(data.size());
	std::iota(indices.begin(), indices.end(), 0);
	std::shuffle(indices.begin(), indices.end(), rng);

	std::vector<int> sampledIndices(indices.begin(), indices.begin() + sampleCount);

	double energyLossTotal = 0.0;

	for (int index : sampledIndices)
	{
		const Snapshot& snapshot = data[index];

		double EsnapshotTotal = 0.0;
		
		for (const auto& atom : snapshot.atoms)
		{
			for (const auto& neighbor : atom.neighbors)
			{
				double r = neighbor.distance;

				double Vlocal = 0.0;
				double Vtotal = 0.0;
				
				if (atom.type == "W" && neighbor.type == "W")
					Vlocal = trees[0].evaluate(r);
				else if (atom.type == "W" && neighbor.type == "Mo")
					Vlocal = trees[1].evaluate(r);
				else if (atom.type == "Mo" && neighbor.type == "Mo")
					Vlocal = trees[2].evaluate(r);
				
				Vtotal = Vlocal + trees[3].evaluate(r);
				EsnapshotTotal += Vtotal;
			}
		}

		// we divide by number of atoms to acount for bigger snapshots
		energyLossTotal +=
			((EsnapshotTotal - snapshot.energy) *
			 (EsnapshotTotal - snapshot.energy))
			/ static_cast<double>(snapshot.numberOfAtoms);
	}

	// Fitness = MSE = sum(DE ^2) / N
	return energyLossTotal / static_cast<double>(sampleCount);
}


// -----------------------------------------------
// this function is not used for the 2-elements potential
constexpr double compPar = 0.3;

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
