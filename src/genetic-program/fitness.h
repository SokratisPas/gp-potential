#pragma once

#include <algorithm>
#include <exception>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "treeClass.h"
#include "POSCARdata.h"
#include "xyz-parser.h"

// #include <omp.h>   (for OpenMP functions)

// -----------------------------------------------
double fitnessFun_2elements(
	const Tree& tree_W,
	const Tree& tree_Mo,
	double cutoffIn,
	double cutoffOut,
	const std::vector<Snapshot>& data,
	int dataSample,
	std::mt19937& rng,
	int rank,
	double sizePenaltyFactor
)
{
	if (data.empty() || dataSample <= 0)
		throw std::runtime_error("Error: No data provided or invalid data sample size.");

	// take dataSample random snapshots from data
	const int sampleCount = std::min(dataSample, static_cast<int>(data.size()));

	std::vector<int> indices(data.size());
	std::iota(indices.begin(), indices.end(), 0);
	std::shuffle(indices.begin(), indices.end(), rng);

	std::vector<int> sampledIndices(indices.begin(), indices.begin() + sampleCount);

	double energyLossTotal = 0.0;
	std::exception_ptr evaluationError;

	// parallel region with openMP
	#pragma omp parallel reduction(+: energyLossTotal) shared(evaluationError)
	{
		#pragma omp for schedule(dynamic)
		for (std::size_t snapshotIndex = 0; snapshotIndex < sampledIndices.size(); ++snapshotIndex)
		{
			try
			{
				const Snapshot& snapshot = data[sampledIndices[snapshotIndex]];

				if (snapshot.numberOfAtoms < 1)
					throw std::runtime_error("Warning: Snapshot has no atoms.");

				// catch one atom snapshot
				// also maybe revisit this later
				if (snapshot.numberOfAtoms == 1)
					continue;

				double EsnapshotTotal = 0.0;

				for (const auto& atom : snapshot.atoms)
				{
					// use tree based on the atom type
					const Tree* atomTree = nullptr;
					if (atom.type == "W")
						atomTree = &tree_W;
					else if (atom.type == "Mo")
						atomTree = &tree_Mo;
					else
						throw std::runtime_error("Unsupported atom type: " + atom.type);

					for (const auto& neighbor : atom.neighbors)
					{
						double r = neighbor.distance;
						const std::vector<Neighbor>& neighbors = atom.neighbors;
						
						const double Vij = atomTree->evaluate(
							r,
							neighbors,
							cutoffIn,
							cutoffOut
						);

						EsnapshotTotal += Vij;
					}
				}

				// we divede by Natoms, bcz energy - Natoms have a linear relationship 
				energyLossTotal +=
					((EsnapshotTotal - snapshot.energy) *
					 (EsnapshotTotal - snapshot.energy))
					/ static_cast<double>(snapshot.numberOfAtoms);
			}
			catch (...)
			{
				#pragma omp critical(fitness_evaluation_error_lock)
				{
					if (!evaluationError)
						evaluationError = std::current_exception();
				}
			}
		}
	}
	// end of parallel region
	// throw exception now from master thread
	if (evaluationError)
		std::rethrow_exception(evaluationError);

	// calc size penalty
	const int totalSize = tree_W.Size() + tree_Mo.Size();

	// each island has different size penalty	
	double sizePenalty = totalSize * sizePenaltyFactor * (rank + 1);	

	// Fitness = MSE = sum[(DE ^2)] / Nsnapshots
	double fitnessTotal = energyLossTotal / static_cast<double>(sampleCount);

	return fitnessTotal + sizePenalty;
}
