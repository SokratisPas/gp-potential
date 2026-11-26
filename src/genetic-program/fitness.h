#pragma once

#include <algorithm>
#include <exception>
#include <numeric>
#include <optional>
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
	std::optional<int> dataSample,
	std::mt19937& rng,
	int rank,
	double sizePenaltyFactor
)
{
	if (data.empty() || (dataSample && *dataSample <= 0))
		throw std::runtime_error("Error: No data provided or invalid data sample size.");

	// (to remember: std::optional overloads the * operator, it's not a pointer)
	const std::size_t sampleCount = dataSample
		? std::min(static_cast<std::size_t>(*dataSample), data.size())
		: data.size();
	std::vector<std::size_t> snapshotIndices(data.size());
	std::iota(snapshotIndices.begin(), snapshotIndices.end(), 0);
	if (dataSample)
		std::shuffle(snapshotIndices.begin(), snapshotIndices.end(), rng);
	snapshotIndices.resize(sampleCount);

	double energyLossTotal = 0.0;
	std::exception_ptr evaluationError;

	// parallel region with openMP
	#pragma omp parallel reduction(+: energyLossTotal) shared(evaluationError)
	{
		#pragma omp for schedule(dynamic)
		for (std::size_t snapshotIndex = 0; snapshotIndex < snapshotIndices.size(); ++snapshotIndex)
		{
			try
			{
				const Snapshot& snapshot = data[snapshotIndices[snapshotIndex]];

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

				// - we divede by Natoms, bcz energy - Natoms have a linear relationship 
				//   bigger snapshots would contribute bigger errors
				// - in the dataset we have the total energy of the snapshot
				//   so we also calculate the total energy as a "sum of all the energy of each atom"
				// - we calc an EAM type energy so each not a sum excactly like a pair potential
				energyLossTotal +=
					((EsnapshotTotal - snapshot.energy) / static_cast<double>(snapshot.numberOfAtoms)) *
					((EsnapshotTotal - snapshot.energy) / static_cast<double>(snapshot.numberOfAtoms));
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

	// Fitness = MSE = sum[(DE ^2) / Nsnapshot] / NsampleTotal
	double fitnessTotal = energyLossTotal / static_cast<double>(sampleCount);

	return fitnessTotal + sizePenalty;
}
