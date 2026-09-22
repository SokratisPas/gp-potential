#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <functional>
#include <stdexcept>

#include "mpi.h"

#include "primitives.h"
#include "treeClass.h"
#include "individual.h"
#include "geneticProgram.h"
#include "POSCARdata.h"
#include "fitness.h"
#include "xyz-parser.h"


/* TO DO
- refactor geneticProgram.h
- add migration
- add Ptournament (and Temperature)
- check how many samples to take (maybe take one sample per evolution loop)
- make input files (not sure if its necessery)
- python runner
- add more primitives 
- make fitnes function better (MSE, complexity penalty)
- add crossover max depth
- add max depth in general
*/ 


int main(int argc, char *argv[])
{
	// ----------------------------------------------
	// GP parameters
	constexpr double cutoff = 5.0;
	int popSize				= 100;
	int gens				= 10;
	int initialIndMaxDepth	= 5;
	int tournamentSize		= 3;
	double mutationProb		= 0.2;
	std::pair constRange	= { -10.0, 10.0 };
	int NdataSample			= 50;					// number of random snapshots for fitness function
	constexpr int NlocalInds 		= 2;   		// number of inds each process sends to global hof
	constexpr int NgensToSendInds 	= 10;    	// number of generations to update the global hof
												// keep in mind each process updates its own individuals 
												// in the global hof independently
	

	// ----------------------------------------------	
	// DATA
	XYZParser W_Mo_parser(cutoff);
	
	std::vector<Snapshot> W_Mo_data = 
		W_Mo_parser.parse("src/data/W-Mo/trainset.xyz");	// size = 8938 snapshots


	// use the last data
	int lastNdata = 100;
	std::vector<Snapshot> reduced_data(W_Mo_data.end() - lastNdata, W_Mo_data.end());

	// ----------------------------------------------	
	// Initialize global hof
	GlobalHOF globalHof;

	// ----------------------------------------------	
	// MPI 
	int numtasks, rank;

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (numtasks < 2)
	{
		throw std::runtime_error("Give process number >= 2!\n");
	}

	globalHof.hofIndividuals.resize(static_cast<size_t>(numtasks) * NlocalInds);
		
	// ----------------------------------------------
	// Genetic Program
	GeneticProgram geneticProgram(
		popSize,
		gens,
		initialIndMaxDepth,
		tournamentSize,
		mutationProb,
		constRange,
		reduced_data,
		NdataSample,
		fitnessFun_2elements,
		rank,
		numtasks,
		&globalHof,
		NlocalInds,
        NgensToSendInds 
	);

	geneticProgram.Run();

	if (rank == 0)
	{
		globalHof.writeGlobalHOF(rank);
	}

	MPI_Finalize();

	return 0;
}