#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <functional>
#include <stdexcept>

#include "primitives.h"
#include "treeClass.h"
#include "individual.h"
#include "geneticProgram.h"
#include "POSCARdata.h"
#include "fitness.h"
#include "xyz-parser.h"


/* TO DO
- write fitness function for 2 elements 
- add Ptournament (and Temperature)
- make input/output files 
- python runner
- add more primitives 
- make fitnes function better (MSE, complexity penalty)
- add crossover max depth
- add max depth in general
- make multiprocessing optimizations
*/ 


int main()
{
	// ----------------------------------------------
	// GP parameters
	constexpr double cutoff = 5.0;
	int popSize				= 10;
	int gens				= 10;
	int initialIndMaxDepth	= 3;
	int tournamentSize		= 3;
	double mutationProb		= 0.1;
	std::pair constRange	= { -10.0, 10.0 };
	

	// ----------------------------------------------	
	// DATA
	XYZParser W_Mo_parser(cutoff);
	
	std::vector<Snapshot> W_Mo_data = 
		W_Mo_parser.parse("src/data/W-Mo/trainset.xyz");
	
	// ----------------------------------------------
	// Genetic Program
	GeneticProgram geneticProgram(
		popSize,
		gens,
		initialIndMaxDepth,
		tournamentSize,
		mutationProb,
		constRange,
		W_Mo_data,
		fitnessFun_2elements
	);

	geneticProgram.Run();

	// ----------------------------------------------
	// Print all individuals at the end
	std::cout << "=======================================\n"
		<< "Print all individuals at the end:\n"
		<< "=======================================\n";
	for (auto& genPop : geneticProgram.population)
	{
		for (int caseInd = 0; caseInd < 4; caseInd++)
		{
			std::cout << "Tree: " 
				<< caseInd << "\n";

			genPop.trees[caseInd].printTree();

			std::cout << "\nFitness : "
				<< genPop.fitness
				<< "\n";
		}
		std::cout << "---------------------------------------\n";
	}
	



	return 0;
}