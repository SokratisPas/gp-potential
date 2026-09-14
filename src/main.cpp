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


/* TO DO
- check problem with npt 100k data
- make fitnes function better (MSE, complexity penalty)
- add crossover max depth
- add max depth in general
- make multiprocessing optimizations
- add SUM opperator
*/ 


int main()
{
	// ----------------------------------------------
	// DATA
	int snapshotSize = 15;	// number of .vasp files (each snapshot is one file)

	std::vector<PoscarData> fccData				= storeData("src/data/LJ_Ar/FCC/", snapshotSize);
	std::vector<PoscarData> NVT20kKData			= storeData("src/data/LJ_Ar/NVT20kK/", snapshotSize);
	std::vector<PoscarData> liquidData			= storeData("src/data/LJ_Ar/liquid/", snapshotSize);
	std::vector<PoscarData> liquidNVT100KData	= storeData("src/data/LJ_Ar/liquidNVT100K/", snapshotSize);

	// liquid NPT 100K
	// there is a problem with the number of vasp files

	const std::vector<std::vector<PoscarData>> LJData 
		= { fccData, NVT20kKData, liquidData, liquidNVT100KData };

	const std::vector<std::vector<PairDistanceData>> precompLJData 
		= PrecomputeDistances(LJData);

	// ----------------------------------------------
	// GP parameters
	int popSize				= 10;
	int gens				= 10;
	int initialIndMaxDepth	= 3;
	int tournamentSize		= 3;
	double crossoverProb	= 0.9;
	double mutationProb		= 0.1;
	std::pair constRange	= { -10.0, 10.0 };

	
	// ----------------------------------------------
	GeneticProgram geneticProgram(
		popSize,
		gens,
		initialIndMaxDepth,
		tournamentSize, 
		crossoverProb,
		mutationProb,
		constRange,
		precompLJData,
		fitnessFunction);

	geneticProgram.Run();

	std::cout << "=======================================\n"
		<< "Print all individuals at the end:\n"
		<< "=======================================\n";
	for (auto& genPop : geneticProgram.population)
	{
		genPop.tree.printTree();
		std::cout << "\tFitness : "
			<< genPop.fitness
			<< "\n";
	}



	return 0;
}