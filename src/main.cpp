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
#include "input-parser.h"


int main(int argc, char *argv[])
{
    std::filesystem::path configPath = "input.txt"; // gp-potential/input.txt

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--input")
        {
            if (i + 1 >= argc)
                throw std::runtime_error("--input requires a path");
            configPath = argv[++i];
        }
        else if (arg == "--help" || arg == "-h")
        {
            printUsage(argv[0]);
            return 0;
        }
        else
        {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    const GPConfig config = loadConfig(configPath);

    const double cutoff = config.cutoff;
    const std::pair<double, double> constRange = { config.constMin, config.constMax };

    // --------------------------------------
    // DATA
    XYZParser W_Mo_parser(cutoff);
    std::vector<Snapshot> W_Mo_data = W_Mo_parser.parse(config.dataPath);

    if (config.lastNdata <= 0 || config.lastNdata > static_cast<int>(W_Mo_data.size()))
        throw std::runtime_error("last_n_data is out of range for the dataset size");

    // get last data 
    std::vector<Snapshot> reduced_data(W_Mo_data.end() - config.lastNdata, W_Mo_data.end());

    GlobalHOF globalHof;


    // --------------------------------------
    // MPI
    int numtasks, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (numtasks < 2)
    {
        throw std::runtime_error("Give process number >= 2!\n");
    }

    // --------------------------------------
    // Genetic Program
    GeneticProgram geneticProgram(
        config.popSize,
        config.gens,
        config.initialIndMaxDepth,
        config.tournamentSize,
        config.mutationProb,
        constRange,
        reduced_data,
        config.NdataSample,
        fitnessFun_2elements,
        rank,
        numtasks,
        &globalHof,
        config.NlocalInds,
        config.NgensToSendInds,
        config.NgensToMigration,
        config.outputDir
    );

    geneticProgram.Run();

    if (rank == 0)
    {
        globalHof.writeGlobalHOF(rank, config.outputDir);
    }

    MPI_Finalize();

    return 0;
}
