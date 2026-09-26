#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <functional>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <map>

#include "mpi.h"

#include "primitives.h"
#include "treeClass.h"
#include "individual.h"
#include "geneticProgram.h"
#include "POSCARdata.h"
#include "fitness.h"
#include "xyz-parser.h"

namespace {

struct GPConfig {
    int popSize = 100;
    int gens = 50;
    int initialIndMaxDepth = 5;
    int tournamentSize = 3;
    double mutationProb = 0.2;
    double constMin = -10.0;
    double constMax = 10.0;
    int NdataSample = 50;
    int lastNdata = 100;
    double cutoff = 5.0;
    int NlocalInds = 2;
    int NgensToSendInds = 10;
    int NgensToMigration = 20;
    std::string dataPath = "src/data/W-Mo/trainset.xyz";
    std::filesystem::path outputDir = "run/output";
};

std::string trim(const std::string& text)
{
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return "";

    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

std::map<std::string, std::string> parseConfigFile(const std::filesystem::path& path)
{
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("Failed to open config file: " + path.string());

    std::map<std::string, std::string> values;
    std::string line;
    int lineNumber = 0;

    while (std::getline(file, line))
    {
        ++lineNumber;
        const std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#')
            continue;

        const auto pos = trimmed.find('=');
        if (pos == std::string::npos)
            throw std::runtime_error("Invalid config line " + std::to_string(lineNumber) + ": " + line);

        const std::string key = trim(trimmed.substr(0, pos));
        const std::string value = trim(trimmed.substr(pos + 1));
        if (key.empty())
            throw std::runtime_error("Empty config key on line " + std::to_string(lineNumber));

        values[key] = value;
    }

    return values;
}

std::string requireValue(const std::map<std::string, std::string>& values, const std::string& key)
{
    const auto it = values.find(key);
    if (it == values.end())
        throw std::runtime_error("Missing config key: " + key);
    return it->second;
}

int parseInt(const std::map<std::string, std::string>& values, const std::string& key)
{
    std::stringstream ss(requireValue(values, key));
    int value = 0;
    ss >> value;
    if (!ss || !ss.eof())
        throw std::runtime_error("Invalid integer value for " + key);
    return value;
}

double parseDouble(const std::map<std::string, std::string>& values, const std::string& key)
{
    std::stringstream ss(requireValue(values, key));
    double value = 0.0;
    ss >> value;
    if (!ss || !ss.eof())
        throw std::runtime_error("Invalid number value for " + key);
    return value;
}

GPConfig loadConfig(const std::filesystem::path& configPath)
{
    const auto values = parseConfigFile(configPath);
    GPConfig config;

    config.popSize = parseInt(values, "pop_size");
    config.gens = parseInt(values, "gens");
    config.initialIndMaxDepth = parseInt(values, "initial_ind_max_depth");
    config.tournamentSize = parseInt(values, "tournament_size");
    config.mutationProb = parseDouble(values, "mutation_prob");
    config.constMin = parseDouble(values, "const_min");
    config.constMax = parseDouble(values, "const_max");
    config.NdataSample = parseInt(values, "ndata_sample");
    config.lastNdata = parseInt(values, "last_n_data");
    config.cutoff = parseDouble(values, "cutoff");
    config.NlocalInds = parseInt(values, "n_local_inds");
    config.NgensToSendInds = parseInt(values, "ngens_to_send_inds");
    config.NgensToMigration = parseInt(values, "ngens_to_migration");
    config.dataPath = requireValue(values, "data_path");

    if (values.count("output_dir") != 0)
        config.outputDir = values.at("output_dir");

    return config;
}

void printUsage(const char* programName)
{
    std::cout << "Usage: " << programName << " [--config path] [--output-dir path]\n";
    std::cout << "Defaults:\n";
    std::cout << "  --config run/input.txt\n";
    std::cout << "  --output-dir run/output\n";
}

}

int main(int argc, char *argv[])
{
    std::filesystem::path configPath = "input.txt";
    std::filesystem::path outputDir = "output";

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--config")
        {
            if (i + 1 >= argc)
                throw std::runtime_error("--config requires a path");
            configPath = argv[++i];
        }
        else if (arg == "--output-dir")
        {
            if (i + 1 >= argc)
                throw std::runtime_error("--output-dir requires a path");
            outputDir = argv[++i];
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
    if (!config.outputDir.empty())
        outputDir = config.outputDir;

    const double cutoff = config.cutoff;
    const std::pair<double, double> constRange = { config.constMin, config.constMax };

    XYZParser W_Mo_parser(cutoff);
    std::vector<Snapshot> W_Mo_data = W_Mo_parser.parse(config.dataPath);

    if (config.lastNdata <= 0 || config.lastNdata > static_cast<int>(W_Mo_data.size()))
        throw std::runtime_error("last_n_data is out of range for the dataset size");

    // get last data 
    std::vector<Snapshot> reduced_data(W_Mo_data.end() - config.lastNdata, W_Mo_data.end());

    GlobalHOF globalHof;

    int numtasks, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (numtasks < 2)
    {
        throw std::runtime_error("Give process number >= 2!\n");
    }


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
        outputDir
    );

    geneticProgram.Run();

    if (rank == 0)
    {
        globalHof.writeGlobalHOF(rank, outputDir);
    }

    MPI_Finalize();

    return 0;
}
