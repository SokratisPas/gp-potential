#pragma once

#include <random>
#include <functional>
#include <utility>
#include <array>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <cstddef>
#include <limits>

#include "mpi.h"

#include "treeClass.h"
#include "individual.h"
#include "primitives.h"
#include "POSCARdata.h"
#include "xyz-parser.h"


// --------------------------------------------------------
class GlobalHOF
{
public:
    struct SerializedEntry
    {
        double fitness = std::numeric_limits<double>::infinity();
        std::array<int, 4> sizes{};
        std::array<int, 4> depths{};
        std::array<std::array<char, 256>, 4> trees{};
    };

    std::vector<Individual> hofIndividuals;    // root-side assembled global HOF
    std::vector<SerializedEntry> gatheredHof;  // root-side gathered local HOFs

    void GatherLocalHOF(int rank,
                       int worldSize,
                       int NlocalInds,
                       const std::vector<Individual>& localHof);

    void writeGlobalHOF(int rank);  // root writes the gathered HOF
};


// --------------------------------------------------------
class GeneticProgram {
public:
    std::vector<Individual> population;

    void Run();
    
    // Constructor
    GeneticProgram(int populationSize,
        int generations, 
        int initialIndMaxDepth,
        int tournamentSize,
        double mutationProbability,
        std::pair<double, double> constRange,
        const std::vector<Snapshot>& data,
        int NdataSample,
        std::function<double(
            const std::array<Tree,4>&, 
            const std::vector<Snapshot>&,
            int dataSample,
            std::mt19937& rng)>
            fitnessFunction,
        int rank,
        int numTasks,
        GlobalHOF* globalHof,
        int NlocalInds,
        int NgensToSendInds 
    )
        :
        populationSize(populationSize),
        generations(generations),
        initialIndMaxDepth(initialIndMaxDepth),
        tournamentSize(tournamentSize),
        mutationProbability(mutationProbability),
        constRange(constRange),
        data(data),
        NdataSample(NdataSample),
        fitnessFunction(fitnessFunction),
        rank(rank),
        numTasks(numTasks),
        globalHof(globalHof),
        NlocalInds(NlocalInds),
        NgensToSendInds(NgensToSendInds), 
        rng(std::random_device{}())   // initialize rng in constructor
        {
        }
            
            
private:
    int populationSize;
    int generations;
    int initialIndMaxDepth;    // depth of initial individuals
    int tournamentSize;
    double mutationProbability;
    int NdataSample;
    std::pair<double, double> constRange;
    const std::vector<Snapshot>& data;    
    int rank;   // ID of each process
    int numTasks;          // total number of MPI ranks
    GlobalHOF* globalHof;   // shared global HOF container
    int NlocalInds;
    int NgensToSendInds;
    std::mt19937 rng;       // each GeneticProgram has unique rng

    // primitive functions (update when adding new primitives in primitives.h)
    // (also update the function "Tree::nodeTypeToString" to print the tree)
    std::vector<const Primitive*> functions =
    {
        &Add,
        &Sub,
        &Mul,
        &Div,
        &Inv6,
        &Inv12
    };
    
    // fitness function
    std::function<double(
        const std::array<Tree,4>&, 
        const std::vector<Snapshot>&,
        int dataSample,
        std::mt19937& rng)>
        fitnessFunction;
        
    void InitializePopulation();
    void EvaluatePopulation();
    void EvaluateIndividual(Individual& individual);

    const Individual& TournamentSelection();

    std::pair<Individual, Individual> Crossover(const Individual& parent1,
                                                const Individual& parent2,
                                                int caseInd);

    Individual Mutate(const Individual& parent, int caseInd);

    double RandomDouble(double min, double max);

    std::shared_ptr<Node> GenerateRandomNode(int depth);

    void ReplaceInd(const Individual& newInd, int index);

    void SortPopulation();
};


// --------------------------------------------------------
// ================================
std::shared_ptr<Node> GeneticProgram::GenerateRandomNode(int depth)
{
    auto node = std::make_shared<Node>();


    // maximum depth reached -> create terminal (var or const)
    if (depth == 0)
    {
        std::uniform_int_distribution<int> terminalDist(0, 1);

        if (terminalDist(rng) == 0)
        {
            node->primitive = &Var;
        }
        else
        {
            node->primitive = &Const;

            std::uniform_real_distribution<double> constDist(constRange.first,
                                                            constRange.second);

            node->constant = constDist(rng);
        }

        return node;
    }


    // choose function or terminal
    std::uniform_int_distribution<int> choice(0, 1);    // 0 -> terminal,  1 -> function


    // chose terminal
    if (choice(rng) == 0)
    {
        std::uniform_int_distribution<int> terminalDist(0, 1);

        if (terminalDist(rng) == 0)
        {
            node->primitive = &Var;
        }
        else
        {
            node->primitive = &Const;

            std::uniform_real_distribution<double> constDist(constRange.first,
                                                            constRange.second);

            node->constant = constDist(rng);
        }

        return node;
    }

    // choose primitive function
    std::uniform_int_distribution<int> funcDist(0, functions.size() - 1);

    node->primitive = functions[funcDist(rng)];


    // create children according to arity
    for (int i = 0; i < node->primitive->arity; i++)
    {
        node->children.push_back(GenerateRandomNode(depth - 1));
    }


    return node;
}

// ================================
void GeneticProgram::InitializePopulation()
{
    population.clear();

    population.reserve(populationSize);


    for (int i = 0; i < populationSize; i++)
    {
        Individual ind;

        // generate random individual with 4 trees 
        for (int j = 0; j < 4; j++)
        {
            Tree tree;

            // generate random node in (1, initialIndMaxDepth)
            std::uniform_int_distribution<int> indDist(1, initialIndMaxDepth);
            tree.root = GenerateRandomNode(indDist(rng));

            ind.trees[j] = tree;
        }

        population.push_back(ind);
    }
}

// ================================
void GeneticProgram::EvaluateIndividual(Individual& individual)
{
    if (individual.evaluated)   // already evaluated fitness
        return;

    individual.fitness = fitnessFunction(individual.trees, data, NdataSample, rng);
    individual.evaluated = true;
}

// ================================
void GeneticProgram::EvaluatePopulation()
{
    for (auto& individual : population)
        EvaluateIndividual(individual);
}

// ================================
const Individual& GeneticProgram::TournamentSelection() 
{
    std::uniform_int_distribution<int> dist(0, population.size() - 1);

    int bestIndex = dist(rng);  // first competitor

    // compare with all other competitors
    for (int i = 1; i < tournamentSize; i++)
    {
        int candidate = dist(rng);  // choose random competitor
        
        // compare competitor with best competitor
        if (population[candidate].fitness < population[bestIndex].fitness)
        {
            bestIndex = candidate;
        }
    }

    return population[bestIndex];
}

// ================================
std::pair<Individual, Individual> GeneticProgram::Crossover(
    const Individual& parent1,
    const Individual& parent2,
    int caseInd)
{
    // clone the parents
    Individual child1 = parent1;
    Individual child2 = parent2;

    child1.trees[caseInd] = parent1.trees[caseInd].Clone();
    child2.trees[caseInd] = parent2.trees[caseInd].Clone();

    // collect all the nodes
    auto nodes1 = child1.trees[caseInd].CollectNodes();
    auto nodes2 = child2.trees[caseInd].CollectNodes();

    // catch empty nodes
    if (nodes1.empty() || nodes2.empty())
        return { child1, child2 };
    

    // choose random crossover points
    std::uniform_int_distribution<size_t> dist1(0, nodes1.size() - 1);
    std::uniform_int_distribution<size_t> dist2(0, nodes2.size() - 1);

    NodeLocation point1 = nodes1[dist1(rng)];
    NodeLocation point2 = nodes2[dist2(rng)];

    // clone the selected subtrees
    auto subtree1 = child1.trees[caseInd].CloneNode(point1.node);
    auto subtree2 = child2.trees[caseInd].CloneNode(point2.node);

    // replace subtrees for children
    // 
    // child 1
    if (point1.parent == nullptr)   // we chose the root
    {
        child1.trees[caseInd].root = subtree2;
    }
    else                            // we chose some other node
    {
        point1.parent->children[point1.childIndex] = subtree2;
    }
    // child 2
    if (point2.parent == nullptr)
    {
        child2.trees[caseInd].root = subtree1;
    }
    else
    {
        point2.parent->children[point2.childIndex] = subtree1;
    }

    // mark children as not evaluated
    child1.evaluated = false;
    child2.evaluated = false;

    child1.fitness = std::numeric_limits<double>::infinity();
    child2.fitness = std::numeric_limits<double>::infinity();

    child1.energyLoss = 0.0;
    child2.energyLoss = 0.0;

    return { child1, child2 };
}

// ================================
Individual GeneticProgram::Mutate(const Individual& parent, int caseInd)
{
    // Clone the parent
    Individual child = parent;
    child.trees[caseInd] = parent.trees[caseInd].Clone();

    // Guard against empty/default trees that can exist before a real parent is assigned.
    // (probably solved)
    if (!child.trees[caseInd].root)
    {
        child.trees[caseInd].root = GenerateRandomNode(3);
        child.evaluated = false;
        child.fitness = std::numeric_limits<double>::infinity();
        child.energyLoss = 0.0;
        return child;
    }

    // Collect all nodes
    auto nodes = child.trees[caseInd].CollectNodes();

    if (nodes.empty())
        return child;

    // Choose a random mutation point
    std::uniform_int_distribution<size_t> dist(0, nodes.size() - 1);
    NodeLocation point = nodes[dist(rng)];

    // Generate a new random subtree
    auto newSubtree = GenerateRandomNode(3);

    // Replace the selected subtree
    if (point.parent == nullptr)
    {
        // Mutating the root
        child.trees[caseInd].root = newSubtree;
    }
    else
    {
        point.parent->children[point.childIndex] = newSubtree;
    }

    // Reset evaluation
    child.evaluated = false;
    child.fitness = std::numeric_limits<double>::infinity();
    child.energyLoss = 0.0;

    return child;
}

// ================================
double GeneticProgram::RandomDouble(double min, double max)
{
    std::uniform_real_distribution<double> dist(min, max);
    return dist(rng);
}

// ================================
void GeneticProgram::ReplaceInd(const Individual& newInd, int index)
{
    if (index < 0 || index >= static_cast<int>(population.size()))
        throw std::out_of_range("Invalid population index");

    population[index] = newInd;
}

// ================================
void GeneticProgram::SortPopulation()
{
    // Sort the population based on fitness
    // the best ind (lower fittnes) is first
    std::sort(population.begin(), population.end(),
              [](const Individual& a, const Individual& b)
              {
                  return a.fitness < b.fitness;
              });
}

// ================================
void GeneticProgram::Run()
{
    // create output dir (output/processID/)
    const std::filesystem::path outputDir = std::filesystem::current_path() / "output" / std::to_string(rank);
    std::filesystem::create_directories(outputDir);

    // create stats file (gp_statistics_processID.txt)
    const std::filesystem::path statsFilePath = 
        outputDir / ("gp_statistics_" + std::to_string(rank) + ".txt");
    std::ofstream statsFile(statsFilePath);
    if (!statsFile)
    {
        throw std::runtime_error("Failed to open output file: " + statsFilePath.string());
    }
    statsFile << "GENERATION"
    << ",Best fitness"
    << ",Max Size[0],Max Size[1],Max Size[2],Max Size[3]"
    << ",Min Size[0],Min Size[1],Min Size[2],Min Size[3]"
    << "\n";

    // Create the initial population
    InitializePopulation();

    // Evaluate intial population
    EvaluatePopulation();

    // sort population
    SortPopulation();

    // find the current worst indices in the sorted population
    std::array<int, 2> worstIndices = {
        (populationSize - 1),
        (populationSize - 2)
    };

    // --------------------------
    // Evolution loop
    for (int generation = 0; generation < generations; generation++)
    {
        // Select first parent 
        const Individual& parent1 = TournamentSelection();
        
        //          Mutation
        if (RandomDouble(0.0, 1.0) < mutationProbability)
        {
            Individual child1 = parent1;

            for (int caseInd = 0; caseInd < 4; caseInd++)
            {
                child1 = Mutate(child1, caseInd);
            }

            EvaluateIndividual(child1);
            ReplaceInd(child1, worstIndices[0]);
        }
        else    //  Crossover
        {
            // select second parent
            // (for crossover we need 2 individuals)
            const Individual& parent2 = TournamentSelection();
            std::pair<Individual, Individual> children{ parent1, parent2 };
            
            for (int caseInd = 0; caseInd < 4; caseInd++)
            {
                children = Crossover(children.first, children.second, caseInd);
            }

            EvaluateIndividual(children.first);
            EvaluateIndividual(children.second);
            ReplaceInd(children.first, worstIndices[0]);
            ReplaceInd(children.second, worstIndices[1]);
        }

        // sort population
        SortPopulation();

        // update the worst individuals
        worstIndices = {
            (populationSize - 1),
            (populationSize - 2)
        };

        // --------------------------
        // Write statistics to local file
        double bestFitness = population[0].fitness; // lowest fitness
        std::array<int, 4> maxSize = {0, 0, 0, 0};
        std::array<int, 4> minSize = {10000, 10000, 10000, 10000};

        for (const auto& individual : population)
        {
            for (int caseInd = 0; caseInd < 4; caseInd++)
            {
                int indSize = individual.trees[caseInd].Size();
                maxSize[caseInd] = std::max(maxSize[caseInd], indSize);
                minSize[caseInd] = std::min(minSize[caseInd], indSize);
            }
        }

        statsFile << generation << ","
            << bestFitness << ","
            << maxSize[0] << "," << maxSize[1] << "," << maxSize[2] << "," << maxSize[3]
            << ","
            << minSize[0] << "," << minSize[1] << "," << minSize[2] << "," << minSize[3]
            << "\n";


        // --------------------------
        // send best individuals to global hof every NgensToSendInds
        // also always send the final generation 
        if (globalHof != nullptr &&
            (generation == generations - 1 || generation % NgensToSendInds == 0))
        {
            std::vector<Individual> localHof;
            localHof.reserve(static_cast<size_t>(NlocalInds));
            for (int i = 0; i < NlocalInds; ++i)
            {
                localHof.push_back(population[i]);
            }

            globalHof->GatherLocalHOF(rank, numTasks, NlocalInds, localHof);
        }
    }

    statsFile.close();


    // --------------------------
    // create local hof file
    const std::filesystem::path hofFilePath = 
        outputDir / ("gp_hof_" + std::to_string(rank) + ".txt");
    std::ofstream hofFile(hofFilePath);
    if (!hofFile)
    {
        throw std::runtime_error("Failed to open output file: " + hofFilePath.string());
    }

    hofFile << "=======================================\n"
        << "HOF 10 best individuals:\n"
        << "=======================================\n";

	for (int i = 0; i < 10; i++)
	{
		Individual indiv = population[i];

		hofFile << "==== Individual: "
			<< i + 1
			<< " ===="
			<<"\nFitness : "
			<< indiv.fitness
			<< "\n";

		for (int caseInd = 0; caseInd < 4; caseInd++)
		{
			hofFile << "Tree: " 
				<< caseInd 
				<< " (Size: "
				<< indiv.trees[caseInd].Size()
				<< ")\n"
                << indiv.trees[caseInd].convertToStr()
                << "\n";
		}
		hofFile << "\n---------------------------------------\n";
	}
    hofFile.close();
}


// ================================
void GlobalHOF::GatherLocalHOF(
    int rank,
    int worldSize,
    int NlocalInds,
    const std::vector<Individual>& localHof)
{
    std::vector<SerializedEntry> localEntries(static_cast<size_t>(NlocalInds));

    for (int i = 0; i < NlocalInds; ++i)
    {
        if (i < static_cast<int>(localHof.size()))
        {
            const auto& ind = localHof[static_cast<size_t>(i)];
            localEntries[static_cast<size_t>(i)].fitness = ind.fitness;

            for (int j = 0; j < 4; ++j)
            {
                std::string treeStr = ind.trees[j].convertToStr();
                auto& treeBuffer = localEntries[static_cast<size_t>(i)].trees[j];
                treeBuffer.fill('\0');
                std::snprintf(
                    treeBuffer.data(),
                    treeBuffer.size(),
                    "%s",
                    treeStr.c_str());
                localEntries[static_cast<size_t>(i)].sizes[j] = ind.trees[j].Size();
                localEntries[static_cast<size_t>(i)].depths[j] = ind.trees[j].Depth();
            }
        }
    }

    MPI_Datatype entryType;
    int blockLengths[] = {1, 4, 4, 4 * 256};
    MPI_Datatype types[] = {MPI_DOUBLE, MPI_INT, MPI_INT, MPI_CHAR};
    MPI_Aint offsets[4];
    offsets[0] = offsetof(SerializedEntry, fitness);
    offsets[1] = offsetof(SerializedEntry, sizes);
    offsets[2] = offsetof(SerializedEntry, depths);
    offsets[3] = offsetof(SerializedEntry, trees);

    // MPI_Type_create_struct(N_diff_blocks/types, N_elementsInBlock[],
    //                  memoryDisplacementOfBlock[], typeOfBlock[], &output)
    MPI_Type_create_struct(4, blockLengths, offsets, types, &entryType);
    MPI_Type_commit(&entryType);

    std::vector<SerializedEntry> gatheredBuffer(static_cast<size_t>(worldSize) * static_cast<size_t>(NlocalInds));

    if (rank == 0)
    {
        // MPI_Gather(&sendbuf,sendcnt,sendtype,&recvbuf,recvcount,recvtype,root,comm)
        MPI_Gather(
            localEntries.data(), NlocalInds,
            entryType, gatheredBuffer.data(),
            NlocalInds, entryType,
            0, MPI_COMM_WORLD
        );

        
        // rank 0 also updates the hofIndividuals vector
        gatheredHof = gatheredBuffer;
        hofIndividuals.clear();
        hofIndividuals.reserve(gatheredHof.size());

        for (const auto& entry : gatheredHof)
        {
            Individual ind;
            ind.fitness = entry.fitness;
            ind.evaluated = true;
            ind.energyLoss = 0.0;
            for (int j = 0; j < 4; ++j)
            {
                ind.trees[j].root = std::make_shared<Node>();
                ind.trees[j].root->primitive = &Const;
                ind.trees[j].root->constant = 0.0;
            }
            hofIndividuals.push_back(ind);
        }
    }
    else
    {
        // MPI_Gather(&sendbuf,sendcnt,sendtype,&recvbuf,recvcount,recvtype,root,comm)
        MPI_Gather(
            localEntries.data(), NlocalInds,
            entryType, nullptr, 
            NlocalInds, entryType,
            0, MPI_COMM_WORLD
        );
    }

    MPI_Type_free(&entryType);
}

// ================================
void GlobalHOF::writeGlobalHOF(int rank)
{
    // only rank 0 writes the globalHof file
    if (rank != 0)
        return;

    const std::filesystem::path outputDirMaster = std::filesystem::current_path() / "output";
    std::filesystem::create_directories(outputDirMaster);

    const std::filesystem::path hofFilePathMaster = outputDirMaster / "global_hof.txt";
    std::ofstream hofFileGlobal(hofFilePathMaster, std::ios::trunc);
    if (!hofFileGlobal)
    {
        throw std::runtime_error("Failed to open output file: " + hofFilePathMaster.string());
    }

    for (const auto& entry : gatheredHof)
    {
        hofFileGlobal << "=====================\n"
            << "Fitness : " << entry.fitness << "\n"
            << "Size : " << entry.sizes[0] << ", " << entry.sizes[1] << ", "
            << entry.sizes[2] << ", " << entry.sizes[3] << "\n"
            << "Depth : " << entry.depths[0] << ", " << entry.depths[1] << ", "
            << entry.depths[2] << ", " << entry.depths[3] << "\n"
            << "Potentials : \n";

        for (int i = 0; i < 4; ++i)
        {
            hofFileGlobal << i + 1 << ") " << entry.trees[i].data() << "\n";
        }
    }

    hofFileGlobal.close();
}