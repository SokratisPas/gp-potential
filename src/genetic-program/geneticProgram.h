#pragma once

#include <random>
#include <functional>
#include <utility>
#include <array>
#include <vector>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <cctype>
#include <cmath>
#include <limits>
#include <string>
#include <stdexcept>

#include "mpi.h"

#include "treeClass.h"
#include "individual.h"
#include "primitives.h"
#include "POSCARdata.h"
#include "xyz-parser.h"
#include "globalHoF.h"



// --------------------------------------------------------
class GeneticProgram {
public:
    std::vector<Individual> population;

    void Run();
    
    // Constructor
    explicit GeneticProgram(
        int populationSize,
        int generations, 
        int initialIndMaxDepth,
        int tournamentSize,
        double mutationProbability,
        double tournSelProbability,
        std::pair<double, double> constRange,
        const std::vector<Snapshot>& data,
        int NdataSample,
        std::function<double(
            const std::array<Tree,4>&, 
            const std::vector<Snapshot>&,
            int dataSample,
            std::mt19937& rng,
            int rank,
            double sizePenaltyFactor
        )> fitnessFunction,
        int rank,
        int numTasks,
        GlobalHOF* globalHof,
        int NlocalInds,
        int NgensToSendInds,
        int NgensToMigration,
        int Nmigrants,
        int NgensToUpd_LHOF,
        double sizePenaltyFactor,
        const std::filesystem::path& outputDir = "output"
    )
        :
        populationSize(populationSize),
        generations(generations),
        initialIndMaxDepth(initialIndMaxDepth),
        tournamentSize(tournamentSize),
        mutationProbability(mutationProbability),
        tournSelProbability(tournSelProbability),
        constRange(constRange),
        data(data),
        NdataSample(NdataSample),
        fitnessFunction(fitnessFunction),
        rank(rank),
        numTasks(numTasks),
        globalHof(globalHof),
        NlocalInds(NlocalInds),
        NgensToSendInds(NgensToSendInds), 
        NgensToMigration(NgensToMigration),
        Nmigrants(Nmigrants),
        NgensToUpd_LHOF(NgensToUpd_LHOF),
        sizePenaltyFactor(sizePenaltyFactor),
        outputDirectory(outputDir),
        rng(std::random_device{}())   // initialize rng in constructor
        {
        }

            
private:
    int populationSize;
    int generations;
    int initialIndMaxDepth;    // depth of initial individuals
    int tournamentSize;
    double mutationProbability;
    double tournSelProbability;
    int NdataSample;
    std::pair<double, double> constRange;
    const std::vector<Snapshot>& data;
    int rank;   // ID of each process
    int numTasks;          // total number of MPI ranks
    GlobalHOF* globalHof;   // shared global HOF container
    int NlocalInds;
    int NgensToSendInds;
    int NgensToMigration;
    int Nmigrants;
    int NgensToUpd_LHOF;
    double sizePenaltyFactor;    
    std::filesystem::path outputDirectory;
    std::mt19937 rng;       // each GeneticProgram has unique rng
    const int maxDepth = 17;    // Koza (1992)
    const int NtestInds = 5;   // number of test individuals for MutateConst, we keep the best

    // primitive functions (update when adding new primitives in primitives.h)
    // (also update the function "Tree::nodeTypeToString" to print the tree)
    std::vector<const Primitive*> functions =
    {
        &Add,
        &Sub,
        &Mul,
        &Div,
        &Pow
    };
    
    // fitness function
    std::function<double(
        const std::array<Tree,4>&, 
        const std::vector<Snapshot>&,
        int dataSample,
        std::mt19937& rng,
        int rank,
        double sizePenaltyFactor
    )> fitnessFunction;
    
    template <typename IntType>
    IntType RandomNumber(IntType min, IntType max);
    double RandomNumber(double min, double max);

    void InitializePopulation();
    void EvaluatePopulation();
    void EvaluateIndividual(Individual& individual);
    const Individual& TournamentSelection();
    std::pair<Individual, Individual> Crossover(
        const Individual& parent1,
        const Individual& parent2,
        int caseInd
    );
    Individual MutateSubtree(const Individual& parent, int caseInd);
    Individual MutatePoint(const Individual& parent, int caseInd);
    Individual MutateConst(const Individual& parent, int caseInd, int NtestInds);
    std::shared_ptr<Node> GenerateRandomNode(int depth);
    void ReplaceInd(const Individual& newInd, int index);
    void SortPopulation();
    bool IsWithinMaxDepth(const Individual& individual) const;
    void WriteLocalHOF(const std::filesystem::path& outputDir) const;
    Individual DeserializeGlobalEntry(const GlobalHOF::SerializedEntry& entry) const;
    std::shared_ptr<Node> ParseTreeExpression(
        const std::string& expression,
        size_t& pos
    ) const;
    std::shared_ptr<Node> ParseNode(const std::string& expression, size_t& pos) const;
    static std::string Trim(const std::string& text);
    void performMigration();
    void sendToGlobalHOF();
    void write_gp_statistics(
        const std::filesystem::path& filePath,
        const int generation
    );
};

// ================================
template <typename IntType>
IntType GeneticProgram::RandomNumber(IntType min, IntType max)
{
    std::uniform_int_distribution<IntType> dist(min, max);
    return dist(rng);
}

// ================================
double GeneticProgram::RandomNumber(double min, double max)
{
    std::uniform_real_distribution<double> dist(min, max);
    return dist(rng);
}

// ================================
std::string GeneticProgram::Trim(const std::string& text)
{
    size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])))
        ++begin;

    size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])))
        --end;

        return text.substr(begin, end - begin);
}

// ================================
std::shared_ptr<Node> GeneticProgram::ParseNode(const std::string& expression, size_t& pos) const
{
    const auto skipSpaces = [&]() {
        while (pos < expression.size() && std::isspace(static_cast<unsigned char>(expression[pos])))
            ++pos;
    };

    skipSpaces();

    if (pos >= expression.size())
        throw std::runtime_error("Unexpected end of tree expression");

    auto node = std::make_shared<Node>();

    if (expression[pos] == 'r')
    {
        node->primitive = &Var;
        ++pos;
        return node;
    }

    if (expression[pos] == '-' || std::isdigit(static_cast<unsigned char>(expression[pos])))
    {
        size_t start = pos;
        if (expression[pos] == '-')
            ++pos;
        while (pos < expression.size() && (std::isdigit(static_cast<unsigned char>(expression[pos])) || expression[pos] == '.'))
            ++pos;
        if (pos < expression.size() && expression[pos] == 'e')
        {
            ++pos;
            if (pos < expression.size() && (expression[pos] == '-' || expression[pos] == '+'))
                ++pos;
            while (pos < expression.size() && std::isdigit(static_cast<unsigned char>(expression[pos])))
                ++pos;
        }

        const std::string constantString = expression.substr(start, pos - start);
        node->primitive = &Const;
        node->constant = std::stod(constantString);
        return node;
    }

    size_t start = pos;
    while (pos < expression.size() && std::isalnum(static_cast<unsigned char>(expression[pos])))
        ++pos;

    const std::string funcName = expression.substr(start, pos - start);
    const Primitive* primitive = PrimitiveFromName(funcName);
    if (primitive == nullptr)
        throw std::runtime_error("Unknown tree primitive in serialized HOF entry: " + funcName);

    node->primitive = primitive;
    skipSpaces();
    if (pos >= expression.size() || expression[pos] != '(')
        throw std::runtime_error("Expected '(' after primitive name: " + funcName);
    ++pos;

    while (true)
    {
        skipSpaces();
        if (pos >= expression.size())
            throw std::runtime_error("Unexpected end of function arguments");

        if (expression[pos] == ')')
        {
            ++pos;
            break;
        }

        node->children.push_back(ParseNode(expression, pos));
        skipSpaces();

        if (pos < expression.size() && expression[pos] == ',')
        {
            ++pos;
            continue;
        }

        if (pos < expression.size() && expression[pos] == ')')
        {
            ++pos;
            break;
        }

        throw std::runtime_error("Expected ',' or ')' in serialized tree expression");
    }

    return node;
}

// ================================
std::shared_ptr<Node> GeneticProgram::ParseTreeExpression(
    const std::string& expression,
    size_t& pos
) const
{
    pos = 0;
    const std::string trimmed = Trim(expression);
    if (trimmed.empty())
        return nullptr;

    const auto root = ParseNode(trimmed, pos);
    while (pos < trimmed.size() && std::isspace(static_cast<unsigned char>(trimmed[pos])))
        ++pos;
    return root;
}

// ================================
Individual GeneticProgram::DeserializeGlobalEntry(const GlobalHOF::SerializedEntry& entry) const
{
    Individual individual;
    individual.fitness = entry.fitness;
    individual.evaluated = true;
    individual.energyLoss = 0.0;

    for (int j = 0; j < 4; ++j)
    {
        std::string treeText = Trim(entry.trees[j].data());
        if (treeText.empty())
        {
            individual.trees[j].root = std::make_shared<Node>();
            individual.trees[j].root->primitive = &Const;
            individual.trees[j].root->constant = 0.0;
            continue;
        }

        size_t pos = 0;
        individual.trees[j].root = ParseTreeExpression(treeText, pos);
        if (!individual.trees[j].root)
        {
            individual.trees[j].root = std::make_shared<Node>();
            individual.trees[j].root->primitive = &Const;
            individual.trees[j].root->constant = 0.0;
        }
    }

    return individual;
}

// ================================
std::shared_ptr<Node> GeneticProgram::GenerateRandomNode(int depth)
{
    auto node = std::make_shared<Node>();

    // maximum depth reached -> create terminal (var or const)
    if (depth == 0)
    {
        if (RandomNumber(0, 1) == 0)
        {
            node->primitive = &Var;
        }
        else
        {
            node->primitive = &Const;

            node->constant = RandomNumber(constRange.first, constRange.second);
        }

        return node;
    }

    // choose function or terminal
    // chose terminal
    if (RandomNumber(0, 1) == 0)
    {
        if (RandomNumber(0, 1) == 0)
        {
            node->primitive = &Var;
        }
        else
        {
            node->primitive = &Const;

            node->constant = RandomNumber(constRange.first, constRange.second);
        }

        return node;
    }

    // choose primitive function
    node->primitive = functions[RandomNumber(size_t{0}, functions.size() - 1)];

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

            // Generate a random tree with depth in [1, initialIndMaxDepth]
            tree.root = GenerateRandomNode(
                RandomNumber(1, std::min(initialIndMaxDepth, maxDepth - 1))
            );

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

    if (!IsWithinMaxDepth(individual))
        throw std::runtime_error("Individual contains a tree exceeding the maximum depth.");

    individual.fitness = fitnessFunction(
        individual.trees,
        data,
        NdataSample,
        rng,
        rank,
        sizePenaltyFactor
    );
    individual.evaluated = true;
}

// ================================
bool GeneticProgram::IsWithinMaxDepth(const Individual& individual) const
{
    for (const auto& tree : individual.trees)
    {
        if (tree.Depth() > maxDepth)
            return false;
    }

    return true;
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
    // chose random contenstants indx for tourn sel
    std::vector<size_t> contestants;
    contestants.reserve(static_cast<size_t>(tournamentSize));

    for (int i = 0; i < tournamentSize; ++i)
    {
        contestants.push_back(RandomNumber(size_t{0}, population.size() - 1));
    }
    
    std::sort(contestants.begin(), contestants.end(),
        [this](size_t a, size_t b)
        {
            return population[a].fitness < population[b].fitness;
        }
    );

    // select fittest contestant with prob tournSelProbability
    for (int i = 0; i < tournamentSize - 1; ++i)
    {
        double pTournament = RandomNumber(0.0, 1.0);

        if ( pTournament < tournSelProbability)
        {
            return population[contestants[i]];
        }
    }

    // if only one contestant remains send him
    return population[contestants[tournamentSize - 1]];
}

// ================================
std::pair<Individual, Individual> GeneticProgram::Crossover(
    const Individual& parent1,
    const Individual& parent2,
    int caseInd
)
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
    NodeLocation point1 = nodes1[RandomNumber(size_t{0}, nodes1.size() - 1)];
    NodeLocation point2 = nodes2[RandomNumber(size_t{0}, nodes2.size() - 1)];

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

    // catch trees with large depth
    if (child1.trees[caseInd].Depth() > maxDepth)
        child1.trees[caseInd] = parent1.trees[caseInd].Clone();
    if (child2.trees[caseInd].Depth() > maxDepth)
        child2.trees[caseInd] = parent2.trees[caseInd].Clone();

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
Individual GeneticProgram::MutateSubtree(const Individual& parent, int caseInd)
{
    // Clone the parent
    Individual child = parent;
    child.trees[caseInd] = parent.trees[caseInd].Clone();

    // Guard against empty/default trees that can exist before a real parent is assigned.
    // (probably solved)
    if (!child.trees[caseInd].root)
    {
        child.trees[caseInd].root = GenerateRandomNode(RandomNumber(1, 3));
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
    NodeLocation point = nodes[RandomNumber(size_t{0}, nodes.size() - 1)];

    // Generate a new random subtree with depth in [1, 3]
    auto newSubtree = GenerateRandomNode(RandomNumber(1, 3));

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

    // Koza (1992) reject individuals with huge depth
    if (child.trees[caseInd].Depth() > maxDepth)
    {
        return parent;
    }

    // Reset evaluation
    child.evaluated = false;
    child.fitness = std::numeric_limits<double>::infinity();
    child.energyLoss = 0.0;

    return child;
}

// ================================
Individual GeneticProgram::MutatePoint(const Individual& parent, int caseInd)
{
    Individual child = parent;
    child.trees[caseInd] = parent.trees[caseInd].Clone();

    if (!child.trees[caseInd].root)
        return child;

    auto nodes = child.trees[caseInd].CollectNodes();
    if (nodes.empty())
        return child;

    const auto& node = nodes[RandomNumber(size_t{0}, nodes.size() - 1)].node;
    if (!node || !node->primitive)
        return child;

    bool mutated = false;
    if (node->primitive->arity > 0)
    {
        std::vector<const Primitive*> alternatives;
        for (const Primitive* function : functions)
        {
            if (function->arity == node->primitive->arity &&
                function->type != node->primitive->type)
            {
                alternatives.push_back(function);
            }
        }

        if (!alternatives.empty())
        {
            node->primitive = alternatives[
                RandomNumber(size_t{0}, alternatives.size() - 1)
            ];
            mutated = true;
        }
    }
    else if (node->primitive->type == NodeType::Const)
    {
        double randomConst = RandomNumber(constRange.first, constRange.second);
        node->constant = randomConst;

        mutated = true;
    }

    if (mutated)
    {
        child.evaluated = false;
        child.fitness = std::numeric_limits<double>::infinity();
        child.energyLoss = 0.0;
    }

    return child;
}

// ================================
Individual GeneticProgram::MutateConst(const Individual& parent, int caseInd, int NtestInds)
{
    Individual bestCandidate = parent;
    bestCandidate.trees[caseInd] = parent.trees[caseInd].Clone();
    bestCandidate.evaluated = false;
    bestCandidate.fitness = std::numeric_limits<double>::infinity();
    bestCandidate.energyLoss = 0.0;

    if (!bestCandidate.trees[caseInd].root)
        return bestCandidate;
    
    // collect all constant nodes
    std::vector<std::shared_ptr<Node>> constNodes;
    const auto collected = bestCandidate.trees[caseInd].CollectNodes();
    for (const auto& location : collected)
    {
        if (location.node && location.node->primitive && location.node->primitive->type == NodeType::Const)
            constNodes.push_back(location.node);
    }

    if (constNodes.empty() || NtestInds <= 0)
        return bestCandidate;

    const int maxMutatedConstants = std::max(1, static_cast<int>(constNodes.size() / 3 + 1));

    for (int test = 0; test < NtestInds; ++test)
    {
        Individual candidate = parent;
        candidate.trees[caseInd] = parent.trees[caseInd].Clone();

        auto candidateNodes = candidate.trees[caseInd].CollectNodes();
        std::vector<std::shared_ptr<Node>> candidateConstNodes;
        for (const auto& location : candidateNodes)
        {
            if (location.node && location.node->primitive &&
                location.node->primitive->type == NodeType::Const)
            {
                candidateConstNodes.push_back(location.node);
            }
        }

        if (candidateConstNodes.empty())
            continue;

        std::shuffle(candidateConstNodes.begin(), candidateConstNodes.end(), rng);

        const int numberToMutate = RandomNumber(
            1,
            std::min<int>(maxMutatedConstants, static_cast<int>(candidateConstNodes.size()))
        );
        for (int i = 0; i < numberToMutate; ++i)
        {
            candidateConstNodes[i]->constant = RandomNumber(constRange.first, constRange.second);
        }

        candidate.evaluated = false;
        candidate.fitness = std::numeric_limits<double>::infinity();
        candidate.energyLoss = 0.0;

        EvaluateIndividual(candidate);

        if (!std::isfinite(bestCandidate.fitness) || candidate.fitness < bestCandidate.fitness)
            bestCandidate = candidate;
    }

    return bestCandidate;
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
              }
    );
}

// ================================
void GeneticProgram::WriteLocalHOF(const std::filesystem::path& outputDir) const
{
    const std::filesystem::path hofFilePath =
        outputDir / ("local_hof_" + std::to_string(rank) + ".txt");
    std::ofstream hofFile(hofFilePath);
    if (!hofFile)
    {
        throw std::runtime_error("Failed to open output file: " + hofFilePath.string());
    }

    hofFile << "=======================================\n"
        << "Local HOF | rank: " << rank << "\n"
        << "=======================================\n";

    const size_t individualsToWrite = std::min<size_t>(10, population.size());
    for (size_t i = 0; i < individualsToWrite; ++i)
    {
        const Individual& indiv = population[i];

        hofFile << "==== Individual: "
            << i + 1
            << " ===="
            << "\nFitness : "
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
void GeneticProgram::performMigration()
{
    std::vector<int> individualIndices(static_cast<size_t>(populationSize));
    std::iota(individualIndices.begin(), individualIndices.end(), 0);
    std::shuffle(individualIndices.begin(), individualIndices.end(), rng);
    
    std::vector<GlobalHOF::SerializedEntry> outgoingEntries(
        static_cast<size_t>(Nmigrants)
    );
    
    for (int i = 0; i < Nmigrants; ++i)
    {
        const Individual& individual =
            population[static_cast<size_t>(individualIndices[static_cast<size_t>(i)])];
        auto& entry = outgoingEntries[static_cast<size_t>(i)];
        entry.fitness = individual.fitness;
        for (size_t treeIndex = 0; treeIndex < individual.trees.size(); ++treeIndex)
        {
            entry.trees[treeIndex] = individual.trees[treeIndex].convertToStr();
            entry.sizes[treeIndex] = individual.trees[treeIndex].Size();
            entry.depths[treeIndex] = individual.trees[treeIndex].Depth();
        }
    }
    
    const std::vector<char> sendBuffer =
        GlobalHOF::SerializeLocalEntries(outgoingEntries);
    const int sendSize = static_cast<int>(sendBuffer.size());
    int receiveSize = 0;
    const int nextRank = (rank + 1) % numTasks;
    const int previousRank = (rank + numTasks - 1) % numTasks;
    MPI_Sendrecv(
        &sendSize, 
        1, 
        MPI_INT, 
        nextRank, 
        0,
        &receiveSize, 
        1, 
        MPI_INT, 
        previousRank, 
        0,
        MPI_COMM_WORLD, 
        MPI_STATUS_IGNORE
    );
    
    std::vector<char> receiveBuffer(static_cast<size_t>(receiveSize));
    
    // (to avoid deadlocks we use MPI_Sendrecv)
    MPI_Sendrecv(
        sendBuffer.data(),      // send buffer
        sendSize,               // send count
        MPI_CHAR,               // send type
        nextRank,               // destination rank
        1,                      // send tag
        receiveBuffer.data(),   // receive buffer
        receiveSize,            // receive count
        MPI_CHAR,               // receive type
        previousRank,           // source rank
        1,                      // receive tag
        MPI_COMM_WORLD,         // communicator
        MPI_STATUS_IGNORE       // status
    );
    
    const auto incomingEntries =
        GlobalHOF::DeserializeLocalEntries(receiveBuffer, static_cast<size_t>(Nmigrants));
    std::shuffle(individualIndices.begin(), individualIndices.end(), rng);
    for (int i = 0; i < Nmigrants; ++i)
    {
        const int replaceIndex = individualIndices[static_cast<size_t>(i)];
        Individual incoming =
            DeserializeGlobalEntry(incomingEntries[static_cast<size_t>(i)]);
        if (IsWithinMaxDepth(incoming))
            population[static_cast<size_t>(replaceIndex)] = std::move(incoming);
    }
    
    SortPopulation();
}

// ================================
void GeneticProgram::sendToGlobalHOF()
{
    std::vector<Individual> localHof;
    localHof.reserve(static_cast<size_t>(NlocalInds));
    
    for (const auto& individual : population)
    {
        if (localHof.size() >= static_cast<size_t>(NlocalInds))
            break;
    
        if (std::isfinite(individual.fitness))
            localHof.push_back(individual);
    }
    
    globalHof->GatherLocalHOF(
        rank,
        numTasks,
        static_cast<int>(localHof.size()),
        localHof
    );
}

// ================================
void GeneticProgram::write_gp_statistics(
    const std::filesystem::path& filePath,
    const int generation
)
{
    std::ofstream statsFile(filePath, std::ios::app);   // open file in append mode (file is not overwritten)
    if (!statsFile)
    {
        throw std::runtime_error("Failed to open output file: " + filePath.string());
    }
    
    double bestFitness = population[0].fitness; // lowest fitness
    std::array<int, 4> maxSize = {0, 0, 0, 0};
    std::array<int, 4> minSize = {10000, 10000, 10000, 10000};
    std::array<int, 4> sumSize = {0, 0, 0, 0};
    
    for (const auto& individual : population)
    {
        for (int caseInd = 0; caseInd < 4; caseInd++)
        {
            int indSize = individual.trees[caseInd].Size();

            sumSize[caseInd] += indSize;

            maxSize[caseInd] = std::max(maxSize[caseInd], indSize);
            minSize[caseInd] = std::min(minSize[caseInd], indSize);
        }
    }

    statsFile << generation + 1 << ","
        << bestFitness << ","
        << maxSize[0] << "," << maxSize[1] << "," << maxSize[2] << "," << maxSize[3]
        << ","
        << minSize[0] << "," << minSize[1] << "," << minSize[2] << "," << minSize[3]
        <<","
        << sumSize[0] / static_cast<double>(populationSize) << "," << sumSize[1] / static_cast<double>(populationSize)
        << "," 
        << sumSize[2] / static_cast<double>(populationSize) << "," << sumSize[3] / static_cast<double>(populationSize)
        << ","
        << population[0].trees[0].Size() << "," << population[0].trees[1].Size() << ","
        << population[0].trees[2].Size() << "," << population[0].trees[3].Size()
        << "\n";
    
    statsFile.close();
}

// ================================
void GeneticProgram::Run()
{
    if (populationSize <= 0)
    {
        throw std::invalid_argument("populationSize must be greater than zero");
    }

    if (Nmigrants <= 0 || Nmigrants > populationSize)
    {
        throw std::invalid_argument("Nmigrants must be between 1 and populationSize");
    }

    if (NgensToMigration <= 0)
    {
        throw std::invalid_argument("NgensToMigration must be greater than zero");
    }

    if (NgensToUpd_LHOF <= 0)
    {
        throw std::invalid_argument("NgensToUpd_LHOF must be greater than zero");
    }

    if (initialIndMaxDepth <= 0)
    {
        throw std::invalid_argument("initialIndMaxDepth must be greater than zero");
    }

    if (tournamentSize <= 0)
    {
        throw std::invalid_argument("tournamentSize must be greater than zero");
    }

    if (!std::isfinite(tournSelProbability) ||
        tournSelProbability < 0.0 || tournSelProbability > 1.0)
    {
        throw std::invalid_argument("tournSelProbability must be between 0 and 1");
    }

    // create output dir (configured output dir / processID/)
    const std::filesystem::path outputDir = std::filesystem::absolute(outputDirectory) / std::to_string(rank);
    std::filesystem::create_directories(outputDir);

    // create stats file (local_statistics_processID.txt)
    const std::filesystem::path statsFilePath = 
        outputDir / ("local_statistics_" + std::to_string(rank) + ".txt");     
        
    // write initial things to local_statistics
    std::ofstream statsFile(statsFilePath);
    if (!statsFile)
    {
        throw std::runtime_error("Failed to open output file: " + statsFilePath.string());
    }
    statsFile << "GENERATION"
        << ",BestFitness"
        << ",MaxSize(W-W),MaxSize(W-Mo),MaxSize(Mo-Mo),MaxSize(Loc)"
        << ",MinSize(W-W),MinSize(W-Mo),MinSize(Mo-Mo),MinSize(Loc)"
        << ",MeanSize(W-W),MeanSize(W-Mo),MeanSize(Mo-Mo),MeanSize(Loc)"
        << ",BISize(W-W),BISize(W-Mo),BISize(Mo-Mo),BISize(Loc)"
        << "\n";
    statsFile.close();

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
        if (RandomNumber(0.0, 1.0) < mutationProbability)
        {
            Individual child1 = parent1;

            for (int caseInd = 0; caseInd < 4; caseInd++)
            {
                const double mutationType = RandomNumber(0.0, 1.0);

                if (mutationType < 0.33)
                    child1 = MutatePoint(child1, caseInd);
                else if (mutationType < 0.67)
                    child1 = MutateSubtree(child1, caseInd);
                else
                    child1 = MutateConst(child1, caseInd, NtestInds);
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
            
            // crossover to all 4 potential trees
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
        write_gp_statistics(statsFilePath, generation);

        // --------------------------
        // send best individuals to global hof every NgensToSendInds
        // also always send the final generation 
        if (globalHof != nullptr &&
            (generation == generations - 1 || generation % NgensToSendInds == 0))
        {
            sendToGlobalHOF();
            globalHof->writeGlobalHOF(rank, outputDirectory);
        }

        // --------------------------
        // Send random individuals to the next rank and receive migrants from the previous rank.
        if (generation > 0 && generation % NgensToMigration == 0)
        {
            performMigration();
        }

        // --------------------------
        // Update the local HOF 
        if ((generation + 1) % NgensToUpd_LHOF == 0)
        {
            WriteLocalHOF(outputDir);
        }
    }

    // Take a final snapshot of local HOF in case of weird number of gens
    if (generations % NgensToUpd_LHOF != 0)
    {
        WriteLocalHOF(outputDir);
    }

}
