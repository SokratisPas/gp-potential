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
        double cutoffIn,
        double cutoffOut,
        const std::vector<Snapshot>& data,
        int NdataSample,
        std::function<double(
            const Tree& tree_W,
            const Tree& tree_Mo,
            double cutoffIn,
            double cutoffOut,
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
        cutoffIn(cutoffIn),
        cutoffOut(cutoffOut),
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
    double cutoffIn;
    double cutoffOut;
    const std::vector<Snapshot>& data;
    int rank;   // ID of each process
    int numTasks;           // total number of MPI ranks
    GlobalHOF* globalHof;   // shared global HOF container
    int NlocalInds;
    int NgensToSendInds;    // send inds to globalHOF and update globalHOF!
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
        &Pow,
        &SUM_W,
        &SUM_Mo
    };
    
    // fitness function
    std::function<double(
        const Tree& tree_W,
        const Tree& tree_Mo,
        double cutoffIn,
        double cutoffOut,
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
        const Individual& parent2
    );
    Individual MutateSubtree(const Individual& parent);
    Individual MutatePoint(const Individual& parent);
    Individual MutateConst(const Individual& parent, int NtestInds);
    std::shared_ptr<Node> GenerateRandomNode(int depth, bool allowSum = true);
    bool HasNestedSum(const std::shared_ptr<Node>& node, bool insideSum = false) const;
    bool HasNestedSum(const Individual& individual) const;
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
    while (pos < expression.size() &&
        (std::isalnum(static_cast<unsigned char>(expression[pos])) || expression[pos] == '_'))
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

    const auto deserializeTree = [this](const std::string& serializedTree)
    {
        const std::string treeText = Trim(serializedTree);
        if (treeText.empty())
        {
            auto root = std::make_shared<Node>();
            root->primitive = &Const;
            root->constant = 0.0;
            return root;
        }

        size_t pos = 0;
        auto root = ParseTreeExpression(treeText, pos);
        if (!root)
        {
            root = std::make_shared<Node>();
            root->primitive = &Const;
            root->constant = 0.0;
        }
        return root;
    };

    individual.tree_W.root = deserializeTree(entry.tree_W);
    individual.tree_Mo.root = deserializeTree(entry.tree_Mo);

    return individual;
}

// ================================
std::shared_ptr<Node> GeneticProgram::GenerateRandomNode(int depth, bool allowSum)
{
    auto node = std::make_shared<Node>();

    // if maximum depth reached create terminal (var or const)
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

    // max depth is not reached
    // Choose a terminal early 30% of the time else choose a function.
    if (RandomNumber(0.0, 1.0) < 0.3)
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

    // Choose only primitives allowed in this subtree.
    // no nested summations allowed
    std::vector<const Primitive*> availableFunctions;
    for (const Primitive* function : functions)
    {
        const bool isSum =
            function->type == NodeType::SUM_W ||
            function->type == NodeType::SUM_Mo;
        if (allowSum || !isSum)
            availableFunctions.push_back(function);
    }

    node->primitive = availableFunctions[
        RandomNumber(size_t{0}, availableFunctions.size() - 1)
    ];

    // create children according to arity
    const bool isSumNode =
        node->primitive->type == NodeType::SUM_W ||
        node->primitive->type == NodeType::SUM_Mo;
    for (int i = 0; i < node->primitive->arity; i++)
    {
        const bool allowChildSum = allowSum && !isSumNode;
        node->children.push_back(GenerateRandomNode(depth - 1, allowChildSum));
    }

    return node;
}

// ================================
bool GeneticProgram::HasNestedSum(
    const std::shared_ptr<Node>& node,
    bool insideSum
) const
{
    if (!node || !node->primitive)
        return false;

    const bool isSum =
        node->primitive->type == NodeType::SUM_W ||
        node->primitive->type == NodeType::SUM_Mo;
    if (insideSum && isSum)
        return true;

    for (const auto& child : node->children)
    {
        if (HasNestedSum(child, insideSum || isSum))
            return true;
    }

    return false;
}

// ================================
bool GeneticProgram::HasNestedSum(const Individual& individual) const
{
    return HasNestedSum(individual.tree_W.root) ||
        HasNestedSum(individual.tree_Mo.root);
}

// ================================
void GeneticProgram::InitializePopulation()
{
    population.clear();

    population.reserve(populationSize);


    for (int i = 0; i < populationSize; i++)
    {
        Individual ind;

        ind.tree_W.root = GenerateRandomNode(
            RandomNumber(1, std::min(initialIndMaxDepth, maxDepth - 1))
        );
        ind.tree_Mo.root = GenerateRandomNode(
            RandomNumber(1, std::min(initialIndMaxDepth, maxDepth - 1))
        );

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
    if (HasNestedSum(individual))
        throw std::runtime_error("Individual contains nested Sum primitives.");

    individual.fitness = fitnessFunction(
        individual.tree_W,
        individual.tree_Mo,
        cutoffIn,
        cutoffOut,
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
    return individual.tree_W.Depth() <= maxDepth &&
        individual.tree_Mo.Depth() <= maxDepth;
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
    const Individual& parent2
)
{
    Individual child1 = parent1;
    Individual child2 = parent2;

    const std::array<Tree Individual::*, 2> trees = {
        &Individual::tree_W,
        &Individual::tree_Mo
    };

    for (Tree Individual::*treeMember : trees)
    {
        Tree& child1Tree = child1.*treeMember;
        Tree& child2Tree = child2.*treeMember;
        const Tree& parent1Tree = parent1.*treeMember;
        const Tree& parent2Tree = parent2.*treeMember;

        child1Tree = parent1Tree.Clone();
        child2Tree = parent2Tree.Clone();

        auto nodes1 = child1Tree.CollectNodes();
        auto nodes2 = child2Tree.CollectNodes();
        if (nodes1.empty() || nodes2.empty())
            continue;

        NodeLocation point1 = nodes1[RandomNumber(size_t{0}, nodes1.size() - 1)];
        NodeLocation point2 = nodes2[RandomNumber(size_t{0}, nodes2.size() - 1)];
        auto subtree1 = child1Tree.CloneNode(point1.node);
        auto subtree2 = child2Tree.CloneNode(point2.node);

        if (point1.parent == nullptr)
            child1Tree.root = subtree2;
        else
            point1.parent->children[point1.childIndex] = subtree2;

        if (point2.parent == nullptr)
            child2Tree.root = subtree1;
        else
            point2.parent->children[point2.childIndex] = subtree1;

        if (child1Tree.Depth() > maxDepth || HasNestedSum(child1Tree.root))
            child1Tree = parent1Tree.Clone();
        if (child2Tree.Depth() > maxDepth || HasNestedSum(child2Tree.root))
            child2Tree = parent2Tree.Clone();
    }

    // mark children as not evaluated
    child1.evaluated = false;
    child2.evaluated = false;

    child1.fitness = std::numeric_limits<double>::infinity();
    child2.fitness = std::numeric_limits<double>::infinity();

    return { child1, child2 };
}

// ================================
Individual GeneticProgram::MutateSubtree(const Individual& parent)
{
    Individual child = parent;
    const std::array<Tree Individual::*, 2> trees = {
        &Individual::tree_W,
        &Individual::tree_Mo
    };

    for (Tree Individual::*treeMember : trees)
    {
        Tree& childTree = child.*treeMember;
        const Tree& parentTree = parent.*treeMember;
        childTree = parentTree.Clone();

        if (!childTree.root)
        {
            childTree.root = GenerateRandomNode(RandomNumber(1, 3));
            continue;
        }

        auto nodes = childTree.CollectNodes();
        if (nodes.empty())
            return parent;

        NodeLocation point = nodes[RandomNumber(size_t{0}, nodes.size() - 1)];
        auto newSubtree = GenerateRandomNode(RandomNumber(1, 3));

        if (point.parent == nullptr)
            childTree.root = newSubtree;
        else
            point.parent->children[point.childIndex] = newSubtree;

        if (childTree.Depth() > maxDepth || HasNestedSum(childTree.root))
            return parent;
    }

    child.evaluated = false;
    child.fitness = std::numeric_limits<double>::infinity();

    return child;
}

// ================================
Individual GeneticProgram::MutatePoint(const Individual& parent)
{
    Individual child = parent;
    child.tree_W = parent.tree_W.Clone();
    child.tree_Mo = parent.tree_Mo.Clone();

    bool mutated = false;
    for (Tree Individual::*treeMember : { &Individual::tree_W, &Individual::tree_Mo })
    {
        Tree& childTree = child.*treeMember;
        if (!childTree.root)
            continue;

        auto nodes = childTree.CollectNodes();
        if (nodes.empty())
            continue;

        const auto& node = nodes[RandomNumber(size_t{0}, nodes.size() - 1)].node;
        if (!node || !node->primitive)
            continue;

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
        else if (node->primitive->type == NodeType::Const ||
            node->primitive->type == NodeType::Var)
        {
            node->primitive = &Const;
            node->constant = RandomNumber(constRange.first, constRange.second);
            mutated = true;
        }
    }

    if (mutated)
    {
        if (HasNestedSum(child))
            return parent;

        child.evaluated = false;
        child.fitness = std::numeric_limits<double>::infinity();
    }
    return child;
}

// ================================
Individual GeneticProgram::MutateConst(const Individual& parent, int NtestInds)
{
    if (NtestInds <= 0)
        return parent;

    const std::array<Tree Individual::*, 2> trees = {
        &Individual::tree_W,
        &Individual::tree_Mo
    };

    bool hasConstants = false;
    for (Tree Individual::*treeMember : trees)
    {
        const Tree& tree = parent.*treeMember;
        for (const auto& location : tree.CollectNodes())
        {
            if (location.node && location.node->primitive &&
                location.node->primitive->type == NodeType::Const)
            {
                hasConstants = true;
                break;
            }
        }
    }

    if (!hasConstants)
        return parent;

    Individual bestCandidate = parent;
    bestCandidate.evaluated = false;
    bestCandidate.fitness = std::numeric_limits<double>::infinity();

    for (int test = 0; test < NtestInds; ++test)
    {
        Individual candidate = parent;
        candidate.tree_W = parent.tree_W.Clone();
        candidate.tree_Mo = parent.tree_Mo.Clone();

        for (Tree Individual::*treeMember : trees)
        {
            Tree& tree = candidate.*treeMember;
            std::vector<std::shared_ptr<Node>> constantNodes;
            for (const auto& location : tree.CollectNodes())
            {
                if (location.node && location.node->primitive &&
                    location.node->primitive->type == NodeType::Const)
                {
                    constantNodes.push_back(location.node);
                }
            }

            if (constantNodes.empty())
                continue;

            std::shuffle(constantNodes.begin(), constantNodes.end(), rng);

            const int maxMutatedConstants = std::max(
                1,
                static_cast<int>(constantNodes.size() / 3 + 1)
            );
            const int numberToMutate = RandomNumber(
                1,
                std::min<int>(maxMutatedConstants, static_cast<int>(constantNodes.size()))
            );
            for (int i = 0; i < numberToMutate; ++i)
                constantNodes[i]->constant = RandomNumber(constRange.first, constRange.second);
        }

        candidate.evaluated = false;
        candidate.fitness = std::numeric_limits<double>::infinity();

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

    constexpr size_t maxIndividualsToWrite = 10;
    std::vector<const Individual*> writtenIndividuals;
    writtenIndividuals.reserve(std::min(maxIndividualsToWrite, population.size()));

    size_t writtenCount = 0;
    for (const Individual& indiv : population)
    {
        const bool alreadyWritten = std::any_of(
            writtenIndividuals.begin(),
            writtenIndividuals.end(),
            [&indiv](const Individual* written)
            {
                return written->tree_W == indiv.tree_W &&
                    written->tree_Mo == indiv.tree_Mo;
            }
        );
        if (alreadyWritten)
            continue;

        writtenIndividuals.push_back(&indiv);

        hofFile << "==== Individual: " << writtenCount + 1 << " ===="
            << "\n== Fitness : "
            << indiv.fitness
            << "\n";

        hofFile << "== Tree_W (Size: " << indiv.tree_W.Size() << ")\n"
            << indiv.tree_W.convertToStr() << "\n"
            << "== Tree_Mo (Size: " << indiv.tree_Mo.Size() << ")\n"
            << indiv.tree_Mo.convertToStr() << "\n";
        hofFile << "---------------------------------------\n";

        if (++writtenCount == maxIndividualsToWrite)
            break;
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
        entry.tree_W = individual.tree_W.convertToStr();
        entry.tree_Mo = individual.tree_Mo.convertToStr();
        entry.size_W = individual.tree_W.Size();
        entry.size_Mo = individual.tree_Mo.Size();
        entry.depth_W = individual.tree_W.Depth();
        entry.depth_Mo = individual.tree_Mo.Depth();
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
        if (IsWithinMaxDepth(incoming) && !HasNestedSum(incoming))
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
    std::array<int, 2> maxSize = {0, 0};
    std::array<int, 2> minSize = {
        std::numeric_limits<int>::max(),
        std::numeric_limits<int>::max()
    };
    std::array<double, 2> sumSize = {0.0, 0.0};
    
    for (const auto& individual : population)
    {
        const std::array<int, 2> indSize = {
            individual.tree_W.Size(),
            individual.tree_Mo.Size()
        };
        for (size_t treeIndex = 0; treeIndex < indSize.size(); ++treeIndex)
        {
            sumSize[treeIndex] += indSize[treeIndex];
            maxSize[treeIndex] = std::max(maxSize[treeIndex], indSize[treeIndex]);
            minSize[treeIndex] = std::min(minSize[treeIndex], indSize[treeIndex]);
        }
    }

    statsFile << generation + 1 << ","
        << bestFitness << ","
        << maxSize[0] << ","
        << minSize[0] << ","
        << sumSize[0] / static_cast<double>(population.size()) << ","
        << population[0].tree_W.Size() << ","
        << maxSize[1] << ","
        << minSize[1] << ","
        << sumSize[1] / static_cast<double>(population.size()) << ","
        << population[0].tree_Mo.Size()
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
        << ",MaxSize_W"
        << ",MinSize_W"
        << ",MeanSize_W"
        << ",BISize_W"
        << ",MaxSize_Mo"
        << ",MinSize_Mo"
        << ",MeanSize_Mo"
        << ",BISize_Mo"
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

            const double mutationType = RandomNumber(0.0, 1.0);
            if (mutationType < 0.33)
                child1 = MutatePoint(child1);
            else if (mutationType < 0.67)
                child1 = MutateSubtree(child1);
            else
                child1 = MutateConst(child1, NtestInds);

            EvaluateIndividual(child1);
            ReplaceInd(child1, worstIndices[0]);
        }
        else    //  Crossover
        {
            // select second parent
            // (for crossover we need 2 individuals)
            const Individual& parent2 = TournamentSelection();
            std::pair<Individual, Individual> children{ parent1, parent2 };
            
            children = Crossover(children.first, children.second);

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

