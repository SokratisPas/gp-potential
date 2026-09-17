#pragma once

#include <random>
#include <functional>
#include <utility>
#include <array>
#include <vector>
#include <algorithm>

#include "treeClass.h"
#include "individual.h"
#include "primitives.h"
#include "POSCARdata.h"

// --------------------------------------------------------
class GeneticProgram {
public:
    std::vector<Individual> population;

    void InitializePopulation();
    void EvaluatePopulation();
    void Run();

    // Constructor
    GeneticProgram(int populationSize,
                   int generations, 
                   int initialIndMaxDepth,
                   int tournamentSize,
                   double mutationProbability,
                   std::pair<double, double> constRange,
                   const std::vector<std::vector<PairDistanceData>>& data,
                   std::function<double(
                        const std::array<Tree,4>&, 
                        const std::vector<std::vector<PairDistanceData>>&)> 
                        fitnessFunction)
        :
        populationSize(populationSize),
        generations(generations),
        initialIndMaxDepth(initialIndMaxDepth),
        tournamentSize(tournamentSize),
        mutationProbability(mutationProbability),
        constRange(constRange),
        data(data),
        fitnessFunction(fitnessFunction),
        rng(std::random_device{}())   // initialize rng in constructor
    {
    }


private:
    int populationSize;
    int generations;
    int initialIndMaxDepth;    // depth of initial individuals
    int tournamentSize;
    double mutationProbability;
    std::pair<double, double> constRange;
    const std::vector<std::vector<PairDistanceData>>& data;

    std::mt19937 rng;   // each GeneticProgram has unique rng

    // fitness function
    std::function<double(
        const std::array<Tree,4>&, 
        const std::vector<std::vector<PairDistanceData>>&)>
        fitnessFunction;

    void EvaluateIndividual(Individual& individual);

    const Individual& TournamentSelection();

    std::pair<Individual, Individual> Crossover(const Individual& parent1,
                                                const Individual& parent2,
                                                int caseInd);

    Individual Mutate(const Individual& parent, int caseInd);

    double RandomDouble(double min, double max);

    std::shared_ptr<Node> GenerateRandomNode(int depth);

    void ReplaceInd(const Individual& newInd, const Individual& oldInd);  

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


    if (choice(rng) == 0)
    {
        // terminal

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
    std::vector<const Primitive*> functions =
    {
        &Add,
        &Sub,
        &Mul,
        &Div,
        &Inv6,
        &Inv12
    };

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

    individual.fitness = fitnessFunction(individual.trees, data);
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
void GeneticProgram::ReplaceInd(const Individual& newInd, const Individual& oldInd)
{
    auto it = std::find(population.begin(), population.end(), oldInd);
    if (it != population.end())
    {
        *it = newInd;
    }
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
    // Create the initial population
    InitializePopulation();

    // Evaluate intial population
    EvaluatePopulation();

    // sort population
    SortPopulation();

    // find worst individuals
    std::array<Individual, 2> worstInds = { 
        population[populationSize - 1], 
        population[populationSize - 2] };

    // --------------------------
    // Evolution loop
    for (int generation = 0; generation < generations; generation++)
    {
        // Select first parent 
        const Individual& parent1 = TournamentSelection();
        
        //          Mutation
        if (RandomDouble(0.0, 1.0) < mutationProbability)
        {
            Individual child1;

            for (int caseInd = 0; caseInd < 4; caseInd++)
            {
                child1 = Mutate(child1, caseInd);
            }

            EvaluateIndividual(child1);
            ReplaceInd(child1, worstInds[0]);
        }
        else    //  Crossover
        {
            // select second parent
            // (for crossover we need 2 individuals)
            const Individual& parent2 = TournamentSelection();
            std::pair<Individual, Individual> children;
            
            for (int caseInd = 0; caseInd < 4; caseInd++)
            {
                children = Crossover(parent1, parent2, caseInd);                
            }

            EvaluateIndividual(children.first);
            EvaluateIndividual(children.second);
            ReplaceInd(children.first, worstInds[0]);
            ReplaceInd(children.second, worstInds[1]);
        }

        // sort population
        SortPopulation();

        // find worst individuals
        std::array<Individual, 2> worstInds = { 
            population[populationSize - 1], 
            population[populationSize - 2] };

        //-------------------------
        // Print hof
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

        std::cout << "GENERATION " 
            << generation
            << "\nBest fitness \t= "
            << bestFitness
            << "\nMax Size \t= "
            << maxSize[0] << ", " << maxSize[1] << ", " << maxSize[2] << ", " << maxSize[3]
            << "\nMin Size \t= "
            << minSize[0] << ", " << minSize[1] << ", " << minSize[2] << ", " << minSize[3]
            << "\n=====================\n";
    }
}