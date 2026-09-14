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
                   double crossoverProbability,
                   double mutationProbability,
                   std::pair<double, double> constRange,
                   const std::vector<std::vector<PairDistanceData>>& data,
                   std::function<double(
                        const Tree&, 
                        const std::vector<std::vector<PairDistanceData>>&)> 
                        fitnessFunction)
        :
        populationSize(populationSize),
        generations(generations),
        initialIndMaxDepth(initialIndMaxDepth),
        tournamentSize(tournamentSize),
        crossoverProbability(crossoverProbability),
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
    double crossoverProbability;
    double mutationProbability;
    std::pair<double, double> constRange;
    const std::vector<std::vector<PairDistanceData>>& data;

    std::mt19937 rng;   // each GeneticProgram has unique rng

    // fitness function
    std::function<double(
        const Tree&, 
        const std::vector<std::vector<PairDistanceData>>&)>
        fitnessFunction;

    void EvaluateIndividual(Individual& individual);

    const Individual& TournamentSelection();

    std::pair<Individual, Individual> Crossover(const Individual& parent1,
                                                const Individual& parent2);

    Individual Mutate(const Individual& parent);

    double RandomDouble(double min, double max);

    const Individual& BestIndividual() const;

    std::shared_ptr<Node> GenerateRandomNode(int depth);
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

        Tree tree;

        // generate random node in (1, initialIndMaxDepth)
        std::uniform_int_distribution<int> indDist(1, initialIndMaxDepth);
        tree.root = GenerateRandomNode(indDist(rng));

        ind.tree = tree;

        population.push_back(ind);
    }
}

// ================================
void GeneticProgram::EvaluateIndividual(Individual& individual)
{
    if (individual.evaluated)   // already evaluated fitness
        return;

    individual.fitness = fitnessFunction(individual.tree, data);
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
    const Individual& parent2)
{
    // clone the parents
    Individual child1 = parent1;
    Individual child2 = parent2;

    child1.tree = parent1.tree.Clone();
    child2.tree = parent2.tree.Clone();

    // collect all the nodes
    auto nodes1 = child1.tree.CollectNodes();
    auto nodes2 = child2.tree.CollectNodes();

    // catch empty nodes
    if (nodes1.empty() || nodes2.empty())
        return { child1, child2 };
    

    // choose random crossover points
    std::uniform_int_distribution<size_t> dist1(0, nodes1.size() - 1);
    std::uniform_int_distribution<size_t> dist2(0, nodes2.size() - 1);

    NodeLocation point1 = nodes1[dist1(rng)];
    NodeLocation point2 = nodes2[dist2(rng)];

    // clone the selected subtrees
    auto subtree1 = child1.tree.CloneNode(point1.node);
    auto subtree2 = child2.tree.CloneNode(point2.node);

    // replace subtrees for children
    // 
    // child 1
    if (point1.parent == nullptr)   // we chose the root
    {
        child1.tree.root = subtree2;
    }
    else                            // we chose some other node
    {
        point1.parent->children[point1.childIndex] = subtree2;
    }
    // child 2
    if (point2.parent == nullptr)
    {
        child2.tree.root = subtree1;
    }
    else
    {
        point2.parent->children[point2.childIndex] = subtree1;
    }

    // mark children as not evaluated
    child1.evaluated = false;
    child2.evaluated = false;

    child1.fitness = 1e6;
    child2.fitness = 1e6;

    child1.energyLoss = 0.0;
    child2.energyLoss = 0.0;
    child1.forceLoss = 0.0;
    child2.forceLoss = 0.0;


    return { child1, child2 };
}

// ================================
Individual GeneticProgram::Mutate(const Individual& parent)
{
    // Clone the parent
    Individual child = parent;
    child.tree = parent.tree.Clone();

    // Collect all nodes
    auto nodes = child.tree.CollectNodes();

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
        child.tree.root = newSubtree;
    }
    else
    {
        point.parent->children[point.childIndex] = newSubtree;
    }

    // Reset evaluation
    child.evaluated = false;
    child.fitness = std::numeric_limits<double>::infinity();
    child.energyLoss = 0.0;
    child.forceLoss = 0.0;

    return child;
}

// ================================
double GeneticProgram::RandomDouble(double min, double max)
{
    std::uniform_real_distribution<double> dist(min, max);
    return dist(rng);
}

// ================================
const Individual& GeneticProgram::BestIndividual() const
{
    return *std::min_element(
        population.begin(),
        population.end(),
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

    // --------------------------
    // Evolution loop
    for (int generation = 0; generation < generations; generation++)
    {
        std::vector<Individual> newPopulation;
        newPopulation.reserve(populationSize);

        // find best individual (we keep him in next generation) (elitism)
        const Individual& bestInd = BestIndividual();

        // make a copy of best ind and store to new population
        Individual bestIndCopy = bestInd;
        bestIndCopy.tree = bestInd.tree.Clone();
        newPopulation.push_back(bestIndCopy);

        // print best individual
        std::cout << "Best individual : ";
        bestIndCopy.tree.printTree();
        std::cout << "\n";


        while (newPopulation.size() < populationSize)
        {
            // Select parents
            const Individual& parent1 = TournamentSelection();
            const Individual& parent2 = TournamentSelection();

            std::pair<Individual, Individual> children;

            // Crossover
            if (RandomDouble(0.0, 1.0) < crossoverProbability)
            {
                children = Crossover(parent1, parent2);
            }
            else
            {
                children = { parent1, parent2 };
            }

            // Mutation
            if (RandomDouble(0.0, 1.0) < mutationProbability)
            {
                children.first = Mutate(children.first);
            }

            if (RandomDouble(0.0, 1.0) < mutationProbability)
            {
                children.second = Mutate(children.second);
            }

            // Evaluate and add first child
            EvaluateIndividual(children.first);
            newPopulation.push_back(children.first);

            // Evaluate and Add second child if there is room
            if (newPopulation.size() < populationSize)
            {
                EvaluateIndividual(children.second);
                newPopulation.push_back(children.second);
            }
        }

        // Replace the old population
        population = std::move(newPopulation);

        //-------------------------
        // Print hof
        double bestFitness = population.front().fitness;
        int sumSize = 0;
        int maxSize = 0;
        int minSize = 10000;

        for (const auto& individual : population)
        {
            bestFitness = std::min(bestFitness, individual.fitness);

            int indSize = individual.tree.Size();
            sumSize += indSize;
            maxSize = std::max(maxSize, indSize);
            minSize = std::min(minSize, indSize);
        }

        std::cout << "GENERATION " 
            << generation
            << "\nBest fitness \t= "
            << bestFitness
            << "\nAverage Size \t= "
            << static_cast<double>(sumSize) / populationSize
            << "\nMax Size \t= "
            << maxSize
            << "\nMin Size \t= "
            << minSize
            << "\n=====================\n";
    }
}