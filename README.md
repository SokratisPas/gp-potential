# GP Potential

This project explores symbolic regression with a genetic program to learn a pair potential for atomic systems. The goal is to evolve expression trees that approximate the energy as a function of interatomic distances.

## Workflow

1. Parse training snapshots from an XYZ file.
2. Build neighbor information for each atom.
3. Evaluate candidate symbolic expressions on interatomic distances.
4. Compare predicted energy against reference energy.
5. Evolve better trees using genetic programming operators such as crossover and mutation.

## Main components

- `src/main.cpp`  
  Entry point. Defines the GP parameters, reads the training data, creates the genetic program, and runs evolution.

- `src/geneticProgram.h`  
  Implements the genetic programming algorithm.

- `src/fitness.h`  
  Contains the fitness functions used to score individuals. The current implementation focuses on a 2-element system and evaluates a random training subset for each fitness call.

- `src/xyz-parser.h`  
  Parses XYZ snapshots and extracts lattice information, atomic positions, and neighbor lists.

- `src/treeClass.h`  
  Defines the expression tree structure used by the GP.

- `src/primitives.h`  
  Contains primitive operations used as tree nodes.

- `src/individual.h`  
  Defines an individual in the population, containing several trees.


## Run with CMake

From the project root:

```bash
cmake -S . -B build
cmake --build build
mpirun -np NUMBR_PROCESSES ./build/gp-potential
```