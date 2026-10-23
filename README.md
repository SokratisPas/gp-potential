# GP Potential

A parallel genetic-programming framework for discovering symbolic interatomic potentials from simulation data.

## Overview

This project applies symbolic regression in order to discover an analytic pair potential from reference data. The objective is to evolve expression trees that approximate the systems potential as a function of interatomic distances.

The implementation combines:

- Genetic programming with symbolic regression for potential discovery.
- MPI-based parallel evolution across multiple worker processes.
- Island-model migration to maintain diversity.
- Hall of Fame tracking for the best performing individuals.
- Training data parsed from atomic configuration files.

## Methodology

The workflow is organized as follows:

1. Parse atomic configurations and reference energies from input data.
2. Construct interatomic distance features from local neighborhoods.
3. Evaluate a population of candidate symbolic expressions against the target energy.
4. Apply evolutionary operators such as crossover and mutation to improve candidate trees.
5. Use migration between parallel islands and retain elite individuals in a Hall of Fame.
6. Record final results in the output directory for analysis and comparison.

## Requirements

The project requires the following tools and libraries:

- CMake 3.10 or newer.
- C++17 compiler.
- MPI implementation.
- OpenMP implementation.

## Clone

```bash
git clone https://github.com/SokratisPas/gp-potential.git
```

## Build

From the project root, configure and build the executable as follows:

```bash
cmake -S . -B build
cmake --build build
```

## Execution

Run the executable from /build with MPI and OpenMP using the input file:

```bash
OMP_NUM_THREADS=YOUR_N_THREADS mpirun -np YOUR_N_PROCESSES ./build/gp-potential --input input.txt
```

Use the help flag to inspect available command-line options:

```bash
./build/gp-potential --help
```

## Configuration

The solver parameters are defined in [input.txt](input.txt). The settings include:

| Parameter | Description |
| --- | --- |
| pop_size | population size of each island |
| gens | number of generations each population evolves |
| initial_ind_max_depth | initial depth of each individual (same for every island) |
| tournament_size | size of tournament selection |
| tournSelProbability | probability of selecting each individual in tournament selection |
| mutation_prob | probability for mutation (crossover probability = 1 - mutation probability) |
| const_min | minimum size for constants |
| const_max | maximum size for constatns |
| ndata_sample | number of data to use for fitness evaluation |
| last_n_data | using the last number of data for training |
| cutoff | cutoff radius |
| n_local_inds | number of individuals each island send to HoF |
| ngens_to_send_inds | number of generations to send individuals to HoF |
| ngens_to_migration | number of generations to perform migration |
| n_migrants | number of individuals each island sends to the next island during migration |
| NgensToUpd_LHOF | number of generations between local HoF file updates |
| data_path |  data path |
| output_dir | output directory (already included) |
