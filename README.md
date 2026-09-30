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

## Repository Structure

| Path | Description |
| --- | --- |
| [CMakeLists.txt](CMakeLists.txt) | Project configuration for CMake and MPI build setup. |
| [input.txt](input.txt) | Main runtime configuration for population size, evolution parameters, dataset path, and output directory. |
| [src/main.cpp](src/main.cpp) | Entry point for MPI setup, configuration loading, and orchestration of the evolutionary run. |
| [src/parsers](src/parsers) | Data parsers for XYZ and POSCAR-style inputs. |
| [src/data](src/data) | Training datasets used for model fitting. |
| [output](output) | Generated statistics and HoF outputs from optimization runs. |

## Requirements

The project requires the following tools and libraries:

- CMake 3.10 or newer.
- C++17 compiler.
- MPI implementation.
- A training dataset in a compatible atomic-structure format.

## Build

From the project root, configure and build the executable as follows:

```bash
cmake -S . -B build
cmake --build build
```

## Execution

Run the solver with MPI using the input file:

```bash
mpirun -np N_PROCESSES ./build/gp-potential --input input.txt
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
| mutation_prob | probability for mutation (crossover probability = 1 - mutation probability) |
| const_min | minimum size for constants |
| const_max | maximum size for constatns |
| ndata_sample | number of data to use for fitness evaluation |
| last_n_data | using the last number of data for training |
| cutoff | cutoff radius |
| n_local_inds | number of individuals each island send to HoF |
| ngens_to_send_inds | number of generations to send individuals to HoF |
| ngens_to_migration | number of generations to perform migration |
| data_path |  |
| output_dir |  |
