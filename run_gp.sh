#!/bin/bash

#SBATCH --job-name=gp-potential
#SBATCH --partition=testing
#SBATCH --ntasks=4
#SBATCH --time=00:30:00
#SBATCH --output=/home/s/spastel/projects/gp-potential/output/%x-%j.out
#SBATCH --error=/home/s/spastel/projects/gp-potential/output/%x-%j.err

set -e

# Go to project directory
cd /home/s/spastel/projects/gp-potential

# Cluster modules
module purge
module load gcc/15.2.0
module load cmake/3.31.8
module load intel-oneapi-mpi/2021.16.0

# Intel MPI + Slurm PMI2
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so

# Build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4

# Create output directory for this job
mkdir -p output/$SLURM_JOB_ID

# Run
srun --mpi=pmi2 -n $SLURM_NTASKS \
    ./build/gp-potential \
    --config input.txt \
    --output-dir output/$SLURM_JOB_ID