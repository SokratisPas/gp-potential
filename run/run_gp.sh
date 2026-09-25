#!/bin/bash
#SBATCH --job-name=gp-potential
#SBATCH --partition=testing
#SBATCH --ntasks=4
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_ROOT="${REPO_ROOT}/run/output"

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_ROOT"

# cluster modules
module purge
module load gcc/15.2.0
module load cmake/3.31.8
module load intel-oneapi-mpi/2021.16.0

# pmi2, intel mpi
export I_MPI_PMI_LIBRARY="${I_MPI_PMI_LIBRARY:-/usr/lib64/libpmi2.so}"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4

JOB_OUT_DIR="${OUTPUT_ROOT}/${SLURM_JOB_ID:-local}"
mkdir -p "$JOB_OUT_DIR"

srun --mpi=pmi2 "${REPO_ROOT}/build/gp-potential" \
  --config "${REPO_ROOT}/run/input.txt" \
  --output-dir "$JOB_OUT_DIR"