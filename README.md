# HPC Stencil Method Project - Advanced HPC 2024-2025

Implementation of a 5-point stencil for heat diffusion using hybrid MPI + OpenMP.

## Compile
module load gnu openmpi
make serial     # For stencil_serial
make parallel   # For stencil_parallel

## Run Locally (Test)
./stencil_serial -x 1000 -y 1000 -n 100
mpirun -np 4 ./stencil_parallel -x 2000 -y 2000 -n 200

## Run on Orfeo
Use slurm scripts: sbatch slurm/openmp.slurm etc.

## Features
- MPI domain decomposition with non-blocking halo exchanges
- OpenMP parallel loops with affinity
- Timing: computation vs communication
- Periodic boundaries support
- Energy conservation check

Results in report.pdf (scaling plots on Orfeo).