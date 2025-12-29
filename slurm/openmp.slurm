#!/bin/bash
#SBATCH -J stencil_openmp
#SBATCH --account=uTS25_Tornator_0
#SBATCH --partition=EPYC
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:30:00
#SBATCH --output=scaling_results/openmp_%j.out

module load gnu openmpi

export OMP_NUM_THREADS=32  # Vary
export OMP_PROC_BIND=spread

mpirun -np 1 ./stencil_parallel -x 10000 -y 10000 -n 200 -p 1