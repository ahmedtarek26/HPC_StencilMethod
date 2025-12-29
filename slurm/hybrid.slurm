#!/bin/bash
#SBATCH -J stencil_hybrid
#SBATCH -A uTS25_Tornator_0
#SBATCH --partition=EPYC
#SBATCH -N 8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=scaling_results/hybrid_%j.out

module load gnu openmpi

export OMP_NUM_THREADS=16
export OMP_PROC_BIND=close

mpirun ./stencil_parallel -x 20000 -y 20000 -n 500 -p 1