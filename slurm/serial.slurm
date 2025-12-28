#!/bin/bash
#SBATCH -J stencil_serial
#SBATCH -A uTS25_Tornator_0
#SBATCH --partition=epyc
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=serial_%j.out

module load gnu openmpi

./stencil_serial -x 1000 -y 1000 -n 100 -o 1