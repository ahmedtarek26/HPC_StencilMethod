#!/bin/bash
#SBATCH -J mpi_strong_scaling
#SBATCH -A uTS25_Tornator_0
#SBATCH --partition=dcgp_usr_prod
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --time=01:00:00
#SBATCH --output=mpi_strong_%j.out

module load gcc/12.2.0
module load openmpi/4.1.6--gcc--12.2.0

EXEC="./stencil_parallel"
ARGS="-x 20000 -y 20000 -n 500 -p 1"
NODE_COUNTS=(1 2 4)

echo "Nodes,TotalTime,CommTime,CompTime" > mpi_strong_results.csv

for n in "${NODE_COUNTS[@]}"; do
    export OMP_NUM_THREADS=112
    export OMP_PLACES=cores
    export OMP_PROC_BIND=spread

    echo "Running on $n nodes..."
    srun --nodes=$n --ntasks=$n --cpus-per-task=112 $EXEC $ARGS > temp_out.txt
    
    total=$(grep "Final Results" temp_out.txt | awk -F'Total=' '{print $2}' | awk -F',' '{print $1}')
    comm=$(grep "Final Results" temp_out.txt | awk -F'Comm=' '{print $2}' | awk -F',' '{print $1}')
    comp=$(grep "Final Results" temp_out.txt | awk -F'Comp=' '{print $2}' | awk '{print $1}')
    
    echo "$n,$total,$comm,$comp" >> scaling_results/mpi_strong_results.csv
done

rm temp_out.txt
echo "MPI strong scaling completed. Results saved in scaling_results/mpi_strong_results.csv"
