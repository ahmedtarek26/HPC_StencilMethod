#!/bin/bash
#SBATCH -J mpi_weak_scaling
#SBATCH -A uTS25_Tornator_0
#SBATCH --partition=dcgp_usr_prod
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --time=01:00:00
#SBATCH --output=mpi_weak_%j.out

module load gcc/12.2.0
module load openmpi/4.1.6--gcc--12.2.0

EXEC="./stencil_parallel"
BASE_SIZE=10000
NODE_COUNTS=(1 2 4)

echo "Nodes,Size,TotalTime,CommTime,CompTime" > mpi_weak_results.csv

for n in "${NODE_COUNTS[@]}"; do
    # Scale size by sqrt(n) to keep work per node constant in 2D
    SIZE=$(echo "scale=0; $BASE_SIZE * sqrt($n)" | bc -l)
    
    export OMP_NUM_THREADS=112
    export OMP_PLACES=cores
    export OMP_PROC_BIND=spread

    echo "Running on $n nodes with size ${SIZE}x${SIZE}..."
    srun --nodes=$n --ntasks=$n --cpus-per-task=112 $EXEC -x $SIZE -y $SIZE -n 500 -p 1 > temp_out.txt
    
    total=$(grep "Final Results" temp_out.txt | awk -F'Total=' '{print $2}' | awk -F',' '{print $1}')
    comm=$(grep "Final Results" temp_out.txt | awk -F'Comm=' '{print $2}' | awk -F',' '{print $1}')
    comp=$(grep "Final Results" temp_out.txt | awk -F'Comp=' '{print $2}' | awk '{print $1}')
    
    echo "$n,$SIZE,$total,$comm,$comp" >> scaling_results/mpi_weak_results.csv
done

rm temp_out.txt
echo "MPI weak scaling completed. Results saved in scaling_results/mpi_weak_results.csv"
