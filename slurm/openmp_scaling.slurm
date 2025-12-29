#!/bin/bash
#SBATCH -J stencil_omp_scaling
#SBATCH -A uTS25_Tornator_0
#SBATCH --partition=dcgp_usr_prod
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --time=01:00:00
#SBATCH --output=omp_scaling_%j.out

module load gcc/12.2.0
module load openmpi/4.1.6--gcc--12.2.0

EXEC="./stencil_parallel"
ARGS="-x 10000 -y 10000 -n 200 -p 1"
THREAD_COUNTS=(1 2 4 8 16 28 32 56 84 112)

# Prepare CSV header
echo "Threads,TotalTime,CommTime,CompTime" > omp_scaling_results.csv

echo "Starting OpenMP scaling tests..."

for t in "${THREAD_COUNTS[@]}"; do
    export OMP_NUM_THREADS=$t
    export OMP_PLACES=cores
    export OMP_PROC_BIND=spread

    echo "Running with $t threads..."
    mpirun -np 1 $EXEC $ARGS > temp_out.txt
    
    # Extract data using awk (adjusting to the print format in the C code)
    total=$(grep "Final Results" temp_out.txt | awk -F'Total=' '{print $2}' | awk -F',' '{print $1}')
    comm=$(grep "Final Results" temp_out.txt | awk -F'Comm=' '{print $2}' | awk -F',' '{print $1}')
    comp=$(grep "Final Results" temp_out.txt | awk -F'Comp=' '{print $2}' | awk '{print $1}')
    
    echo "$t,$total,$comm,$comp" >> scaling_results/omp_scaling_results.csv
done

rm temp_out.txt
echo "OpenMP scaling tests completed. Results saved in scaling_results/omp_scaling_results.csv"
