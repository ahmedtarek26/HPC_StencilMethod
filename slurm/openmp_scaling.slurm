#!/bin/bash
#!/bin/bash
#SBATCH -J stencil_scaling
#SBATCH -A a.elsayed
#SBATCH --partition=EPYC
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=01:00:00

module load openMPI/5.0.5   

#SBATCH --output=omp_scaling_%j.out

module load gcc/12.2.0
module load openMPI/5.0.5


EXEC="./stencil_parallel"
ARGS="-x 10000 -y 10000 -n 200 -p 1"
# EPYC nodes have 128 cores
THREAD_COUNTS=(1 2 4 8 16 32 64 128)

echo "Threads,TotalTime,CommTime,CompTime" > omp_scaling_results.csv

for t in "${THREAD_COUNTS[@]}"; do
    export OMP_NUM_THREADS=$t
    export OMP_PLACES=cores
    export OMP_PROC_BIND=spread
    mpirun -np 1 $EXEC $ARGS > temp_out.txt
    total=$(grep "Final Results" temp_out.txt | awk -F'Total=' '{print $2}' | awk -F',' '{print $1}')
    comm=$(grep "Final Results" temp_out.txt | awk -F'Comm=' '{print $2}' | awk -F',' '{print $1}')
    comp=$(grep "Final Results" temp_out.txt | awk -F'Comp=' '{print $2}' | awk '{print $1}')
    echo "$t,$total,$comm,$comp" >> scaling_results/omp_scaling_results.csv
done
rm temp_out.txt
