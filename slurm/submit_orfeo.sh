#!/bin/bash
#SBATCH --partition=EPYC          # Use the correct partition name from your list
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1       # 1 MPI task per node for OpenMP scaling
#SBATCH --cpus-per-task=56        # All cores for single-node tests
#SBATCH --time=00:30:00           # 30 minutes for testing
#SBATCH --job-name=stencil_test
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --exclusive               # Uncomment for full node access if needed

# Orfeo environment settings
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_SCHEDULE=static

# Print debug info
echo "=== Job Info ==="
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $SLURMD_NODENAME"
echo "Tasks: $SLURM_NTASKS"
echo "CPUs/Task: $SLURM_CPUS_PER_TASK"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"

# Load modules (adjust based on 'module avail')
module purge
module load gcc/11.3.0
module load openmpi/4.1.4

# Compile
make clean
make

# Run with parameters
echo "=== Execution Start ==="
srun ./stencil_parallel -x 2000 -y 2000 -n 1000 -p 0
echo "=== Execution End ==="

# Check output
if [ -f output_*.txt ]; then
    tail -20 output_*.txt
fi