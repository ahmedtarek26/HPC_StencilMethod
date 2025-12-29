# Orfeo-specific Makefile - adjust modules as needed

# Compiler and flags
CC = mpicc
CFLAGS = -Wall -O3 -fopenmp -std=c99
LDFLAGS = -fopenmp
TARGET = stencil_parallel
OBJS = stencil_template_parallel.o

# Modules to load (check with 'module avail' on Orfeo)
# module load gcc/11.3.0
# module load openmpi/4.1.4

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) output_*.txt error_*.txt

run: $(TARGET)
	srun ./$(TARGET) -x 1000 -y 1000 -n 100 -p 0

# Scaling study targets
openmp_scaling: $(TARGET)
	@echo "Running OpenMP scaling study..."
	@for threads in 1 2 4 8 14 28 56; do \
		echo "Testing with $$threads threads..."; \
		sbatch --cpus-per-task=$$threads submit_orfeo.sh; \
	done

.PHONY: all clean run openmp_scaling