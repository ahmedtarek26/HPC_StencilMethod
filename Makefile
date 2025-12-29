CC = mpicc
CFLAGS = -Wall -O3 -fopenmp -std=c99
LDFLAGS = -fopenmp
TARGET = stencil_parallel
OBJS = stencil_template_parallel.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) output_*.txt error_*.txt

run: $(TARGET)
	srun ./$(TARGET) -x 1000 -y 1000 -n 100 -p 0