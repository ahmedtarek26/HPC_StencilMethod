CC = gcc
MPICC = mpicc
CFLAGS = -O3 -lm -Iinclude
OMPFLAGS = -fopenmp

all: serial parallel

serial: src/stencil_template_serial.c
	$(CC) $(CFLAGS) -o stencil_serial src/stencil_template_serial.c

parallel: src/stencil_template_parallel.c
	$(MPICC) $(OMPFLAGS) $(CFLAGS) -o stencil_parallel src/stencil_template_parallel.c

clean:
	rm -f stencil_serial stencil_parallel *.out *.err