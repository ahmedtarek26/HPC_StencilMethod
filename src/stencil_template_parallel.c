#include "stencil_template_parallel.h"
#include <math.h>
#include <stdlib.h>

int g_n_omp_threads = 1;
double* g_per_thread_comp_time = NULL;
__thread double thread_local_comp_time = 0.0;

// STENCIL PARAMETERS - ADJUST AS NEEDED
#define ALPHA 0.1
#define DX 1.0
#define DY 1.0
#define DT 0.01

// Direction indices
enum { NORTH=0, SOUTH, WEST, EAST, OLD=0, NEW };

int main(int argc, char **argv) {
    MPI_Comm myCOMM_WORLD;
    int Rank, Ntasks, level_obtained;
    int neighbours[4];
    int Niterations, periodic, output_energy_stat_perstep;
    vec2_t S, N;
    int Nsources, Nsources_local;
    vec2_t *Sources_local;
    double energy_per_source;
    plane_t planes[2];
    buffers_t buffers[2];

    // Initialize MPI with thread support
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &level_obtained);
    MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
    MPI_Comm_size(MPI_COMM_WORLD, &Ntasks);
    MPI_Comm_dup(MPI_COMM_WORLD, &myCOMM_WORLD);

    // Get OpenMP threads
    #pragma omp parallel
    {
        #pragma omp master
        g_n_omp_threads = omp_get_num_threads();
    }
    g_per_thread_comp_time = (double*)calloc(g_n_omp_threads, sizeof(double));

    // Initialize domain
    initialize(&myCOMM_WORLD, Rank, Ntasks, argc, argv, &S, &N, &periodic, &output_energy_stat_perstep,
               neighbours, &Niterations, &Nsources, &Nsources_local, &Sources_local, &energy_per_source,
               &planes[0], &buffers[0]);

    int current = OLD;
    double t_start = MPI_Wtime();
    double total_comm_time = 0.0, total_comp_time = 0.0;
    double alpha_dt_dx2 = ALPHA * DT / (DX * DX);
    double alpha_dt_dy2 = ALPHA * DT / (DY * DY);

    // MAIN LOOP
    for (int iter = 0; iter < Niterations; ++iter) {
        // Inject energy
        inject_energy(periodic, Nsources_local, Sources_local, energy_per_source, &planes[current], N);

        // === COMMUNICATION PHASE ===
        double comm_start = MPI_Wtime();
        double* cp = planes[current].data;
        int sx = planes[current].size[_x_], sy = planes[current].size[_y_];
        int fsx = sx + 2;

        // Set up buffer pointers (NORTH/SOUTH are contiguous)
        buffers[SEND][NORTH] = cp + fsx;
        buffers[SEND][SOUTH] = cp + sy * fsx;
        buffers[RECV][NORTH] = cp;
        buffers[RECV][SOUTH] = cp + (sy + 1) * fsx;

        // Allocate East/West buffers ONCE
        if (!buffers[SEND][EAST]) {
            buffers[SEND][EAST] = (double*)malloc(sy * sizeof(double));
            buffers[RECV][EAST] = (double*)malloc(sy * sizeof(double));
            buffers[SEND][WEST] = (double*)malloc(sy * sizeof(double));
            buffers[RECV][WEST] = (double*)malloc(sy * sizeof(double));
        }

        // Pack non-contiguous East/West data
        #pragma omp parallel for schedule(static)
        for (int j = 1; j <= sy; j++) {
            buffers[SEND][WEST][j-1] = cp[j * fsx + 1];
            buffers[SEND][EAST][j-1] = cp[j * fsx + sx];
        }

        // Non-blocking MPI exchanges
        MPI_Request reqs[8]; int ri = 0;
        if (neighbours[NORTH] != MPI_PROC_NULL) {
            MPI_Isend(buffers[SEND][NORTH], sx, MPI_DOUBLE, neighbours[NORTH], 0, myCOMM_WORLD, &reqs[ri++]);
            MPI_Irecv(buffers[RECV][NORTH], sx, MPI_DOUBLE, neighbours[NORTH], 1, myCOMM_WORLD, &reqs[ri++]);
        }
        if (neighbours[SOUTH] != MPI_PROC_NULL) {
            MPI_Isend(buffers[SEND][SOUTH], sx, MPI_DOUBLE, neighbours[SOUTH], 1, myCOMM_WORLD, &reqs[ri++]);
            MPI_Irecv(buffers[RECV][SOUTH], sx, MPI_DOUBLE, neighbours[SOUTH], 0, myCOMM_WORLD, &reqs[ri++]);
        }
        if (neighbours[WEST] != MPI_PROC_NULL) {
            MPI_Isend(buffers[SEND][WEST], sy, MPI_DOUBLE, neighbours[WEST], 2, myCOMM_WORLD, &reqs[ri++]);
            MPI_Irecv(buffers[RECV][WEST], sy, MPI_DOUBLE, neighbours[WEST], 3, myCOMM_WORLD, &reqs[ri++]);
        }
        if (neighbours[EAST] != MPI_PROC_NULL) {
            MPI_Isend(buffers[SEND][EAST], sy, MPI_DOUBLE, neighbours[EAST], 3, myCOMM_WORLD, &reqs[ri++]);
            MPI_Irecv(buffers[RECV][EAST], sy, MPI_DOUBLE, neighbours[EAST], 2, myCOMM_WORLD, &reqs[ri++]);
        }
        MPI_Waitall(ri, reqs, MPI_STATUSES_IGNORE);
        total_comm_time += MPI_Wtime() - comm_start;

        // Unpack East/West data
        #pragma omp parallel for schedule(static)
        for (int j = 1; j <= sy; j++) {
            cp[j * fsx] = buffers[RECV][WEST][j-1];
            cp[j * fsx + sx + 1] = buffers[RECV][EAST][j-1];
        }

        // === COMPUTATION PHASE ===
        double comp_start = MPI_Wtime();
        #pragma omp parallel
        {
            double thread_start = omp_get_wtime();
            #pragma omp for collapse(2) schedule(static)
            for (int j = 1; j <= sy; j++) {
                for (int i = 1; i <= sx; i++) {
                    int idx = j * fsx + i;
                    planes[!current].data[idx] = cp[idx] + 
                        alpha_dt_dx2 * (cp[idx-1] + cp[idx+1] - 2*cp[idx]) +
                        alpha_dt_dy2 * (cp[idx-fsx] + cp[idx+fsx] - 2*cp[idx]);
                }
            }
            g_per_thread_comp_time[omp_get_thread_num()] += omp_get_wtime() - thread_start;
        }
        total_comp_time += MPI_Wtime() - comp_start;

        current = !current;
        
        if (output_energy_stat_perstep) {
            output_energy_stat(iter, &planes[current], energy_per_source, Rank, &myCOMM_WORLD);
        }
    }

    // Final output
    if (Rank == 0) {
        printf("=== FINAL TIMING RESULTS ===\n");
        printf("Total time: %.3f seconds\n", MPI_Wtime()-t_start);
        printf("Communication time: %.3f seconds (%.1f%%)\n", 
               total_comm_time, total_comm_time/(MPI_Wtime()-t_start)*100);
        printf("Computation time: %.3f seconds (%.1f%%)\n", 
               total_comp_time, total_comp_time/(MPI_Wtime()-t_start)*100);
        printf("\nPer-thread computation times:\n");
        for (int i = 0; i < g_n_omp_threads; i++) {
            printf("  Thread %2d: %.3f seconds\n", i, g_per_thread_comp_time[i]);
        }
    }

    memory_release(planes, &buffers[0]);
    MPI_Finalize();
    return 0;
}

// ================= IMPLEMENTATIONS =================

int inject_energy(int periodic, int Nsources_local, vec2_t *Sources_local, 
                  double energy_per_source, plane_t *plane, vec2_t N) {
    if (!Sources_local) return 0;
    
    int sx = plane->size[_x_], sy = plane->size[_y_];
    int fsx = sx + 2;
    
    for (int s = 0; s < Nsources_local; s++) {
        int i = (int)Sources_local[s][_x_];
        int j = (int)Sources_local[s][_y_];
        if (i >= 0 && i < sx && j >= 0 && j < sy) {
            plane->data[(j+1) * fsx + (i+1)] += energy_per_source;
        }
    }
    return 0;
}

int initialize(MPI_Comm *Comm, int Me, int Ntasks, int argc, char **argv, 
               vec2_t *S, vec2_t *N, int *periodic, int *output_energy_stat_perstep,
               int *neighbours, int *Niterations, int *Nsources, int *Nsources_local, 
               vec2_t **Sources_local, double *energy_per_source, plane_t *planes, 
               buffers_t *buffers) {
    
    // Default parameters
    (*S)[_x_] = 1000; (*S)[_y_] = 1000; 
    *Nsources = 1; *energy_per_source = 1.0; 
    *Niterations = 100; *periodic = 0; *output_energy_stat_perstep = 0;
    
    // Parse command line
    int opt; 
    while ((opt = getopt(argc, argv, "x:y:n:p:s:")) != -1) {
        switch(opt) { 
            case 'x': (*S)[_x_] = atoi(optarg); break; 
            case 'y': (*S)[_y_] = atoi(optarg); break; 
            case 'n': *Niterations = atoi(optarg); break; 
            case 'p': *periodic = atoi(optarg); break; 
            case 's': *Nsources = atoi(optarg); break; 
        }
    }

    // ===== 2D DECOMPOSITION - CRITICAL FIX =====
    int Nx = (int)sqrt(Ntasks);
    while (Nx > 1 && Ntasks % Nx != 0) Nx--;
    int Ny = Ntasks / Nx;
    (*N)[_x_] = Nx;
    (*N)[_y_] = Ny;

    // Calculate neighbors WITH PERIODIC SUPPORT
    int X = Me % Nx, Y = Me / Nx;
    neighbours[NORTH] = (Y > 0) ? Me - Nx : (*periodic ? Me + Nx*(Ny-1) : MPI_PROC_NULL);
    neighbours[SOUTH] = (Y < Ny-1) ? Me + Nx : (*periodic ? Me - Nx*(Ny-1) : MPI_PROC_NULL);
    neighbours[WEST] = (X > 0) ? Me - 1 : (*periodic ? Me + Nx - 1 : MPI_PROC_NULL);
    neighbours[EAST] = (X < Nx-1) ? Me + 1 : (*periodic ? Me - Nx + 1 : MPI_PROC_NULL);

    // Domain decomposition
    planes[OLD].size[_x_] = (*S)[_x_] / Nx;
    planes[OLD].size[_y_] = (*S)[_y_] / Ny;
    planes[NEW].size[_x_] = planes[OLD].size[_x_];
    planes[NEW].size[_y_] = planes[OLD].size[_y_];

    memory_allocate(neighbours, *N, buffers, planes);
    
    // Simple source initialization (center of local domain)
    *Nsources_local = (*Nsources > 0 && Me == 0) ? 1 : 0;
    if (*Nsources_local > 0) {
        *Sources_local = (vec2_t*)malloc(sizeof(vec2_t));
        (*Sources_local)[0][_x_] = planes[OLD].size[_x_] / 2.0;
        (*Sources_local)[0][_y_] = planes[OLD].size[_y_] / 2.0;
    } else {
        *Sources_local = NULL;
    }

    return 0;
}

int memory_allocate (const int neighbours[4], const vec2_t N, buffers_t *buffers_ptr, plane_t *planes_ptr) {
    int sx = planes_ptr[OLD].size[_x_];
    int sy = planes_ptr[OLD].size[_y_];
    int fs = (sx + 2) * (sy + 2) * sizeof(double);
    
    planes_ptr[OLD].data = (double*)malloc(fs); memset(planes_ptr[OLD].data, 0, fs);
    planes_ptr[NEW].data = (double*)malloc(fs); memset(planes_ptr[NEW].data, 0, fs);
    
    // Initialize buffer pointers to NULL (allocated in main loop)
    (*buffers_ptr)[EAST] = NULL;
    (*buffers_ptr)[WEST] = NULL;
    
    return 0;
}

int memory_release (plane_t *planes, buffers_t *buffers) {
    if (planes) { 
        free(planes[OLD].data); 
        free(planes[NEW].data); 
    }
    if (buffers) {
        if ((*buffers)[EAST]) free((*buffers)[EAST]);
        if ((*buffers)[WEST]) free((*buffers)[WEST]);
    }
    if (g_per_thread_comp_time) free(g_per_thread_comp_time);
    return 0;
}

int output_energy_stat(int step, plane_t *plane, double budget, int Me, MPI_Comm *Comm) {
    double se = 0.0, tse = 0.0;
    int sx = plane->size[_x_], sy = plane->size[_y_];
    int fsx = sx + 2;
    
    #pragma omp parallel for reduction(+:se)
    for (int j = 1; j <= sy; j++) {
        for (int i = 1; i <= sx; i++) {
            se += plane->data[j * fsx + i];
        }
    }
    
    MPI_Reduce(&se, &tse, 1, MPI_DOUBLE, MPI_SUM, 0, *Comm);
    if (Me == 0) printf("Step %d: Total Energy %g\n", step, tse);
    return 0;
}

// Stub implementations for compatibility
int simple_factorization(uint A, int *Nf, uint **f) { 
    *Nf = 2;
    *f = (uint*)malloc(2 * sizeof(uint));
    (*f)[0] = (uint)sqrt(A);
    while (A % (*f)[0] != 0) (*f)[0]--;
    (*f)[1] = A / (*f)[0];
    return 0; 
}

int initialize_sources(int Me, int Nt, MPI_Comm *C, vec2_t ms, int Ns, int *Nsl, vec2_t **S) { 
    // Sources are initialized in initialize() for simplicity
    return 0; 
}