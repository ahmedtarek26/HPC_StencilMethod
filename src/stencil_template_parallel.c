#include "stencil_template_parallel.h"

int g_n_omp_threads = 1;
double* g_per_thread_comp_time = NULL;
__thread double thread_local_comp_time = 0.0;

int main(int argc, char **argv)
{
  MPI_Comm myCOMM_WORLD;
  int Rank, Ntasks;
  int neighbours[4];
  int Niterations, periodic, output_energy_stat_perstep;
  vec2_t S, N;
  int Nsources, Nsources_local;
  vec2_t *Sources_local;
  double energy_per_source;
  plane_t planes[2];
  buffers_t buffers[2];

  {
    int level_obtained;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &level_obtained);
    MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
    MPI_Comm_size(MPI_COMM_WORLD, &Ntasks);
    MPI_Comm_dup(MPI_COMM_WORLD, &myCOMM_WORLD);
  }

  #pragma omp parallel
  {
    #pragma omp master
    {
      g_n_omp_threads = omp_get_num_threads();
    }
  }
  g_per_thread_comp_time = (double*)calloc(g_n_omp_threads, sizeof(double));

  initialize(&myCOMM_WORLD, Rank, Ntasks, argc, argv, &S, &N, &periodic, &output_energy_stat_perstep,
             neighbours, &Niterations, &Nsources, &Nsources_local, &Sources_local, &energy_per_source,
             &planes[0], &buffers[0]);

  int current = OLD;
  double t_start = MPI_Wtime();
  double total_comm_time = 0.0, total_comp_time = 0.0;

  for (int iter = 0; iter < Niterations; ++iter) {
    inject_energy(periodic, Nsources_local, Sources_local, energy_per_source, &planes[current], N);
    double section_start = MPI_Wtime();

    double* cp = planes[current].data;
    int sx = planes[current].size[_x_], sy = planes[current].size[_y_];
    int fsx = sx + 2;

    // Buffer pointers
    buffers[SEND][NORTH] = cp + fsx;
    buffers[SEND][SOUTH] = cp + sy * fsx;
    buffers[RECV][NORTH] = cp;
    buffers[RECV][SOUTH] = cp + (sy + 1) * fsx;

    if (buffers[SEND][EAST] == NULL) {
      buffers[SEND][EAST] = (double*)malloc(sy * sizeof(double));
      buffers[RECV][EAST] = (double*)malloc(sy * sizeof(double));
      buffers[SEND][WEST] = (double*)malloc(sy * sizeof(double));
      buffers[RECV][WEST] = (double*)malloc(sy * sizeof(double));
    }

    #pragma omp parallel for
    for (int j = 1; j <= sy; j++) {
      buffers[SEND][WEST][j-1] = cp[j * fsx + 1];
      buffers[SEND][EAST][j-1] = cp[j * fsx + sx];
    }

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
    total_comm_time += MPI_Wtime() - section_start;

    #pragma omp parallel for
    for (int j = 1; j <= sy; j++) {
      cp[j * fsx] = buffers[RECV][WEST][j-1];
      cp[j * fsx + sx + 1] = buffers[RECV][EAST][j-1];
    }

    section_start = MPI_Wtime();
    update_plane(periodic, N, &planes[current], &planes[!current]);
    total_comp_time += MPI_Wtime() - section_start;
    current = !current;
  }

  if (Rank == 0) printf("Total time: %f, Comm: %f, Comp: %f\n", MPI_Wtime()-t_start, total_comm_time, total_comp_time);
  memory_release(planes, &buffers[0]);
  MPI_Finalize();
  return 0;
}

int memory_release (plane_t *planes, buffers_t *buffers) {
  if (planes) { free(planes[OLD].data); free(planes[NEW].data); }
  if (buffers) {
    if ((*buffers)[EAST]) free((*buffers)[EAST]);
    if ((*buffers)[WEST]) free((*buffers)[WEST]);
  }
  return 0;
}

int memory_allocate (const int neighbours[4], const vec2_t N, buffers_t *buffers_ptr, plane_t *planes_ptr) {
  unsigned int fs = (planes_ptr[OLD].size[_x_] + 2) * (planes_ptr[OLD].size[_y_] + 2) * sizeof(double);
  planes_ptr[OLD].data = (double*)malloc(fs); memset(planes_ptr[OLD].data, 0, fs);
  planes_ptr[NEW].data = (double*)malloc(fs); memset(planes_ptr[NEW].data, 0, fs);
  uint sy = planes_ptr[OLD].size[_y_];
  (*buffers_ptr)[EAST] = (double*)malloc(sy * sizeof(double));
  (*buffers_ptr)[WEST] = (double*)malloc(sy * sizeof(double));
  return 0;
}

int output_energy_stat(int step, plane_t *plane, double budget, int Me, MPI_Comm *Comm) {
  double se = 0.0, tse = 0.0; get_total_energy(plane, &se);
  MPI_Reduce(&se, &tse, 1, MPI_DOUBLE, MPI_SUM, 0, *Comm);
  if (Me == 0) printf("Step %d: Energy %g\n", step, tse);
  return 0;
}

int initialize(MPI_Comm *Comm, int Me, int Ntasks, int argc, char **argv, vec2_t *S, vec2_t *N, int *periodic, int *output_energy_stat_perstep, int *neighbours, int *Niterations, int *Nsources, int *Nsources_local, vec2_t **Sources_local, double *energy_per_source, plane_t *planes, buffers_t *buffers) {
  (*S)[_x_] = 1000; (*S)[_y_] = 1000; *Nsources = 1; *energy_per_source = 1.0; *Niterations = 100; *periodic = 0;
  int opt; while ((opt = getopt(argc, argv, "x:y:n:p:")) != -1) {
    switch(opt) { case 'x': (*S)[_x_]=atoi(optarg); break; case 'y': (*S)[_y_]=atoi(optarg); break; case 'n': *Niterations=atoi(optarg); break; case 'p': *periodic=atoi(optarg); break; }
  }
  (*N)[_x_] = 1; (*N)[_y_] = Ntasks;
  int X = Me % (*N)[_x_], Y = Me / (*N)[_x_];
  neighbours[NORTH] = (Y > 0) ? Me - (*N)[_x_] : MPI_PROC_NULL;
  neighbours[SOUTH] = (Y < (*N)[_y_] - 1) ? Me + (*N)[_x_] : MPI_PROC_NULL;
  neighbours[WEST] = MPI_PROC_NULL; neighbours[EAST] = MPI_PROC_NULL;
  planes[OLD].size[_x_] = (*S)[_x_]; planes[OLD].size[_y_] = (*S)[_y_] / Ntasks;
  planes[NEW].size[_x_] = planes[OLD].size[_x_]; planes[NEW].size[_y_] = planes[OLD].size[_y_];
  memory_allocate(neighbours, *N, buffers, planes);
  *Nsources_local = 0;
  return 0;
}

uint simple_factorization(uint A, int *Nf, uint **f) { return 0; }
int initialize_sources(int Me, int Nt, MPI_Comm *C, vec2_t ms, int Ns, int *Nsl, vec2_t **S) { return 0; }
