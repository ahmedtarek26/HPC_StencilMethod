/*

 *
 *  mysizex   :   local x-extension of your patch
 *  mysizey   :   local y-extension of your patch
 *
 */


#include "stencil_template_parallel.h"

// --- GLOBAL VARIABLE DEFINITIONS ---
int g_n_omp_threads = 1;
double* g_per_thread_comp_time = NULL;
__thread double thread_local_comp_time = 0.0;

// ------------------------------------------------------------------
// ------------------------------------------------------------------

int main(int argc, char **argv)
{
  MPI_Comm myCOMM_WORLD;
  int  Rank, Ntasks;
  uint neighbours[4];

  int  Niterations;
  int  periodic;
  vec2_t S, N;
  
  int      Nsources;
  int      Nsources_local;
  vec2_t  *Sources_local;
  double   energy_per_source;

  plane_t   planes[2];  
  buffers_t buffers[2];
  
  int output_energy_stat_perstep;
  
  /* initialize MPI environment */
  {
    int level_obtained;
    
    MPI_Init_thread( &argc, &argv, MPI_THREAD_FUNNELED, &level_obtained );
    if ( level_obtained < MPI_THREAD_FUNNELED ) {
      printf("MPI_thread level obtained is %d instead of %d\n",
	     level_obtained, MPI_THREAD_FUNNELED );
      MPI_Finalize();
      exit(1); }
    
    MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
    MPI_Comm_size(MPI_COMM_WORLD, &Ntasks);
    MPI_Comm_dup (MPI_COMM_WORLD, &myCOMM_WORLD);
  }
  
  #pragma omp parallel
  {
    #pragma omp master
    { g_n_omp_threads = omp_get_num_threads(); }
  }
  g_per_thread_comp_time = (double*)calloc(g_n_omp_threads, sizeof(double));

  /* argument checking and setting */
  int ret = initialize ( &myCOMM_WORLD, Rank, Ntasks, argc, argv, &S, &N, &periodic, &output_energy_stat_perstep,
			 neighbours, &Niterations,
			 &Nsources, &Nsources_local, &Sources_local, &energy_per_source,
			 &planes[0], &buffers[0] );

  if ( ret )
    {
      printf("task %d is opting out with termination code %d\n",
	     Rank, ret );
      
      MPI_Finalize();
      return 0;
    }
  
  int current = OLD;
  double t_start = MPI_Wtime();
  double total_comm_time = 0.0;
  double total_comp_time = 0.0;

  for (int iter = 0; iter < Niterations; ++iter)
    {
      double section_start_time;

      inject_energy( periodic, Nsources_local, Sources_local, energy_per_source, &planes[current], N );
      
      section_start_time = MPI_Wtime();

      double* current_plane = planes[current].data;
      const int sizex = planes[current].size[_x_];
      const int sizey = planes[current].size[_y_];
      const int full_sizex = sizex + 2;
      const int full_sizey = sizey + 2;

      // Pointers for N/S (contiguous)
      buffers[SEND][NORTH] = current_plane + full_sizex;
      buffers[SEND][SOUTH] = current_plane + sizey * full_sizex;
      buffers[RECV][NORTH] = current_plane;
      buffers[RECV][SOUTH] = current_plane + (sizey + 1) * full_sizex;

      // Allocate E/W buffers only once
      if (buffers[SEND][EAST] == NULL) {
        buffers[SEND][EAST] = malloc(sizey * sizeof(double));
        buffers[RECV][EAST] = malloc(sizey * sizeof(double));
        buffers[SEND][WEST] = malloc(sizey * sizeof(double));
        buffers[RECV][WEST] = malloc(sizey * sizeof(double));
      }

      // Fill E/W send buffers
      #pragma omp parallel for
      for (int j = 1; j <= sizey; j++) {
        buffers[SEND][WEST][j-1] = current_plane[j * full_sizex + 1];
        buffers[SEND][EAST][j-1] = current_plane[j * full_sizex + sizex];
      }

      MPI_Request reqs[8];
      int req_idx = 0;

      if (neighbours[NORTH] != MPI_PROC_NULL) {
        MPI_Isend(buffers[SEND][NORTH], sizex, MPI_DOUBLE, neighbours[NORTH], 0, myCOMM_WORLD, &reqs[req_idx++]);
        MPI_Irecv(buffers[RECV][NORTH], sizex, MPI_DOUBLE, neighbours[NORTH], 1, myCOMM_WORLD, &reqs[req_idx++]);
      }
      if (neighbours[SOUTH] != MPI_PROC_NULL) {
        MPI_Isend(buffers[SEND][SOUTH], sizex, MPI_DOUBLE, neighbours[SOUTH], 1, myCOMM_WORLD, &reqs[req_idx++]);
        MPI_Irecv(buffers[RECV][SOUTH], sizex, MPI_DOUBLE, neighbours[SOUTH], 0, myCOMM_WORLD, &reqs[req_idx++]);
      }
      if (neighbours[WEST] != MPI_PROC_NULL) {
        MPI_Isend(buffers[SEND][WEST], sizey, MPI_DOUBLE, neighbours[WEST], 2, myCOMM_WORLD, &reqs[req_idx++]);
        MPI_Irecv(buffers[RECV][WEST], sizey, MPI_DOUBLE, neighbours[WEST], 3, myCOMM_WORLD, &reqs[req_idx++]);
      }
      if (neighbours[EAST] != MPI_PROC_NULL) {
        MPI_Isend(buffers[SEND][EAST], sizey, MPI_DOUBLE, neighbours[EAST], 3, myCOMM_WORLD, &reqs[req_idx++]);
        MPI_Irecv(buffers[RECV][EAST], sizey, MPI_DOUBLE, neighbours[EAST], 2, myCOMM_WORLD, &reqs[req_idx++]);
      }

      MPI_Waitall(req_idx, reqs, MPI_STATUSES_IGNORE);

      total_comm_time += MPI_Wtime() - section_start_time;

      // Copy received E/W halos
      #pragma omp parallel for
      for (int j = 1; j <= sizey; j++) {
        current_plane[j * full_sizex] = buffers[RECV][WEST][j-1];
        current_plane[j * full_sizex + sizex + 1] = buffers[RECV][EAST][j-1];
      }

      // Non-periodic boundary: zero halo (infinite sink)
      if (!periodic) {
        if (neighbours[NORTH] == MPI_PROC_NULL) memset(current_plane, 0, full_sizex * sizeof(double));
        if (neighbours[SOUTH] == MPI_PROC_NULL) memset(current_plane + (sizey + 1) * full_sizex, 0, full_sizex * sizeof(double));
        if (neighbours[WEST] == MPI_PROC_NULL) {
          for (int j = 0; j < full_sizey; j++) current_plane[j * full_sizex] = 0.0;
        }
        if (neighbours[EAST] == MPI_PROC_NULL) {
          for (int j = 0; j < full_sizey; j++) current_plane[j * full_sizex + sizex + 1] = 0.0;
        }
      }

      section_start_time = MPI_Wtime();
      update_plane( periodic, N, &planes[current], &planes[!current] );
      total_comp_time += MPI_Wtime() - section_start_time;

      if ( output_energy_stat_perstep )
	output_energy_stat ( iter, &planes[!current], (iter+1) * Nsources*energy_per_source, Rank, &myCOMM_WORLD );
	
      current = !current;
    }

  double t_end = MPI_Wtime() - t_start;

  double max_comm_time, max_comp_time, max_total_time;
  MPI_Reduce(&total_comm_time, &max_comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, myCOMM_WORLD);
  MPI_Reduce(&total_comp_time, &max_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, myCOMM_WORLD);
  MPI_Reduce(&t_end, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, myCOMM_WORLD);

  if (Rank == 0) {
    printf("Total time: %f seconds\n", max_total_time);
    printf("Total communication time: %f seconds\n", max_comm_time);
    printf("Total computation time: %f seconds\n", max_comp_time);
  }

  output_energy_stat ( -1, &planes[current], Niterations * Nsources * energy_per_source, Rank, &myCOMM_WORLD );

  memory_release(planes, &buffers[0]);
  if (Sources_local) free(Sources_local);
  free(g_per_thread_comp_time);
  MPI_Finalize();
  return 0;
}

/* Required supporting functions - must be in .c file */

int memory_release(plane_t *planes, buffers_t *buffers)
{
  if (planes) {
    if (planes[OLD].data) free(planes[OLD].data);
    if (planes[NEW].data) free(planes[NEW].data);
  }

  if (buffers) {
    if ((*buffers)[SEND][EAST]) free((*buffers)[SEND][EAST]);
    if ((*buffers)[RECV][EAST]) free((*buffers)[RECV][EAST]);
    if ((*buffers)[SEND][WEST]) free((*buffers)[SEND][WEST]);
    if ((*buffers)[RECV][WEST]) free((*buffers)[RECV][WEST]);
  }

  return 0;
}

int memory_allocate(const uint neighbours[4], const vec2_t N, buffers_t *buffers_ptr, plane_t *planes_ptr)
{
  if (!planes_ptr || !buffers_ptr) return 1;

  unsigned int frame_size = (planes_ptr[OLD].size[_x_] + 2) * (planes_ptr[OLD].size[_y_] + 2) * sizeof(double);

  planes_ptr[OLD].data = malloc(frame_size);
  if (!planes_ptr[OLD].data) return 1;
  memset(planes_ptr[OLD].data, 0, frame_size);

  planes_ptr[NEW].data = malloc(frame_size);
  if (!planes_ptr[NEW].data) return 1;
  memset(planes_ptr[NEW].data, 0, frame_size);

  uint sizey = planes_ptr[OLD].size[_y_];
  (*buffers_ptr)[SEND][EAST] = malloc(sizey * sizeof(double));
  (*buffers_ptr)[RECV][EAST] = malloc(sizey * sizeof(double));
  (*buffers_ptr)[SEND][WEST] = malloc(sizey * sizeof(double));
  (*buffers_ptr)[RECV][WEST] = malloc(sizey * sizeof(double));

  return 0;
}

int output_energy_stat(int step, plane_t *plane, double budget, int Me, MPI_Comm *Comm)
{
  double system_energy = 0.0;
  double tot_system_energy = 0.0;

  get_total_energy(plane, &system_energy);

  MPI_Reduce(&system_energy, &tot_system_energy, 1, MPI_DOUBLE, MPI_SUM, 0, *Comm);

  if (Me == 0) {
    if (step >= 0) printf(" [ step %4d ] ", step);
    printf("total injected energy is %g, system energy is %g (avg %g per point)\n",
           budget, tot_system_energy, tot_system_energy / (plane->size[_x_] * plane->size[_y_]));
  }

  return 0;
}