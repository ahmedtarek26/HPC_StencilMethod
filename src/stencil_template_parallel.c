/*

/*
 *
 *  mysizex   :   local x-extension of your patch
 *  mysizey   :   local y-extension of your patch
 *
 */


#include "stencil_template_parallel.h"

// --- GLOBAL VARIABLE DEFINITIONS ---
// This is where the global variables are actually created.
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
    
    // NOTE: change MPI_FUNNELED if appropriate
    //
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
  
    // --- SETUP FOR PER-THREAD TIMING ---
    // Get the number of threads that will be used.
    #pragma omp parallel
    {
        #pragma omp master
        { g_n_omp_threads = omp_get_num_threads(); }
    }
    // Allocate memory for the global timing array and initialize to zero.
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
  double t_start, t_end;
  double total_comm_time = 0.0;
  double total_comp_time = 0.0;

  
  t_start = MPI_Wtime();


  for (int iter = 0; iter < Niterations; ++iter)
    
    {
      double section_start_time;
 
      
      /* new energy from sources */
      inject_energy( periodic, Nsources_local, Sources_local, energy_per_source, &planes[current], N );
      
      section_start_time = MPI_Wtime();

      /* -------------------------------------- */

      double* current_plane = planes[current].data;
      const int sizex = planes[current].size[_x_];
      const int sizey = planes[current].size[_y_];
      const int full_sizex = sizex + 2;
      const int full_sizey = sizey + 2;

      // [A] fill the buffers, and/or make the buffers' pointers pointing to the correct position

      // North and South buffers can point directly to the data (contiguous lines)
      buffers[SEND][NORTH] = current_plane + full_sizex;  // First inner row
      buffers[SEND][SOUTH] = current_plane + sizey * full_sizex;  // Last inner row

      buffers[RECV][NORTH] = current_plane;  // Halo above first row
      buffers[RECV][SOUTH] = current_plane + (sizey + 1) * full_sizex;  // Halo below last row

      // East and West need allocation and copying (non-contiguous)
      if (buffers[SEND][EAST] == NULL) {
        buffers[SEND][EAST] = (double*)malloc(sizey * sizeof(double));
        buffers[RECV][EAST] = (double*)malloc(sizey * sizeof(double));
        buffers[SEND][WEST] = (double*)malloc(sizey * sizeof(double));
        buffers[RECV][WEST] = (double*)malloc(sizey * sizeof(double));
      }

      // Fill send buffers for East and West
      #pragma omp parallel for
      for (int j = 1; j <= sizey; j++) {
        buffers[SEND][WEST][j-1] = current_plane[j * full_sizex + 1];  // First inner column
        buffers[SEND][EAST][j-1] = current_plane[j * full_sizex + sizex];  // Last inner column
      }

      // [B] perform the halo communications
      // Use non-blocking Isend/Irecv for potential overlap

      MPI_Request reqs[8];
      int req_idx = 0;

      // Send/Recv North
      if (neighbours[NORTH] != MPI_PROC_NULL) {
        MPI_Isend(buffers[SEND][NORTH], sizex, MPI_DOUBLE, neighbours[NORTH], 0, myCOMM_WORLD, &reqs[req_idx++]);
        MPI_Irecv(buffers[RECV][NORTH], sizex, MPI_DOUBLE, neighbours[NORTH], 1, myCOMM_WORLD, &reqs[req_idx++]);
      }

      // Send/Recv South
      if (neighbours[SOUTH] != MPI_PROC_NULL) {
        MPI_Isend(buffers[SEND][SOUTH], sizex, MPI_DOUBLE, neighbours[SOUTH], 1, myCOMM_WORLD, &reqs[req_idx++]);
        MPI_Irecv(buffers[RECV][SOUTH], sizex, MPI_DOUBLE, neighbours[SOUTH], 0, myCOMM_WORLD, &reqs[req_idx++]);
      }

      // Send/Recv West
      if (neighbours[WEST] != MPI_PROC_NULL) {
        MPI_Isend(buffers[SEND][WEST], sizey, MPI_DOUBLE, neighbours[WEST], 2, myCOMM_WORLD, &reqs[req_idx++]);
        MPI_Irecv(buffers[RECV][WEST], sizey, MPI_DOUBLE, neighbours[WEST], 3, myCOMM_WORLD, &reqs[req_idx++]);
      }

      // Send/Recv East
      if (neighbours[EAST] != MPI_PROC_NULL) {
        MPI_Isend(buffers[SEND][EAST], sizey, MPI_DOUBLE, neighbours[EAST], 3, myCOMM_WORLD, &reqs[req_idx++]);
        MPI_Irecv(buffers[RECV][EAST], sizey, MPI_DOUBLE, neighbours[EAST], 2, myCOMM_WORLD, &reqs[req_idx++]);
      }

      // To overlap: Update inner grid while comms happen (if possible, but for small grids may not help)
      // For now, wait immediately; optimize later by splitting update into border/inner.

      MPI_Waitall(req_idx, reqs, MPI_STATUSES_IGNORE);

      total_comm_time += MPI_Wtime() - section_start_time;

      // [C] copy the haloes data (for East/West recv buffers)
      #pragma omp parallel for
      for (int j = 1; j <= sizey; j++) {
        current_plane[j * full_sizex] = buffers[RECV][WEST][j-1];  // Left halo
        current_plane[j * full_sizex + sizex + 1] = buffers[RECV][EAST][j-1];  // Right halo
      }

      // If non-periodic and at boundary, set halo to 0 (infinite sink)
      if (!periodic) {
        if (neighbours[NORTH] == MPI_PROC_NULL) {
          memset(current_plane, 0, full_sizex * sizeof(double));  // Top halo
        }
        if (neighbours[SOUTH] == MPI_PROC_NULL) {
          memset(current_plane + (sizey + 1) * full_sizex, 0, full_sizex * sizeof(double));  // Bottom halo
        }
        if (neighbours[WEST] == MPI_PROC_NULL) {
          for (int j = 0; j < full_sizey; j++) current_plane[j * full_sizex] = 0.0;  // Left halo
        }
        if (neighbours[EAST] == MPI_PROC_NULL) {
          for (int j = 0; j < full_sizey; j++) current_plane[j * full_sizex + sizex + 1] = 0.0;  // Right halo
        }
      }

      /* --------------------------------------  */
      /* update grid points */

      section_start_time = MPI_Wtime();
      update_plane( periodic, N, &planes[current], &planes[!current] );
      total_comp_time += MPI_Wtime() - section_start_time;

      /* output if needed */
      if ( output_energy_stat_perstep )
	output_energy_stat ( iter, &planes[!current], (iter+1) * Nsources*energy_per_source, Rank, &myCOMM_WORLD );
	
      /* swap plane indexes for the new iteration */
      current = !current;
      
    }
  
  t_end = MPI_Wtime() - t_start;

  // Aggregate and print timings (reduce across ranks)
  double max_comm_time, max_comp_time, max_total_time;
  MPI_Reduce(&total_comm_time, &max_comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, myCOMM_WORLD);
  MPI_Reduce(&total_comp_time, &max_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, myCOMM_WORLD);
  MPI_Reduce(&t_end, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, myCOMM_WORLD);

  if (Rank == 0) {
    printf("Total time: %f seconds\n", max_total_time);
    printf("Total communication time: %f seconds\n", max_comm_time);
    printf("Total computation time: %f seconds\n", max_comp_time);
  }

  // Output final energy
  output_energy_stat ( -1, &planes[current], Niterations * Nsources * energy_per_source, Rank, &myCOMM_WORLD );

  memory_release(planes, &buffers[0], Sources_local);
  free(g_per_thread_comp_time);
  MPI_Finalize();
  return 0;
}

// Complete memory_release (3 args to match your call)
int memory_release ( plane_t   *planes,
                     buffers_t *buffers,
                     vec2_t    *sources_local
		     )
{

  if ( planes != NULL )
    {
      if ( planes[OLD].data != NULL )
	free (planes[OLD].data);
      
      if ( planes[NEW].data != NULL )
	free (planes[NEW].data);
    }

  // Free buffers (E/W allocated)
  if ((*buffers)[SEND][EAST] != NULL) free((*buffers)[SEND][EAST]);
  if ((*buffers)[RECV][EAST] != NULL) free((*buffers)[RECV][EAST]);
  if ((*buffers)[SEND][WEST] != NULL) free((*buffers)[SEND][WEST]);
  if ((*buffers)[RECV][WEST] != NULL) free((*buffers)[RECV][WEST]);
  // North/South are pointers, no free

  if (sources_local != NULL) free(sources_local);
      
  return 0;
}

// Complete memory_allocate
int memory_allocate ( const uint    neighbours[4],
		      const vec2_t  N,
		      buffers_t    *buffers_ptr,
		      plane_t      *planes_ptr )
{

  if (planes_ptr == NULL ) {
    fprintf(stderr, "Error: Invalid planes_ptr.\n");
    return 1;
  }

  if (buffers_ptr == NULL ) {
    fprintf(stderr, "Error: Invalid buffers_ptr.\n");
    return 1;
  }
    

  // Allocate planes with halo frame
  unsigned int frame_size = (planes_ptr[OLD].size[_x_]+2) * (planes_ptr[OLD].size[_y_]+2) * sizeof(double);

  planes_ptr[OLD].data = (double*)malloc( frame_size );
  if ( planes_ptr[OLD].data == NULL ) {
    perror("malloc failed for OLD plane");
    return 1;
  }
  memset ( planes_ptr[OLD].data, 0, frame_size );

  planes_ptr[NEW].data = (double*)malloc( frame_size );
  if ( planes_ptr[NEW].data == NULL ) {
    perror("malloc failed for NEW plane");
    return 1;
  }
  memset ( planes_ptr[NEW].data, 0, frame_size );

  // Allocate E/W buffers (sizey doubles)
  uint sizey = planes_ptr[OLD].size[_y_];
  (*buffers_ptr)[SEND][EAST] = (double*)malloc(sizey * sizeof(double));
  (*buffers_ptr)[RECV][EAST] = (double*)malloc(sizey * sizeof(double));
  (*buffers_ptr)[SEND][WEST] = (double*)malloc(sizey * sizeof(double));
  (*buffers_ptr)[RECV][WEST] = (double*)malloc(sizey * sizeof(double));

  // N/S point to plane data (set in loop)

  return 0;
}

int output_energy_stat ( int step, plane_t *plane, double budget, int Me, MPI_Comm *Comm )
{

  double system_energy = 0;
  double tot_system_energy = 0;
  get_total_energy ( plane, &system_energy );
  
  MPI_Reduce ( &system_energy, &tot_system_energy, 1, MPI_DOUBLE, MPI_SUM, 0, *Comm );
  
  if ( Me == 0 )
    {
      if ( step >= 0 )
	printf(" [ step %4d ] ", step ); fflush(stdout);

      
      printf( "total injected energy is %g, "
	      "system energy is %g "
	      "( in avg %g per grid point)\n",
	      budget,
	      tot_system_energy,
	      tot_system_energy / (plane->size[_x_]*plane->size[_y_]) );
    }
  
  return 0;
}

int initialize ( MPI_Comm *Comm,
                 int       Me,
		 int       Ntasks,
		 int       argc,
		 char    **argv,
                 vec2_t   *S,
                 vec2_t   *N,                 
		 int      *periodic,
                 int      *output_energy_stat_perstep,
		 uint     *neighbours,
		 int      *Niterations,
		 int      *Nsources,
		 int      *Nsources_local,
                 vec2_t  **Sources_local,
                 double   *energy_per_source,
                 plane_t  *planes,
                 buffers_t *buffers
		 )
{
  int ret = 0;
  int verbose = 0;

  vec2_t Grid;

  // argument parsing
  int opt;
  double freq = 0.0;

  (*S)[_x_] = 1000;
  (*S)[_y_] = 1000;
  *Nsources = 1;
  *energy_per_source = 1.0;
  *Niterations = 100;
  *periodic = 0;
  *output_energy_stat_perstep = 0;

  while ((opt = getopt(argc, argv, "x:y:e:E:f:n:p:o:v:h")) != -1) {
    switch (opt) {
      case 'x':
        (*S)[_x_] = atoi(optarg);
        break;
      case 'y':
        (*S)[_y_] = atoi(optarg);
        break;
      case 'e':
        *Nsources = atoi(optarg);
        break;
      case 'E':
        *energy_per_source = atof(optarg);
        break;
      case 'f':
        freq = atof(optarg);
        break;
      case 'n':
        *Niterations = atoi(optarg);
        break;
      case 'p':
        *periodic = atoi(optarg);
        break;
      case 'o':
        *output_energy_stat_perstep = atoi(optarg);
        break;
      case 'v':
        verbose = atoi(optarg);
        break;
      case 'h':
        if (Me == 0) {
          printf("usage: %s [options]\n"
                 "options (overriding the default values ):\n"
                 "-x    x size of the plate [1000]\n"
                 "-y    y size of the plate [1000]\n"
                 "-e    how many energy sources on the plate [1]\n"
                 "-E    energy per source [1.0]\n"
                 "-f    the frequency of energy injection [0.0]\n"
                 "-n    how many iterations [100]\n"
                 "-p    whether periodic boundaries applies  [0 = false]\n"
                 "-o    whether to print the energy budget at every step [0 = false]\n"
                 "-v    verbosity level [0]\n",
                 argv[0]);
        }
        MPI_Finalize();
        exit(0);
        break;
      case ':':
        if (Me == 0) printf("option -%c requires an argument\n", optopt);
        MPI_Finalize();
        exit(1);
        break;
      case '?':
        if (Me == 0) printf(" -------- help unavailable ----------\n");
        MPI_Finalize();
        exit(1);
        break;
    }
  }

  // Set injection frequency
  if ( freq == 0 ) {
    // Assume injection_frequency is a variable; set to 1 if not defined
    int injection_frequency = 1;  // Fix if needed
  } else {
    freq = (freq > 1.0 ? 1.0 : freq );
    int injection_frequency = (int)(freq * *Niterations);
  }

  // Parameter validation
  if ((*S)[_x_] <= 0 || (*S)[_y_] <= 0 || *Nsources <= 0 || *Niterations <= 0) {
    if (Me == 0) printf("Error: Invalid parameters.\n");
    MPI_Finalize();
    exit(1);
  }

  // Domain decomposition
  int Nfactors;
  uint *factors;
  simple_factorization(Ntasks, &Nfactors, &factors);

  // Choose square-like decomposition
  Grid[_x_] = 1;
  Grid[_y_] = Ntasks;
  int sqrt_n = (int)sqrt(Ntasks);
  for (int i = sqrt_n; i > 0; i--) {
    if (Ntasks % i == 0) {
      Grid[_x_] = i;
      Grid[_y_] = Ntasks / i;
      break;
    }
  }

  free(factors);

  (*N)[_x_] = Grid[_x_];
  (*N)[_y_] = Grid[_y_];

  // Compute coordinates
  int X = Me % Grid[_x_];
  int Y = Me / Grid[_x_];

  // Set neighbors with periodic
  neighbours[NORTH] = (Y > 0) ? Me - Grid[_x_] : (*periodic ? Me + (Grid[_y_] - 1) * Grid[_x_] : MPI_PROC_NULL);
  neighbours[SOUTH] = (Y < Grid[_y_] - 1) ? Me + Grid[_x_] : (*periodic ? Me - (Grid[_y_] - 1) * Grid[_x_] : MPI_PROC_NULL);
  neighbours[WEST] = (X > 0) ? Me - 1 : (*periodic ? Me + (Grid[_x_] - 1) : MPI_PROC_NULL);
  neighbours[EAST] = (X < Grid[_x_] - 1) ? Me + 1 : (*periodic ? Me - (Grid[_x_] - 1) : MPI_PROC_NULL);

  // Local size
  vec2_t mysize;
  uint s = (*S)[_x_] / Grid[_x_];
  uint r = (*S)[_x_] % Grid[_x_];
  mysize[_x_] = s + (X < r ? 1 : 0);
  s = (*S)[_y_] / Grid[_y_];
  r = (*S)[_y_] % Grid[_y_];
  mysize[_y_] = s + (Y < r ? 1 : 0);

  planes[OLD].size[_x_] = mysize[_x_];
  planes[OLD].size[_y_] = mysize[_y_];
  planes[NEW].size[_x_] = mysize[_x_];
  planes[NEW].size[_y_] = mysize[_y_];

  if (verbose > 0 && Me == 0) {
    printf("Grid: %d x %d\n", Grid[_x_], Grid[_y_]);
  }

  // Allocate memory
  ret = memory_allocate(neighbours, *N, buffers, planes);
  if (ret) return ret;

  // Initialize sources
  ret = initialize_sources( Me, Ntasks, Comm, mysize, *Nsources, Nsources_local, Sources_local );
  if (ret) return ret;

  return 0;
}


uint simple_factorization( uint A, int *Nfactors, uint **factors )
{
  int N = 0;
  int f = 2;
  uint _A_ = A;

  while ( f * f <= _A_ ) {
    while( _A_ % f == 0 ) {
	N++;
	_A_ /= f; }
    f++;
  }
  if (_A_ > 1) N++;

  *Nfactors = N;
  uint *_factors_ = (uint*)malloc( N * sizeof(uint) );

  N = 0;
  f = 2;
  _A_ = A;

  while ( f * f <= _A_ ) {
    while( _A_ % f == 0 ) {
	_factors_[N++] = f;
	_A_ /= f; }
    f++;
  }
  if (_A_ > 1) _factors_[N++] = _A_;

  *factors = _factors_;
  return 0;
}


int initialize_sources( int       Me,
			int       Ntasks,
			MPI_Comm *Comm,
			vec2_t    mysize,
			int       Nsources,
			int      *Nsources_local,
			vec2_t  **Sources )
{

  srand48(time(NULL) ^ Me);
  int *tasks_with_sources = (int*)malloc( Nsources * sizeof(int) );
  
  if ( Me == 0 ) {
    for ( int i = 0; i < Nsources; i++ )
	tasks_with_sources[i] = (int)lrand48() % Ntasks;
  }
  
  MPI_Bcast( tasks_with_sources, Nsources, MPI_INT, 0, *Comm );

  int nlocal = 0;
  for ( int i = 0; i < Nsources; i++ )
    nlocal += (tasks_with_sources[i] == Me);
  *Nsources_local = nlocal;
  
  if ( nlocal > 0 ) {
    vec2_t * helper = (vec2_t*)malloc( nlocal * sizeof(vec2_t) );      
    for ( int s = 0; s < nlocal; s++ ) {
	  helper[s][_x_] = 1 + lrand48() % mysize[_x_];
	  helper[s][_y_] = 1 + lrand48() % mysize[_y_];
	}

    *Sources = helper;
  }
  
  free( tasks_with_sources );

  return 0;
}