/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <time.h>
#include <math.h>

#include <omp.h>
#include <mpi.h>


#define NORTH 0
#define SOUTH 1
#define EAST  2
#define WEST  3

#define SEND 0
#define RECV 1

#define OLD 0
#define NEW 1

#define _x_ 0
#define _y_ 1

typedef unsigned int uint;

typedef uint    vec2_t[2];
typedef double *restrict buffers_t[4];

typedef struct {
    double   * restrict data;
    vec2_t     size;
} plane_t;



// --- GLOBAL VARIABLES FOR PER-THREAD TIMING ---
extern int g_n_omp_threads;
extern double* g_per_thread_comp_time;

// Thread-local timing variables for more accurate measurement
extern __thread double thread_local_comp_time;

// Function declarations
extern int inject_energy ( const int      ,
                           const int      ,
			   const vec2_t  *,
			   const double   ,
                                 plane_t *,
                           const vec2_t   );


extern int update_plane ( const int      ,
                          const vec2_t   ,
                          const plane_t *,
                                plane_t * );


extern int get_total_energy( plane_t *,
                             double  * );

int initialize ( MPI_Comm *,
                 int       ,
		 int       ,
		 int       ,
		 char    **,
                 vec2_t   *,
                 vec2_t   *,                 
		 int      *,
                 int      *,
		 int      *,
		 int      *,
		 int      *,
		 int      *,
                 vec2_t  **,
                 double   *,
                 plane_t  *,
                 buffers_t * );


int memory_release (plane_t   *, buffers_t * );

int memory_allocate (const int *, const vec2_t, buffers_t *, plane_t *);

int output_energy_stat ( int      ,
                         plane_t *,
                         double   ,
                         int      ,
                         MPI_Comm *);

// Additional function declarations that were missing
uint simple_factorization( uint, int *, uint ** );

int initialize_sources( int       ,
			int       ,
			MPI_Comm  *,
			uint      [2],
			int       ,
			int      *,
			vec2_t  ** );



inline int inject_energy ( const int      periodic,
                           const int      Nsources,
			   const vec2_t  *Sources,
			   const double   energy,
                                 plane_t *plane,
                           const vec2_t   N
                           )
{
    const uint register sizex = plane->size[_x_]+2;
    double * restrict data = plane->data;
    
   #define IDX( i, j ) ( (j)*sizex + (i) )
    for (int s = 0; s < Nsources; s++)
        {
            int x = Sources[s][_x_];
            int y = Sources[s][_y_];
            
            data[ IDX(x,y) ] += energy;
            
            if ( periodic )
                {
                    if ( (N[_x_] == 1)  )
                        {
                            // If a source is on the left edge (x=1)...
                            if (x == 1) {
                                // ...add energy to the right ghost cell.
                                data[IDX(plane->size[_x_] + 1, y)] += energy;
                            }
                            // If a source is on the right edge...
                            if (x == plane->size[_x_]) {
                                // ...add energy to the left ghost cell.
                                data[IDX(0, y)] += energy;
                            }
                        }
                    
                    if ( (N[_y_] == 1) )
                        {
                            // If a source is on the bottom edge (y=1)...
                            if (y == 1) {
                                // ...add energy to the top ghost cell.
                                data[IDX(x, plane->size[_y_] + 1)] += energy;
                            }
                            // If a source is on the top edge...
                            if (y == plane->size[_y_]) {
                                // ...add energy to the bottom ghost cell.
                                data[IDX(x, 0)] += energy;
                            }
                        }
                }                
        }
 #undef IDX
    
  return 0;
}





inline int update_plane ( const int      periodic, 
                          const vec2_t   N,         // the grid of MPI tasks
                          const plane_t *oldplane,
                                plane_t *newplane
                          )
    
{
    uint register fxsize = oldplane->size[_x_]+2;
    uint register fysize = oldplane->size[_y_]+2;
    
    uint register xsize = oldplane->size[_x_];
    uint register ysize = oldplane->size[_y_];
    
   #define IDX( i, j ) ( (j)*fxsize + (i) )
    
    // HINT: you may attempt to
    //       (i)  manually unroll the loop
    //       (ii) ask the compiler to do it
    // for instance
    //#pragma GCC unroll 4
    //
    // HINT: in any case, this loop is a good candidate
    //       for openmp parallelization
    double * restrict old = oldplane->data;
    double * restrict new = newplane->data;

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        double t0 = omp_get_wtime();
        
        #pragma omp for collapse(2) schedule(static)
    	for (uint j = 1; j <= ysize; j++){
        	for ( uint i = 1; i <= xsize; i++)
       		{ 
                // NOTE: (i-1,j), (i+1,j), (i,j-1) and (i,j+1) always exist even
                //       if this patch is at some border without periodic conditions;
                //       in that case it is assumed that the +-1 points are outside the
                //       plate and always have a value of 0, i.e. they are an
                //       "infinite sink" of heat
                
                // five-points stencil formula
                //
                // HINT : check the serial version for some optimization
                //

                double alpha = 0.6;
                double result = old[IDX(i,j)] * alpha;
                double sum_i  = (old[IDX(i-1, j)] + old[IDX(i+1, j)]) / 4.0 * (1.0-alpha);
                double sum_j  = (old[IDX(i, j-1)] + old[IDX(i, j+1)]) / 4.0 * (1.0-alpha);
                result += (sum_i + sum_j );
                new[IDX(i,j)] = result;
            }
        }
        double t1 = omp_get_wtime();
        #pragma omp atomic
        g_per_thread_comp_time[thread_id] += (t1 - t0);
    }

    if ( periodic )
    {
        // If the task grid is only 1 process wide...
        if ( N[_x_] == 1 )
        {
            // ...propagate the East/West boundaries locally.
            for ( int j = 1; j <= ysize; j++ ) {
                new[ IDX(0, j) ] = new[ IDX(xsize, j) ];
                new[ IDX(xsize + 1, j) ] = new[ IDX(1, j) ];
            }
        }
  
        // If the task grid is only 1 process tall...
        if ( N[_y_] == 1 ) 
        {
            // ...propagate the North/South boundaries locally.
            for ( int i = 1; i <= xsize; i++ ) {
                new[ IDX(i, 0) ] = new[ IDX(i, ysize) ];
                new[ IDX(i, ysize + 1) ] = new[ IDX(i, 1) ];
            }
        }
    }

 #undef IDX
  return 0;
}



inline int get_total_energy( plane_t *plane,
                             double  *energy )
/*
 * NOTE: this routine a good candiadate for openmp
 *       parallelization
 */
{
    const int register xsize = plane->size[_x_];
    const int register ysize = plane->size[_y_];
    const int register fsize = xsize+2;

    double * restrict data = plane->data;

   #define IDX( i, j ) ( (j)*fsize + (i) )

   #if defined(LONG_ACCURACY)    
    long double totenergy = 0;
   #else
    double totenergy = 0;    
   #endif
   
   #pragma omp parallel
   {
       int thread_id = omp_get_thread_num();
       double t0 = omp_get_wtime();
       
       #pragma omp for collapse(2) reduction(+:totenergy) schedule(static)
        for ( int j = 1; j <= ysize; j++ ){
            for ( int i = 1; i <= xsize; i++ ){
                totenergy += data[ IDX(i, j) ];
            }
       } 
       double t1 = omp_get_wtime();
       #pragma omp atomic
       g_per_thread_comp_time[thread_id] += (t1 - t0);
   }
   
   #undef IDX

    *energy = (double)totenergy;
    return 0;
}



