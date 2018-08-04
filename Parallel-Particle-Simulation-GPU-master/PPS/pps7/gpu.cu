#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include "common.h"
//#include <vector>

#define NUM_THREADS 256

//typedef vector< particle_t* > gridBin;
//typedef vector< gridBin > particleGrid;

using namespace std;

extern double size;

//
//  benchmarking program
//

__device__ void apply_force_gpu( particle_t &particle, particle_t &neighbor )
{
	double dx = neighbor.x - particle.x;
	double dy = neighbor.y - particle.y;
  	double r2 = dx * dx + dy * dy;

	if( r2 > cutoff*cutoff )
        {
      		return;
	}

  	//r2 = fmax( r2, min_r*min_r );
  	r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;

	double r = sqrt( r2 );

  	//
  	//  very simple short-range repulsive force
  	//
  	double coef = ( 1 - cutoff / r ) / r2 / mass;
  	particle.ax += coef * dx;
  	particle.ay += coef * dy;

}

__global__ void compute_forces_gpu( particle_t * particles, int n )
{
  	// Get thread (particle) ID
  	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if( tid >= n )
	{
		return;
	}

  	particles[ tid ].ax = particles[ tid ].ay = 0;

	for( int j = 0 ; j < n ; j++ )
	{
    		apply_force_gpu( particles[ tid ], particles[ j ] );
	}

}

__global__ void move_gpu ( particle_t * particles, int n, double size )
{

  	// Get thread (particle) ID
  	int tid = threadIdx.x + blockIdx.x * blockDim.x;
  	if( tid >= n )
	{
		return;
	}
  
	particle_t * p = &particles[ tid ];

	//
    	//  slightly simplified Velocity Verlet integration
    	//  conserves energy better than explicit Euler method
    	//
	float pvx = p->vx;
	float pvy = p->vy;
	float px = p->x;
	float py = p->y;
	float pax = p->ax;
	float pay = p->ay;

	pvx += pax * dt;
    	pvy += pay * dt;
    	px  += pvx * dt;
    	py  += pvy * dt;

    	//
   	//  bounce from walls
    	//
    	while( px < 0 || px > size )
    	{
        	px  = px < 0 ? -( px ) : 2 * size - px;
        	pvx = -( pvx );
    	}

	while( py < 0 || py > size )
    	{
        	py  = py < 0 ? -( py ) : 2 * size - py;
        	pvy = -( pvy );
    	}
        p->vx = pvx:
	p->vy = pvy;
	p->x = px;
	p->y = py;

}



int main( int argc, char **argv )
{
    	// This takes a few seconds to initialize the runtime
    	//cudaThreadSynchronize();

    	if( find_option( argc, argv, "-h" ) >= 0 )
    	{
        	printf( "Options:\n" );
        	printf( "-h to see this help\n" );
        	printf( "-n <int> to set the number of particles\n" );
        	printf( "-o <filename> to specify the output file name\n" );

        	return 0;
    	}

    	int n = read_int( argc, argv, "-n", 1000 );

    	char *savename = read_string( argc, argv, "-o", NULL );

    	FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    	particle_t *particles;

	if( ( particles = ( particle_t* ) malloc( n * sizeof( particle_t ) ) ) == NULL )
	{
		fprintf( stderr, "particle malloc NULL at line %d in file %s\n", __LINE__, __FILE__ );

		return -1;
	}

    	// GPU particle data structure
    	particle_t * d_particles;
    	cudaMalloc( ( void ** ) &d_particles, n * sizeof( particle_t ) );

    	set_size( n );

    	init_particles( n, particles );

    	//cudaThreadSynchronize( );

	double copy_time = read_timer( );

    	// Copy the particles to the GPU
    	cudaMemcpy( d_particles, particles, n * sizeof( particle_t ), cudaMemcpyHostToDevice );

    	//cudaThreadSynchronize( );
    	copy_time = read_timer( ) - copy_time;

    	//
    	//  simulate a number of time steps
    	//
    	//cudaThreadSynchronize();
    	double simulation_time = read_timer( );

    	for( int step = 0; step < NSTEPS; step++ )
    	{
    		//
        	//  compute forces
        	//

		int blks = ( n + NUM_THREADS - 1 ) / NUM_THREADS;
		compute_forces_gpu <<< blks, NUM_THREADS >>> ( d_particles, n );

        	//
        	//  move particles
        	//
		move_gpu <<< blks, NUM_THREADS >>> ( d_particles, n, size );

        	//
        	//  save if necessary
        	//
        	if( fsave && ( step%SAVEFREQ ) == 0 )
		{
		    	// Copy the particles back to the CPU
            		cudaMemcpy( particles, d_particles, n * sizeof( particle_t ), cudaMemcpyDeviceToHost );
            		save( fsave, n, particles );
		}
    	}

	//cudaThreadSynchronize( );
    	simulation_time = read_timer( ) - simulation_time;

    	printf( "CPU-GPU copy time = %g seconds\n", copy_time );
    	printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );

    	free( particles );
    	cudaFree( d_particles );

	if( fsave )
	{
        	fclose( fsave );
    	}

    	return 0;
}
