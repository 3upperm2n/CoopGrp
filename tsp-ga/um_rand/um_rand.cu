/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample demonstrates the use of CURAND to generate
 * random numbers on GPU and CPU.
 */

// Utilities and system includes
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


// Utilities and system includes
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper for CUDA Error handling

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>


const int ga_seed = 87651111; 

using namespace std;


__global__ void kern_rng_using_cuRand(float *data, unsigned int seed, int N)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < N)
	{
		curandStateMRG32k3a state;  // prev: not print
		//curandState state;  // now: working! 
		curand_init( seed, tid, 0, &state);

		float a1 = curand_uniform(&state);
		printf("%d: \t %f\n", tid, a1);
		data[tid] = a1;


	}

} 



///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	float *data;	

	int N = 32;
	size_t N_bytes = sizeof(float) * N;

	cudaMallocManaged((void**)&data, N_bytes);

	int blksize = 1024;
	int grdsize = (N + blksize - 1) / blksize;



	//-------------------------------------------------------------------------
	// rng using cuRand 
	//-------------------------------------------------------------------------

	cudaEventRecord(start);

	kern_rng_using_cuRand <<<grdsize, blksize >>>(data, ga_seed, N); 

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float timer_ms = 0.f;
	cudaEventElapsedTime(&timer_ms, start, stop);
	printf("[cuRand] \t %f (ms)\n", timer_ms);

	cudaFree(data);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaDeviceReset();
}

