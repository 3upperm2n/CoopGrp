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


#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
//#include <thrust/transform_reduce.h>


using namespace std;

//-----------------------------------------------------------------------------
// rng using thrust
// https://github.com/thrust/thrust/blob/master/examples/monte_carlo.cu
//-----------------------------------------------------------------------------

__host__ __device__
unsigned int hash(unsigned int a)
{
	a = (a+0x7ed55d16) + (a<<12);
	a = (a^0xc761c23c) ^ (a>>19);
	a = (a+0x165667b1) + (a<<5);
	a = (a+0xd3a2646c) ^ (a<<9);
	a = (a+0xfd7046c5) + (a<<3);
	a = (a^0xb55a4f09) ^ (a>>16);
	return a;
}

__host__ __device__ float thrust_rng(unsigned int thread_id)
{
	unsigned int seed = hash(thread_id);

	// seed a random number generator
	thrust::default_random_engine rng(seed);

	// create a mapping from random numbers to [0,1)
	thrust::uniform_real_distribution<float> u01(0,1);

	return u01(rng);
}


__global__ void kern_rng_using_thrust(float *data)
{
	uint tid = threadIdx.x + blockIdx.x * blockDim.x;
	data[tid] = thrust_rng(tid);
}



__global__ void kern_rng_using_cuRand(float *data, const unsigned int seed)                                
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	//curandState_t state;
	curandStateMRG32k3a state;
		                                                                                
	// Initialise the RNG                                                       
	curand_init(seed, tid, 0, &state);                                 

	data[tid] = curand_uniform(&state);

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

	int N = 10000;
	size_t N_bytes = sizeof(float) * N;

	cudaMallocManaged((void**)&data, N_bytes);



	//-------------------------------------------------------------------------
	// rng using thrust
	//-------------------------------------------------------------------------
	int blksize = 1024;
	int grdsize = (N + blksize - 1) / blksize;

	//cout << grdsize.x << endl;
	//cout << blksize.x << endl;


	cudaEventRecord(start);

	kern_rng_using_thrust <<< grdsize, blksize >>> (data);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float timer_ms = 0.f;
	cudaEventElapsedTime(&timer_ms, start, stop);
	printf("[thrust] \t %f (ms)\n", timer_ms);

	//cudaDeviceSynchronize();
	//cout << data[0] << " " << data[1] << endl;

	//-------------------------------------------------------------------------
	// rng using cuRand 
	//-------------------------------------------------------------------------


	cudaEventRecord(start);

	kern_rng_using_cuRand <<<grdsize, blksize >>>(data, ga_seed); 

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	timer_ms = 0.f;
	cudaEventElapsedTime(&timer_ms, start, stop);
	printf("[cuRand] \t %f (ms)\n", timer_ms);

	//cudaDeviceSynchronize();
	//cout << data[0] << " " << data[1] << endl;


	cudaFree(data);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaDeviceReset();

}

