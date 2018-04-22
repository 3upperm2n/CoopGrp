#include <ctime>
#include <iostream>
#include <string.h>
#include <sstream>
#include <cstring>
#include <string>
#include <cstdio>
#include "common.h"
#include "ga_gpu.h"


using namespace std;
//namespace cg = cooperative_groups;

/*
void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{
	if (n == 1)
	{
		threads = 1;
		blocks = 1;
	}
	else
	{
		threads = (n < maxThreads*2) ? nextPow2(n / 2) : maxThreads;
		blocks = max(1, n / (threads * 2));
	}

	blocks = min(maxBlocks, blocks);
}
*/


int main(int argc, char **argv)
{
	int devID = 0;

	if(argc ==2) {
		devID = atoi(argv[1]);
	}
	printf("select device : %d\n", devID);
	cudaSetDevice(devID);

	cudaError_t error;
	cudaDeviceProp deviceProp;
	int numSms = 0;

	error = cudaGetDeviceProperties(&deviceProp, devID);
	if (error != cudaSuccess)
	{
		printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", 
		cudaGetErrorString(error), error, __LINE__);
	}
	else
	{
		numSms = deviceProp.multiProcessorCount;
		printf("GPU Device %d: \"%s\" with compute capability %d.%d (%d)SMs\n\n", 
		devID, deviceProp.name, deviceProp.major, deviceProp.minor, numSms);
	}

	if (numSms == 0) {
		printf("[ERROR: something is wrong!] numSms = 0\n");
		exit(1);
	}


	// GA parameters
	float prob_mutation  = (float)0.15; // The probability of a mutation
	float prob_crossover = (float)0.8;  // The probability of a crossover
	int world_seed       = 12438955;    // Seed for initial city selection
	int ga_seed          = 87651111;    // Seed for all other random numbers
	
	// World parameters
	int world_width  = 10000;
	int world_height = 10000;
	
	// The test cases
	int iterations          = 1;  // Number of full runs
	const int num_cases     = 1; // How many trials to test
	int cases[num_cases][3] =     // num_cities, pop_size, max_gen
	{
		// {25, 100,    1000},
		// {25, 1000,   1000},
		// {25, 10000,  100},
		// {25, 100000, 10},

		// {50, 100,    1000},
		// {50, 1000,   1000},
		// {50, 10000,  100},
		// {50, 100000, 10},

		// {100, 100,    1000},
		// {100, 1000,   1000},
		// {100, 10000,  100},
		{100, 100000, 10}

	};
	for (int i=0; i<num_cases; i++)
	{
		int num_cities = cases[i][0];
		int pop_size   = cases[i][1];
		int max_gen    = cases[i][2];
		World* world = new World[sizeof(World)];
		make_world(world, world_width, world_height, num_cities, world_seed);

		int pop_bytes  = pop_size * sizeof(World);


		// Tile and grid variables
		int tile_size  = THREADS_PER_BLOCK; // 1024
		int grid_size  = (int)ceil((float)pop_size / tile_size);
		int grid_size2 = (int)ceil((float)(2 * pop_size) / tile_size);

		//=================//
		// 
		//=================//

		cout << "GPU Version - START" << endl;
		for (int j=0; j<iterations; j++)
		{

			if(g_execute(prob_mutation, prob_crossover, pop_size, max_gen,world, ga_seed,
			tile_size, grid_size, grid_size2, pop_bytes, numSms))
				{
					cout<<"GPU Related error - Could be an issue if GPU is being used by others"<<endl
						<<"Please try running again when there is memory free + GPU is free"	<<endl;
				}

		}

		cudaDeviceReset();

		cout << "GPU Version - END" << endl;

		free_world(world);
	}
	
	return 0;
}
