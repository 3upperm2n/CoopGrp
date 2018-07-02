// unoptimized implementation of single source shortest path
// currently only tracks path cost - not route for shortest path
// based on pseudocode from Singh et al.'s Efficient Parallel Implementation
// of Single Source Shortest Path Algorithm on GPU Using CUDA,
// specifically Algorithm 2 / PNBA variant

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <map> 
#include <set>
#include <limits>
#include <cooperative_groups.h>
#include "graphReader.h"

#define THREADS_PER_BLOCK 1024

using namespace std;
using namespace cooperative_groups;

extern "C" __global__ void SSSPCompleteKernel(float *nodeWeights, 
	  				      int *flags,
				   	      int *vertices,
				   	      int *edges,
				   	      float *weights,
				   	      int *lock, 
				   	      int source, 
				   	      int numNodes,
				   	      int numEdges)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	//extern __shared__ int flags[];
	grid_group my_grid = this_grid();	
	atomicExch(lock, 1); 

        for (int j = tid; j < numNodes; j += my_grid.size())
	{	
		if (tid == source)
		{
			nodeWeights[j] = 0.0f;
			flags[j] = 1;
		}
		else
		{
			nodeWeights[tid] = 10000000.0f;
			flags[j] = 0;
		}
	}	

	my_grid.sync();
	
	while(*lock == 1)
	{
		my_grid.sync();
		atomicExch(lock, 0); 

		my_grid.sync();
	        for (int j = tid; j < numNodes; j += my_grid.size())	
		//if (tid < numNodes)
		{
			if (flags[j] == 1)
			{
				flags[j] = 0;
				int numNeighbors = (j != numNodes - 1) ? (vertices[j+1]-vertices[tid]) : (numEdges-vertices[j]);
				for (int i = 0; i < numNeighbors; i++)
                        	{
                                	if (nodeWeights[edges[vertices[j]+i]] > (nodeWeights[j] + weights[vertices[j] + i]))
                                	{
						nodeWeights[edges[vertices[j]+ i]] = nodeWeights[j] + weights[vertices[j] + i];
//		                             	atomicAdd(&nodeWeights[edges[vertices[tid]+i]], -nodeWeights[edges[vertices[tid]+i]] + (nodeWeights[tid] + weights[vertices[tid]+i]));
                                        	*lock = 1;
                                        	flags[edges[vertices[j]+i]] = 1;
                                	}
                        	}	
			}
		}

		my_grid.sync();
	}
}

void checkCudaErrors(cudaError_t error)
{
	if (error != 0)
	{
		cout << "Cuda error: " << cudaGetErrorString(error) << endl;
		throw runtime_error("CUDA Failure");	
	}
}

void getLastCudaError(string failureMessage)
{
	cudaError_t error = cudaPeekAtLastError();
	if (error != 0)
	{
		cout << "Cuda error: " << cudaGetErrorString(error) << endl;
		throw runtime_error(failureMessage);
	}
}

float runSSSP(int *vertices,
	     	       int *edges,
	     	       float *weights,
	     	       int numNodes,
	     	       int numEdges,
	     	       int source,
		       std::map<int, long> &vertexMap)
{
	int *devVertices, *devEdges, *devFlags, *devLock;
	float *devWeights, *devNodeWeights;

	size_t nodeSize = numNodes*sizeof(int);
	size_t weightNodeSize = numNodes*sizeof(float);
	size_t edgeSize = numEdges*sizeof(int);
	size_t weightSize = numEdges*sizeof(float);

	checkCudaErrors(cudaSetDevice(0));

	// make unified memory
	checkCudaErrors(cudaMallocManaged( (void **)&devVertices, nodeSize));
	checkCudaErrors(cudaMallocManaged( (void **)&devEdges, edgeSize));
	checkCudaErrors(cudaMallocManaged( (void **)&devWeights, weightSize));
	checkCudaErrors(cudaMallocManaged( (void **)&devFlags, nodeSize));
	checkCudaErrors(cudaMallocManaged( (void **)&devNodeWeights, weightNodeSize));
	checkCudaErrors(cudaMallocManaged( (void **)&devLock, sizeof(int)));

	checkCudaErrors(cudaMemcpy(devVertices, vertices, nodeSize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(devEdges, edges, edgeSize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(devWeights, weights, weightSize, cudaMemcpyHostToDevice));

	// initialize values to allow for repeated runs
	*devLock = 1;

	// set up the runs and timer variables
	int num_blocks = 1 + ((numNodes - 1)/THREADS_PER_BLOCK);

	dim3 threadsPerBlock(THREADS_PER_BLOCK, 1);
	dim3 numBlocks(num_blocks, 1);

	printf("Num Blocks: %d \n", num_blocks);	

	float total_time = 0.0;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int sharedMem = 0; //nodeSize;
	void *kernelArgs[] = {
		(void*)&devNodeWeights,
		(void*)&devFlags,
		(void*)&devVertices,
		(void*)&devEdges,
		(void*)&devWeights,
		(void*)&devLock,
		(void*)&source,
		(void*)&numNodes,
		(void*)&numEdges
	};

	cudaEventRecord(start);
		
	// run the kernel
	cudaLaunchCooperativeKernel((void*)SSSPCompleteKernel, numBlocks, threadsPerBlock, kernelArgs, sharedMem, 0);

	cudaDeviceSynchronize();
		
	cudaEventRecord(stop);

//	getLastCudaError("SSSP kernel failed!");

	cudaEventSynchronize(stop);

	for (int j = 0; j < numNodes; j++)
	{
		cout << "Path from " << vertexMap[source] << " to " << vertexMap[j] << " has weight " << devNodeWeights[j] << endl;
	}
		
	// time the kernel
	float milli_time = 0;
	cudaEventElapsedTime(&milli_time, start, stop);
		
	total_time += milli_time;

	// free arrays
	cudaFree(devNodeWeights);
	cudaFree(devFlags);
	cudaFree(devVertices);
	cudaFree(devEdges);
	cudaFree(devWeights);
	cudaFree(devLock);

	return total_time;
	return (total_time/numNodes);
}

// read file from command line, then run SSSP to test it
int main(int argc, const char * argv[])
{
	if (argc != 2)
	{
		cout << "Specify a file to read!";
		return -1;
	}

	string filename = argv[1];	

	ifstream myFile;
	myFile.open(filename.c_str());

	if(!myFile.is_open())
	{
		cout << "Invalid filename: " << filename;
		return -2;
	}

	int numNodes = 0;
	int numEdges = 0;
	int source = 0;

	readGraphFileStart(numNodes, numEdges, source,  myFile);

	cout << "Number Nodes: " << numNodes << endl;
	cout << "Number Edges: " << numEdges << endl;
	cout << "Source Node: " << source << endl;

	int* vertices = new int[numNodes];
	int* edges = new int[numEdges];
	float* weights = new float[numEdges];

	cout << "Set vertices to -1 as a default" << endl;
	for (int i=0; i<numNodes;i++)
	{
		vertices[i] = -1;
	}

	edgePair* edgePairs = new edgePair[numEdges];
	
	map<long, int> vertexMap = map<long, int>();
	map<int, long> reverseMapping = map<int, long>();
	set<long> vertexSet = set<long>();

	cout << "Reading the vertices in the graph from the file" << endl;
	readVerticesInGraph(vertexSet, edgePairs, numEdges, myFile);
	
	cout << "Closing the file" << endl;
	myFile.close();	

	cout << "Mapping the nodes in the graph file to a 1-N version suitable for arrays" << endl;
	constructVertexMapsFromKnownVertexSet(vertexSet, vertexMap, reverseMapping);

	cout << "Setting source to its translated version" << endl;
	source = vertexMap[source];

	cout << "Populate local arrays with values from the graph file" << endl;
	readGraphFile(vertices, 
		      edges, 
		      weights,
		      numEdges,
 		      vertexMap,
		      edgePairs);

	delete [] edgePairs;
/*
	for (int i = 0; i < numNodes; i++)
	{
		cout << i << " points to " << vertices[i] << endl;
	}
*/
	/*	
	for (int i=0; i < (numNodes-1); i++)
	{
		if (vertices[i] == -1)
			continue;

		for (int j = 0; j < (vertices[i+1]-vertices[i]); j++)
		{
			cout << "Source " << reverseMapping[i] << " connects to " << reverseMapping[edges[vertices[i] + j]] << " with weight " << weights[vertices[i] + j] << endl;
		}	
	}

	if (vertices[numNodes-1] != -1)
	{
		for (int i=0; i < (numEdges-vertices[numNodes-1]); i++)
		{
			cout << "Source " << reverseMapping[(numNodes-1)] << " connects to " << reverseMapping[edges[vertices[numNodes-1] + i]] << " with weight " << weights[vertices[numNodes-1] + i] << endl; 
		}
	}*/	
	try
	{
		// do kernel stuff
		float sum = 0.0;
		int repeats = 5;

		cout << "Starting SSSP!" << endl;

		for (int i = 0; i < repeats; i++)
		{
			float run_time = runSSSP(vertices, edges, weights, numNodes, numEdges, source, reverseMapping);
			cout << "Run time for run " << i << ": " << run_time << endl;
			sum += run_time;
		}
	
		float avg_run_time = sum/repeats;
		cout << "Average run time is " << avg_run_time << " milliseconds" << endl;
	}
	catch (...)
	{
		cout << "Cuda failed!" << endl;
	}

	delete [] vertices;
	delete [] edges;
	delete [] weights;

	return 0;
}
