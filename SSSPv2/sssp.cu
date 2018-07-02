// unoptimized implementation of single source shortest path
// currently only tracks path cost - not route for shortest path
// based on pseudocode from Singh et al.'s Efficient Parallel Implementation
// of Single Source Shortest Path Algorithm on GPU Using CUDA,
// specifically Algorithm 2 / PNBA variant

#include <iostream>
#include <fstream>
#include <cooperative_groups.h>
#include "graphReader.h"
#define THREADS_PER_BLOCK 1024

using namespace std;
using namespace cooperative_groups;

// initialize kernel
__global__ void Initialize(float *nodeArray, int *flagArray, int source, int n)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
		
	if (tid < n)
	{

		if (tid == source)
		{
			nodeArray[tid] = 0.0;
			flagArray[tid] = 1;
		} 
		else
		{
			nodeArray[tid] = 10000000;
			flagArray[tid] = 0;
		}
	}	
}

// relax connections between nodes
__global__ void Relax(int *vertices, 
		      int *edges, 
		      float *weights, 
		      int *flags, 
		      float *nodeWeights, 
		      int *lock, 
		      int numNodes, 
		      int numEdges)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int gridSize = blockDim.x*gridDim.x;
		
	for (int j = tid; j < numNodes; j += gridSize)
	//if (tid < numNodes)
	{
		
		if (flags[j] == 1)
		{
			flags[j] = 0;
			int numNeighbors = (j != numNodes - 1) ? vertices[j+1]-vertices[j] : numEdges-vertices[j];
			for (int i = 0; i < numNeighbors; i++)
			{
				if (nodeWeights[edges[vertices[j]+i]] > nodeWeights[j] + weights[vertices[j] + i])
				{
					nodeWeights[edges[vertices[j]+i]] = nodeWeights[j] + weights[vertices[j]+i];
			//		atomicMin(&nodeWeights[edges[vertices[tid]+i]], nodeWeights[tid] + weights[vertices[tid]+i]);
					lock[0] = 1;
					flags[edges[vertices[j]+i]] = 1;
				}
			}
		}
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
	int* devVertices, *devEdges, *devFlags, *lock;
	float *devWeights, *devNodeWeights;

	size_t nodeSize = numNodes*sizeof(int);
	size_t edgeSize = numEdges*sizeof(int);
	size_t weightSize = numEdges*sizeof(float);
	size_t weightNodeSize = numNodes*sizeof(float);

	// initializing GPU memory
	cudaMallocManaged( (void **)&devVertices, nodeSize);
	cudaMallocManaged( (void **)&devEdges, edgeSize);
	cudaMallocManaged( (void **)&devWeights, weightSize);
	cudaMallocManaged( (void **)&devFlags, nodeSize);
	cudaMallocManaged( (void **)&devNodeWeights, weightNodeSize);
	cudaMallocManaged( (void **)&lock, sizeof(int));

	cudaMemcpy(devVertices, vertices, nodeSize, cudaMemcpyHostToDevice);
	cudaMemcpy(devEdges, edges, edgeSize, cudaMemcpyHostToDevice);
	cudaMemcpy(devWeights, weights, weightSize, cudaMemcpyHostToDevice);
	lock[0] = 1;

	int num_blocks = 1 + ((numNodes - 1)/THREADS_PER_BLOCK);

	dim3 threadsPerBlock(THREADS_PER_BLOCK, 1);
	dim3 numBlocks(num_blocks, 1);

	float total_time = 0.0;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for (int i=source; i <= source; i++)
	{

		cout << "Running initialize" << endl;
		Initialize<<< numBlocks, threadsPerBlock >>>(devNodeWeights, devFlags, i, numNodes);

		cudaDeviceSynchronize();	

		cudaEventRecord(start);

		while (lock[0] == 1)
		{
			lock[0] = 0;

			Relax<<< numBlocks, threadsPerBlock >>>(devVertices, 
								devEdges, 
								devWeights, 
								devFlags, 
								devNodeWeights, 
								lock, 
								numNodes, 
								numEdges);
			cudaDeviceSynchronize();

		}

		cudaEventRecord(stop);

		cudaEventSynchronize(stop);

		for (int j = 0; j < numNodes; j++)
		{
			cout << "Path from " << vertexMap[source] << " to " << vertexMap[j] << " has weight " << devNodeWeights[j] << endl;
		}

		float milli_time = 0;
		cudaEventElapsedTime(&milli_time, start, stop);
		
		total_time += milli_time;
	}
	
	cudaFree(devVertices);
	cudaFree(devEdges);
	cudaFree(devWeights);
	cudaFree(devFlags);
	cudaFree(devNodeWeights);
	cudaFree(lock);
	cudaFree(start);
	cudaFree(stop);

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
