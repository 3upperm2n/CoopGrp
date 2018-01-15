// unoptimized implementation of single source shortest path
// currently only tracks path cost - not route for shortest path
// based on pseudocode from Singh et al.'s Efficient Parallel Implementation
// of Single Source Shortest Path Algorithm on GPU Using CUDA,
// specifically Algorithm 2 / PNBA variant

#include <iostream>
#include <fstream>

using namespace std;


// initialize kernel
__global__ void Initialize(int *nodeArray, int *flagArray, int source, int n)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid < n)
	{
		nodeArray[tid] = INT_MAX;
		flagArray[tid] = 0;
		
		if (tid == source)
		{
			nodeArray[tid] = 0;
			flagArray[tid] = 1;
		}

	}
}

// relax connections between nodes
__global__ void Relax(int *vertices, 
		      int *edges, 
		      int *weights, 
		      int *flags, 
		      int *nodeWeights, 
		      int *lock, 
		      int numNodes, 
		      int numEdges)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (tid < numNodes)
	{
	
		if (flags[tid] == 1)
		{
			flags[tid] = 0;
			int numNeighbors = (tid != numNodes - 1) ? vertices[tid+1]-vertices[tid] : numEdges-vertices[tid];
			for (int i = 0; i < numNeighbors; i++)
			{
				if (nodeWeights[edges[vertices[tid]+i]] > nodeWeights[tid] + weights[vertices[tid] + i])
				{
				
					atomicMin(&nodeWeights[edges[vertices[tid]+i]], nodeWeights[tid] + weights[vertices[tid]+i]);
					lock[0] = 1;
					flags[edges[vertices[tid]+i]] = 1;
				}
			}
		}
	}
}

void runSSSP(int *vertices,
	     int *edges,
	     int *weights,
	     int numNodes,
	     int numEdges,
	     int source,
	     int destination)
{
	int* lock = new int[1];
	lock[0] = 1; 

	int* flags = new int[numNodes];
	int* nodeWeights = new int[numNodes];

	for (int i = 0; i < numNodes; i++)
	{
		flags[i] = 0;
		nodeWeights[i] = INT_MAX;
	}

	int* devVertices, *devEdges, *devWeights, *devFlags, *devNodeWeights, *devLock;

	size_t nodeSize = numNodes*sizeof(int);
	size_t edgeSize = numEdges*sizeof(int);

	cudaMalloc( (void **)&devVertices, nodeSize);
	cudaMalloc( (void **)&devEdges, edgeSize);
	cudaMalloc( (void **)&devWeights, edgeSize);
	cudaMalloc( (void **)&devFlags, nodeSize);
	cudaMalloc( (void **)&devNodeWeights, nodeSize);
	cudaMalloc( (void **)&devLock, sizeof(int));

	cudaMemcpy(devVertices, vertices, nodeSize, cudaMemcpyHostToDevice);
	cudaMemcpy(devEdges, edges, edgeSize, cudaMemcpyHostToDevice);
	cudaMemcpy(devWeights, weights, edgeSize, cudaMemcpyHostToDevice);
	cudaMemcpy(devFlags, flags, nodeSize, cudaMemcpyHostToDevice);
	cudaMemcpy(devNodeWeights, nodeWeights, nodeSize, cudaMemcpyHostToDevice);
	cudaMemcpy(devLock, lock, sizeof(int), cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(512, 1);
	dim3 numBlocks(4, 1);

	Initialize<<< numBlocks, threadsPerBlock >>>(devNodeWeights, devFlags, source, numNodes);

	cudaMemcpy(flags, devFlags, nodeSize, cudaMemcpyDeviceToHost);

	while (lock[0] == 1)
	{
		lock[0] = 0;
		cudaMemcpy(devLock, lock, sizeof(int), cudaMemcpyHostToDevice);

		Relax<<< numBlocks, threadsPerBlock >>>(devVertices, 
							devEdges, 
							devWeights, 
							devFlags, 
							devNodeWeights, 
							devLock, 
							numNodes, 
							numEdges);
		cudaMemcpy(lock, devLock, sizeof(int), cudaMemcpyDeviceToHost);
	}

	cudaMemcpy(nodeWeights, devNodeWeights, nodeSize, cudaMemcpyDeviceToHost);

	for (int i = 0; i < numNodes; i++)
	{
		cout << "Distance to node " << i << ": " << nodeWeights[i] << endl;
	}

	cudaFree(devLock);
	cudaFree(devVertices);
	cudaFree(devEdges);
	cudaFree(devWeights);
	cudaFree(devFlags);
	cudaFree(devNodeWeights);

	delete [] lock;
	delete [] flags;
	delete [] nodeWeights;
}

// read start of file
void readGraphFileStart(int& source,
			int& destination,
			int& numNodes,
			int& numEdges,
			ifstream &fin)
{

	fin >> numNodes;
	fin >> source;
	fin >> destination;
	fin >> numEdges;

}

// read entire graph file
void readGraphFile(int *vertices, 
		   int *edges,
		   int *weights,
		   ifstream &fin)
{
	int lastNode = 0;
	int edgeIndex = 0;
	int readSource, readDest, readWeight;

	vertices[0] = 0;

	while (fin.peek() != '.')
	{
		fin >> readSource >> readDest >> readWeight;
		
		if (lastNode != readSource)
		{
			lastNode = readSource;
			vertices[lastNode] = edgeIndex;
		}

		edges[edgeIndex] = readDest;
		weights[edgeIndex] = readWeight;
		edgeIndex++;
	}
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

	int source = 0;
	int destination = 0;	
	int numNodes = 0;
	int numEdges = 0;

	readGraphFileStart(source, destination, numNodes, numEdges, myFile);

	int* vertices = new int[numNodes];
	int* edges = new int[numEdges];
	int* weights = new int[numEdges];

	for (int i = 0; i < numNodes; i++)
	{
		vertices[i] = -1;
	}

	readGraphFile(vertices, 
		      edges, 
		      weights, 
		      myFile);

	myFile.close();

	// do kernel stuff

	runSSSP(vertices, edges, weights, numNodes, numEdges, source, destination);

	delete [] vertices;
	delete [] edges;
	delete [] weights;

	return 0;
}
