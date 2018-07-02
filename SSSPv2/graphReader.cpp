#include "graphReader.h"

using namespace std;

// read start of file
void readGraphFileStart(int& numNodes,
                        int& numEdges,
                        int& source,
                        ifstream &fin)
{
        
        fin >> numNodes;
        fin >> numEdges;
        fin >> source;
}

// Gives the vertices in the graph
void readVerticesInGraph(std::set<long> &vertexSet,
                         edgePair *edges,
                         int numEdges,
                         ifstream &fin)
{       
        int edge = 0;
        while (edge != numEdges)
        {       
                fin >> edges[edge].source >> edges[edge].destination >> edges[edge].weight;
                vertexSet.insert(edges[edge].source);
                vertexSet.insert(edges[edge].destination);
                
                edge++;
        }
        cout << "Total number of edges: " << edge << endl;
}

// Map the Vertices in order
void constructVertexMapsFromKnownVertexSet(std::set<long> &vertexSet,
                                           std::map<long, int> &mapToPopulate,
                                           std::map<int, long> &mapForResults)
{       
        int count = 0;
        std::set<long>::iterator itr; 
        for (itr = vertexSet.begin(); itr != vertexSet.end(); ++itr)
        {       
                mapToPopulate[*itr] = count;
                mapForResults[count] = *itr;
                count++;
        }
        
        cout << "Total number of vertices: " << count << endl;
}

// read entire graph file
void readGraphFile(int *vertices,
                   int *edges,
                   float *weights,
                   int numEdges,
                   std::map<long, int> &vertexMap,
                   edgePair *edgePairs)
{       
        bool firstNode = true;
        int lastNode = 0;
        int edgeIndex = 0;
        
        while (edgeIndex != numEdges)
        {       
                
                if (edgeIndex >= 14869159)
                {       
                        cout << "MADE IT!" << endl;
                        cout << "Source: " << edgePairs[edgeIndex].source << endl;
                        cout << "Destination: " << edgePairs[edgeIndex].destination << endl; 
                        cout << "ResolvedSource: " << vertexMap[edgePairs[edgeIndex].source] << endl;
                        cout << "ResolvedDest: " << vertexMap[edgePairs[edgeIndex].destination] << endl;
                        cout << "Weight: " << edgePairs[edgeIndex].weight << endl;
                }
                int sourceNode = vertexMap[edgePairs[edgeIndex].source];
                int destinationNode = vertexMap[edgePairs[edgeIndex].destination];
                
                if (firstNode)
                {       
                        lastNode = sourceNode; 
                        vertices[sourceNode] = 0;
                        firstNode = false;
                }
                
                if (lastNode != sourceNode)
                {       
                        lastNode = sourceNode;
                        vertices[lastNode] = edgeIndex;
                }
                
                edges[edgeIndex] = destinationNode;
                weights[edgeIndex] = edgePairs[edgeIndex].weight;
                edgeIndex++;
        }
}

