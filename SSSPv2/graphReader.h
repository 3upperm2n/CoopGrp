#ifndef GRAPH_READER
#define GRAPH_READER

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <map>
#include <set>


struct edgePair
{       
        long source;
        long destination;
        float weight;
};

void readGraphFileStart(int& numNodes,
                        int& numEdges,
                        int& source,
                        std::ifstream &fin);

void readVerticesInGraph(std::set<long> &vertexSet,
                         edgePair *edges,
                         int numEdges,
                         std::ifstream &fin);

void constructVertexMapsFromKnownVertexSet(std::set<long> &vertexSet,
                                           std::map<long, int> &mapToPopulate,
                                           std::map<int, long> &mapForResults);

void readGraphFile(int *vertices,
                   int *edges,
                   float *weights,
                   int numEdges,
                   std::map<long, int> &vertexMap,
                   edgePair *edgePairs);

#endif
