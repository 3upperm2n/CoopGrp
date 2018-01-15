//
//  main.cpp
//  Shortest Path
//
//

#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <time.h>
#include <stdlib.h>
#include <sstream>
#include <algorithm>

using namespace std;

string generateConnectedGraph(const int numVertices, const int maxConnections, const int maxWeight)
{
    if (numVertices < maxConnections)
    {
        return "INVALID INPUT; MUST HAVE FEWER NEIGHBORS THAN TOTAL VERTICES";
    }
    
    vector<list<int> > neighbors(numVertices);
    vector<list<int> > weights(numVertices);
    
    srand(time(NULL));
    
    int numEdges = 0;

    // generate a list of all numbers
    for (int i = 0; i < numVertices; i++)
    {
        int connectionsToHave = (rand() % maxConnections) + 1;
        
        // generate new neighbors
        for (int j = 0; j < connectionsToHave; j++){
            
            int newNodeIndex = rand() % (numVertices-1);
            
            if (newNodeIndex == i)
            {
                newNodeIndex = numVertices-1;
            }
            
            int weight = (rand() % maxWeight) + 1;
            numEdges++;
            neighbors[i].push_back(newNodeIndex);
            weights[i].push_back(weight);
        }
        
    }
    
    stringstream summary;
    summary << numVertices << "\n";
    summary << 0 << "\n" << numVertices-1 << "\n" << numEdges;
    
    for (int i = 0; i < numVertices; i++){
        
        list<int>::iterator j = neighbors[i].begin();
        list<int>::iterator k = weights[i].begin();
    
        for(; (j != neighbors[i].end()) && (k != weights[i].end()); ++j, k++){
            summary << "\n" << i << " " << *j << " " << *k;
        }
    }
    
    summary << ".\n";
    
    return summary.str();
}

int main(int argc, const char * argv[]) {
    // insert code here...
    
    if (argc != 5){
        return -1;
    }

    int numNodes = atoi(argv[1]);
    int maxConnections = atoi(argv[2]);
    int maxWeight = atoi(argv[3]);
    
    string fileName = argv[4];
    
    ofstream myFile;
    myFile.open(fileName);
    
    if(!myFile.is_open()){
        return -2;
    }
    
    
    myFile << generateConnectedGraph(numNodes, maxConnections, maxWeight);
    myFile.close();
    
    return 0;
}

