#ifndef _INSTANCE_H_
#define _INSTANCE_H_

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <stdlib.h>

using namespace std;

class Instance 
{
	public:
		Instance();
		Instance(string filePath);
		~Instance();

		// Just used for draw solutions
		int n, q;

		// Total of operations
		int o;
		// Number of machines
		int m;		
		// Process time of a operation i in the machine k
		int **p;
		// Set A. If operation i preceeds operation j, 
		//then A[i][j]=1. 0 otherwise.
		int **A;
		// List of machines which process an operation i
		vector<int> *F;
		// List of operations that succeeds an operation i
		vector<int> *S;
		// List of operations that preceeds an operation i
		vector<int> *P;
		
};

#endif