#include "Instance.h"

Instance::Instance()
{
	
}

Instance::Instance(string filePath)
{	
	std::ifstream inFile;
	//std::cout << filePath.c_str();
	inFile.open(filePath.c_str(), std::ifstream::in);

	if (inFile.fail()) {
		std::cerr << "File could not be opened2." << filePath.c_str() << "\n";
		abort();
	}

	/** Start read the instance */
	// Number total of jobs (just used for draw solutions)
	int n;
	inFile >> n;

	// Number of operations per job (just used for draw solutions)
	int q;
	inFile >> q;

	// Number total of operations
	inFile >> this->o;
	
	// Number of precedence
	int nPreced;
	inFile >> nPreced;

	// Number of Machines
	inFile >> this->m;

	// Map precedences and which job pertences each operation
	this->P = new vector<int>[o];
	this->S = new vector<int>[o];
	
	int i = 0;
	int tmp1, tmp2;

	this->A = new int*[o];

	for(int i = 0; i < o; i++)
	{
		this->A[i] = new int[o];
		
		for(int j = 0; j < o; j++)
			this->A[i][j] = 0;
	}

	for(int t = 0; t < nPreced; t++)
	{
		inFile >> tmp1;
		inFile >> tmp2;

		this->A[tmp1][tmp2] = 1;

		this->P[tmp2].push_back(tmp1);
		this->S[tmp1].push_back(tmp2);
	}
	
	// Process time
	this->p = new int*[o];

	for(int k = 0; k < o; k++)
	{
		this->p[k] = new int[m];

		for(int j = 0; j < m; j++)
			this->p[k][j] = -1;
	}

	int amin = 100000, amax = -1, atotal = 0;

	// List of possible machines that process an operation i
	this->F = new vector<int>[this->o];
	
	for(int i = 0; i < this->o; i++)
	{
		int nMach = -1;
		inFile >> nMach;

		amin = min(amin, nMach);
		amax = max(amax, nMach);
		atotal += nMach;

		for(int h = 0; h < nMach; h++)
		{
			int machine, processTime;

			inFile >> machine >> processTime;

			this->p[i][machine] = processTime;
			this->F[i].push_back(machine);
		}
	}	

	int qmin = 10000, qmax = -1;

	for(int i = 0; i < o; i++)
	{
		qmin = min(qmin, (int) this->P[i].size());
		qmax = max(qmax, (int) this->P[i].size());
	}
	
	inFile.close();

	/*std::ofstream resultsFile("results.xls", std::ios::out 
							| std::ios::app);

	resultsFile << this->n << "\t" << this->m << "\t" << "\t" << "\t" << this->o << "\t"
		<< amin << "\t" << amax << "\t" << atotal << "\t" << qmin << "\t" << qmax << "\t" << nPreced << "\n";
	resultsFile.close();*/
}

Instance::~Instance(void)
{
	delete [] F;
	delete [] S;
	delete [] P;

	for(int i = 0; i < o; i++)
	{
		delete [] p[i];
		delete [] A[i];
	}
	delete [] p;
	delete [] A;
}