#include <iostream>      
#include <string>        
#include <fstream>       
#include <math.h>        
#include <vector>        
#include <time.h>        
#include <stdlib.h>      
#include <string>        
#include <sstream>       
#include <algorithm>     
#include <limits.h>
#include "instance.h" 

int bMin, bMax;
double bMed;

class Node
{
public:
	int o;
	int m;
	int ell;
	int *s;
	int *w;
	int Cmax;
	int *u;
	int *v;
	int *L;
	int *eta;
	vector<int> Omega;
	int CmaxEst;

	Node(int o, int m, int ell, int *s, int *w, 
		int Cmax, int *u, int *v, int *L, 
		int *eta, vector<int> Omega, int CmaxEst)
	{
		this->o = o;
		this->m = m;
		this->ell = ell;
		this->s = new int[o];
		this->w = new int[o];
		this->Cmax = Cmax;
		this->u = new int[o];
		this->v = new int[m];
		this->L = new int[m];
		this->eta = new int[o];
		this->Omega = Omega;
		this->CmaxEst = CmaxEst;

		for(int i = 0; i < o; i++)
		{
			this->s[i] = s[i];
			this->w[i] = w[i];
			this->u[i] = u[i];
			this->eta[i] = eta[i];
		}

		for(int k = 0; k < m; k++)
		{
			this->v[k] = v[k];
			this->L[k] = L[k];
		}
	}

	Node (const Node & n)
	{			
		this->o = n.o;
		this->m = n.m;
		this->ell = n.ell;
		this->s = new int[o];
		this->w = new int[o];
		this->Cmax = n.Cmax;
		this->u = new int[o];
		this->v = new int[m];
		this->L = new int[m];
		this->eta = new int[o];
		this->Omega = n.Omega;
		this->CmaxEst = n.CmaxEst;

		for(int i = 0; i < o; i++)
		{
			this->s[i] = n.s[i];
			this->w[i] = n.w[i];
			this->u[i] = n.u[i];
			this->eta[i] = n.eta[i];
		}

		for(int k = 0; k < m; k++)
		{
			this->v[k] = n.v[k];
			this->L[k] = n.L[k];
		}
	}

	Node operator=(Node n)
	{
		delete [] s;
		delete [] w;
		delete [] u;
		delete [] v;
		delete [] L;
		delete [] eta;
		Omega.clear();
		
		this->o = n.o;
		this->m = n.m;
		this->ell = n.ell;
		this->s = new int[o];
		this->w = new int[o];
		this->Cmax = n.Cmax;
		this->u = new int[o];
		this->v = new int[m];
		this->L = new int[m];
		this->eta = new int[o];
		this->Omega = n.Omega;
		this->CmaxEst = n.CmaxEst;

		for(int i = 0; i < o; i++)
		{
			this->s[i] = n.s[i];
			this->w[i] = n.w[i];
			this->u[i] = n.u[i];
			this->eta[i] = n.eta[i];
		}

		for(int k = 0; k < m; k++)
		{
			this->v[k] = n.v[k];
			this->L[k] = n.L[k];
		}
		return (*this);
	}

	bool operator() (Node n1, Node n2) { return (n1.CmaxEst<n2.CmaxEst); }
	
	bool operator==(Node n)
	{
		for(int i = 0; i < o; i++)
		{
			if(this->s[i] != n.s[i] || this->w[i] != n.w[i])
				return false;
		}

		return true;
	}
	bool operator<(const Node n) const
	{
		return (this->CmaxEst<n.CmaxEst); 
	}
	~Node(void)
	{
		delete [] s;
		delete [] w;
		delete [] u;
		delete [] v;
		delete [] L;
		delete [] eta;
		Omega.clear();
	}
};

class DNid
{
public:
	int lambda;
	int theta;
	Node *n;

	DNid(int lambda, int theta, Node n)
	{
		this->n = new Node(n);
		this->lambda = lambda;
		this->theta = theta;
	}
	DNid (const DNid & d)
	{			
		this->lambda = d.lambda;
		this->theta = d.theta;
		this->n = new Node(*d.n);
	}
	DNid operator=(DNid d)
	{
		delete n;
		
		this->lambda = d.lambda;
		this->theta = d.theta;
		this->n = new Node(*d.n);

		return (*this);
	}	
	bool operator==(DNid d)
	{
		for(int i = 0; i < d.n->o; i++)
		{
			if(this->n->s[i] != d.n->s[i] || this->n->w[i] != d.n->w[i])
				return false;
		}

		return true;
	}
	bool operator<(const DNid d) const
	{
		return (this->lambda<d.lambda || (this->lambda==d.lambda && this->theta<d.theta)); 
	}
	~DNid(void)
	{
		delete n;
	}
};

class ListSchedulingReturn 
{
public:
	int o;
	int Cmax;

	int *s;
	int *w;	

	ListSchedulingReturn(int o, int Cmax, int *s, int *w)
	{
		this->o = o;
		this->Cmax = Cmax;
		this->s = s;
		this->w = w;
	}

	~ListSchedulingReturn(void)
	{
		delete [] s;
		delete [] w;
	};
};

class SelectReturn 
{
public:
	int lambda;
	int theta;
	int omegaPos;

	SelectReturn(int lambda, int theta, int omegaPos)
	{
		this->lambda = lambda;
		this->theta = theta;
		this->omegaPos = omegaPos;
	}

	~SelectReturn(void)
	{
	};
};

class BeamSearchReturn 
{
public:
	int o;
	int *s;
	int *w;
	int Cmax;

	BeamSearchReturn(int o, int *s, int *w, int Cmax)
	{
		this->o = o;
		this->s = new int[o];
		this->w = new int[o];
		this->Cmax = Cmax;
		
		for(int i = 0; i < o; i++)
		{
			this->s[i] = s[i];
			this->w[i] = w[i];
		}
	}

	BeamSearchReturn(const BeamSearchReturn & d)
	{	
		this->o = d.o;
		this->s = new int[o];
		this->w = new int[o];
		this->Cmax = d.Cmax;
		
		for(int i = 0; i < o; i++)
		{
			this->s[i] = d.s[i];
			this->w[i] = d.w[i];
		}
	}
	
	BeamSearchReturn operator=(BeamSearchReturn d)
	{
		delete [] s;
		delete [] w;
		
		this->o = d.o;
		this->s = new int[o];
		this->w = new int[o];
		this->Cmax = d.Cmax;
		
		for(int i = 0; i < o; i++)
		{
			this->s[i] = d.s[i];
			this->w[i] = d.w[i];
		}

		return *this;
	}

	~BeamSearchReturn(void)
	{
		delete [] s;
		delete [] w;
	};
};


class Pair
{
public:
	int i;
	int k;

	Pair(int i, int k)
	{
		this->i = i;
		this->k = k;
	}
	~Pair(void)
	{
	};
};

ListSchedulingReturn listScheduling(int m, int o, vector<int> *F, int **p, vector<int> *S, vector<int> *P)
{
	double *pBar = new double[o];

	// Compute pBar
	for(int i = 0; i < o; i++)
	{
		pBar[i] = 0;

		for(unsigned int l = 0; l < F[i].size(); l++)
		{
			int k = F[i][l];

			pBar[i] += p[i][k];
			//pBar[i] = max<double>(pBar[i], p[i][k]);
		}

		pBar[i] /= F[i].size();
	}

	vector<int> Q;
	bool *inQ = new bool[o];
	double *RW = new double[o];

	// Put the operations that has no sucessors	in Q
	// and initialize RW of these operations
	for(int i = 0; i < o; i++)
	{
		if( (int) S[i].size() == 0 )
		{
			Q.push_back(i);
			inQ[i] = true;
			RW[i] = pBar[i];
		}
		else
		{
			RW[i] = 0;
			inQ[i] = false;
		}
	}
	
	// Compute RW of the operations that has sucessors
	// According the longest path using a dijkstra modified
	while(Q.size() > 0)
	{
		// Remove the last element z from Q
		int z =	Q.back();
		Q.pop_back();
		inQ[z] = false;

		for(unsigned int e = 0; e < P[z].size(); e++)
		{
			// Set i as the e'th predecessor of the operation z
			int i = P[z][e];
			
			if( RW[z] + pBar[i] > RW[i] )
			{
				RW[i] = RW[z] + pBar[i];
				
				if( ! inQ[i] )
				{
					Q.push_back(i);
					inQ[i] = true;
				}
			}
		}
	}

	int Cmax = 0; int *u = new int[o]; 
	int *v = new int[m]; int *L = new int[m];
	int *s = new int[o]; int *w = new int[o];

	for(int i = 0; i < o; i++)
	{
		u[i] = 0;
		s[i] = INT_MAX;
		w[i] = -1;
	}

	for(int k = 0; k < m; k++)
		v[k] = L[k] = 0;

	// Compute the load L for each machine
	for(int i = 0; i < o; i++)
		for(unsigned int k = 0; k < F[i].size(); k++)
			L[ F[i][k] ] += p[i][ F[i][k] ];

	// Compute eta and initialize Omega
	int *eta = new int[o];
	vector<int> Omega;
	for(int i = 0; i < o; i++)
	{
		// Set eta
		eta[i] = (int) P[i].size();

		// If i has no predecessor, then put it in Omega
		if(eta[i] == 0)
			Omega.push_back(i);
	}

	// Main loop
	int times = 0;
	do
	{
		int stHat = INT_MAX;
		int lambda,theta;
		unsigned int cPos;
		double RWHat;

		for(unsigned int c = 0; c < Omega.size(); c++)
		{
			int i = Omega[c];

			int stTil = INT_MAX;
			int ptTil,kTil;			

			// Define the best machine to process i
			for(unsigned int l = 0; l < F[i].size(); l++)
			{
				int k = F[i][l];

				int st = max(u[i], v[k]);

				if(
					(st < stTil) ||
					(st == stTil && p[i][k] < ptTil) || 
					(st == stTil && p[i][k] == ptTil && L[k] < L[kTil]) ||
					(st == stTil && p[i][k] == ptTil && L[k] == L[kTil] && k < kTil)
				)
				{
					stTil = st; ptTil = p[i][k]; kTil = k;
				}
			}

			if(
				(stTil < stHat) ||
				(stTil == stHat && RW[i] > RWHat) || 
				(stTil == stHat && RW[i] == RWHat && L[kTil] > L[theta]) ||
				(stTil == stHat && RW[i] == RWHat && L[kTil] == L[theta] && i < lambda)
			)
			{
				stHat = stTil; RWHat = RW[i]; theta = kTil; lambda = i; cPos = c;
			}
		}
		
		// Update the schedule
		s[lambda] = max(u[lambda], v[theta]);
		w[lambda] = theta;
		Cmax = max(Cmax, s[lambda] + p[lambda][theta]);
		v[theta] = s[lambda] + p[lambda][theta];
		Omega.erase(Omega.begin() + cPos);

		// Update the load L
		for(unsigned int l = 0; l < F[lambda].size(); l++)
		{
			int k = F[lambda][l];
			L[k] -= p[lambda][k];
		}

		// Update eta, u and put in omega those operations
		// that has no more precedences (eta == 0)
		for(unsigned int c = 0; c < S[lambda].size(); c++)
		{
			int i = S[lambda][c];
			
			eta[i]--;
			u[i] = max(u[i], s[lambda] + p[lambda][theta]);

			if(eta[i] == 0)
				Omega.push_back(i);
		}
	}
	while(++times < o);
	
	delete [] pBar;
	delete [] RW;
	delete [] u;
	delete [] v; 
	delete [] L;
	delete [] inQ;
	delete [] eta;

	return ListSchedulingReturn (o, Cmax, s, w);

}

/** 
* Return true if the vector Forb does not contain
* the pair (i,k). False otherwise.
*/
bool Forbidden(int i, int k, vector<Pair> Forb, bool ** ForbMat)
{
	if ( Forb.size() == 0 )
		return false;
	else
		return ForbMat[i][k];
}

bool notInForb(int i, int k, vector<Pair> Forb)
{
	for(unsigned int l = 0; l < Forb.size(); l++)
		if(Forb[l].i == i && Forb[l].k == k)
			return false;

	return true;
}

SelectReturn select(int m, int o, vector<int> *F, int **p, double *RW, int *u, int *v, 
					int *L, vector<int> Omega, vector<Pair> Forb, bool ** ForbMat)
{
	int stHat = INT_MAX;
	double RWHat = 0;

	int theta;
	int lambda;
	int omegaPos;

	// ForEach i in Omega
	for(unsigned int l = 0; l < Omega.size(); l++)
	{
		int i = Omega[l];

		int stTil = INT_MAX;
		int ptTil, kTil, omegaPosTil;

		// ForEach k in F[i]
		for(unsigned g = 0; g < F[i].size(); g++)
		{
			int k = F[i][g];

			// If (i,k) is not forbidden
			if( ! Forbidden(i,k,Forb,ForbMat) )
			{
				int st = max(u[i], v[k]);

				if( 
					(st < stTil) ||
					(st == stTil && p[i][k] < ptTil) ||
					(st == stTil && p[i][k] == ptTil && L[k] < L[kTil]) ||
					(st == stTil && p[i][k] == ptTil && L[k] == L[kTil] && k < kTil)
				)
				{
					stTil = st; ptTil = p[i][k]; kTil = k; omegaPosTil = l;
				}
			}
		}
		if(
			(stTil < stHat) || 
			(stTil == stHat && RW[i] > RWHat) ||
			(stTil == stHat && RW[i] == RWHat && L[kTil] > L[theta]) ||
			(stTil == stHat && RW[i] == RWHat && L[kTil] == L[theta] && i < lambda)
		)
		{
			stHat = stTil; RWHat = RW[i]; theta = kTil; lambda = i; omegaPos = omegaPosTil;
		}
	}

	return SelectReturn(lambda, theta, omegaPos);
}

int cMaxEst(int m, int o, vector<int> *F, int **p, double *RW2, vector<int> *S, 
			int ell, int *s2, int *w2, int Cmax, int *u2, int *v2, int *L2, 
			int *eta2, vector<int> Omega2)
{
	vector<Pair> Forb;

	double *RW = new double[o];
	int *s = new int[o];
	int *w = new int[o];
	int *u = new int[o];
	int *v = new int[m];
	int *L = new int[m];
	int *eta = new int[o];
	vector<int> Omega(Omega2);

	for(int i = 0; i < o; i++)
	{
		s[i] = s2[i];
		w[i] = w2[i];
		u[i] = u2[i];
		eta[i] = eta2[i];
		RW[i] = RW2[i];
	}

	for(int k = 0; k < m; k++)
	{
		v[k] = v2[k];
		L[k] = L2[k];
	}

	for(int b = 0; b < (o - ell); b++)
	{
		// Select the best operation to be scheduled
		SelectReturn sel = select(m, o, F, p, RW, u, v, L, Omega, Forb, (bool**)NULL);

		// Update the schedule
		s[sel.lambda] = max(u[sel.lambda], v[sel.theta]);
		w[sel.lambda] = sel.theta;
		Cmax = max(Cmax, s[sel.lambda] + p[sel.lambda][sel.theta]);
		v[sel.theta] = s[sel.lambda] + p[sel.lambda][sel.theta];
		
		// Remove from Omega the scheduled operation
		Omega.erase(Omega.begin() + sel.omegaPos);
		
		// Update the load machine L
		for(unsigned int ki = 0; ki < F[sel.lambda].size(); ki++)
		{
			int k = F[sel.lambda][ki];
			L[k] = L[k] - p[sel.lambda][k];
		}
		
		// Update Omega set
		for(unsigned int pos = 0; pos < S[sel.lambda].size(); pos++)
		{
			int i = S[sel.lambda][pos];

			eta[i] = eta[i] - 1;			
			u[i] = max(u[i], s[sel.lambda] + p[sel.lambda][sel.theta]);

			if(eta[i] == 0)
				Omega.push_back(i);
		}

	}

	delete [] s;
	delete [] w;
	delete [] u;
	delete [] v;
	delete [] L;
	delete [] eta;
	delete [] RW;

	return Cmax;
}

BeamSearchReturn beamSearch(int m, int o, vector<int> *F, int **p, vector<int> *S, double *RW, double alpha, double ksi, vector<Node> N)
{
	vector<Pair> Forb;
	bool **ForbMat = new bool*[o];	
	
	for(int i = 0; i < o; i++)
	{
		ForbMat[i] = new bool[m];
		for(int k = 0; k < m; k++)
			ForbMat[i][k] = false;
	}

	int times = 0;
	do
	{
		int nid = -1;
		vector< vector<DNid> > Dnid;

		Node *var;

		for(unsigned int l = 0; l < N.size(); l++)
		{
			Node n(N[l]);

			nid = nid + 1;

			// Sum the possible pairs (i,k) for each i in Omega
			// and k in F[i]
			int sum = 0;
			for(unsigned int z = 0; z < n.Omega.size(); z++)
			{
				int i = n.Omega[z];
				sum += (int) F[i].size();
			}
			
			// Compute alphaHat1
			int alphaHat1 = (int) min((int)ceil(alpha * sum), sum);

			int stmin = INT_MAX;
			int ptmax = 0;
			
			// For each i in n.Omega
			for(unsigned int z = 0; z < n.Omega.size(); z++)
			{
				int i = n.Omega[z];
				
				// For each k in F[i]
				for(unsigned int l = 0; l < F[i].size(); l++)
				{
					int k = F[i][l];

					int st = max(n.u[i], n.v[k]);
					stmin = min(st, stmin);
					ptmax = max(ptmax, p[i][k]);
				}
			}

			int alphaHat2 = 0;

			// For each i in n.Omega
			for(unsigned int z = 0; z < n.Omega.size(); z++)
			{
				int i = n.Omega[z];
				
				// For each k in F[i]
				for(unsigned int l = 0; l < F[i].size(); l++)
				{
					int k = F[i][l];

					int st = max(n.u[i], n.v[k]);
					
					if(st <= (stmin + ksi * ptmax))
						alphaHat2++;
				}
			}

			// Clean ForbMat structure			
			for (unsigned int i=0; i < Forb.size(); i++)
				ForbMat[ Forb[i].i ][ Forb[i].k ] = false;
			Forb.clear();			
			
			vector<DNid> D;

			// Apply the global search (cMaxEst) over the 
			// min(alphaHat1, alphaHat2) candidate children.
			for(int t = 0; t < min(alphaHat1, alphaHat2); t++)
			{
				Node n2(n); var = &n2;
				
				// Select the best pair (lambda,theta)
				SelectReturn sel = select(m, o, F, p, RW, n2.u, n2.v, n2.L, n2.Omega, Forb, ForbMat);

				int lambda = sel.lambda; int theta = sel.theta; int pos = sel.omegaPos;

				// Schedule in the node n2 the operation lambda to be processed on machine theta 
				n2.s[lambda] = max(n2.u[lambda], n2.v[theta]);
				n2.w[lambda] = theta;
				n2.Cmax = max(n2.Cmax, n2.s[lambda] + p[lambda][theta]);
				n2.v[theta] = n2.s[lambda] + p[lambda][theta];
				n2.Omega.erase(n2.Omega.begin() + pos);

				for(unsigned int l = 0; l < F[lambda].size(); l++)
				{
					int k = F[lambda][l];
					n2.L[k] -= p[lambda][k];
				}

				for(unsigned int l = 0; l < S[lambda].size(); l++)
				{
					int i = S[lambda][l];

					n2.eta[i]--;
					n2.u[i] = max(n2.u[i], n2.s[lambda] + p[lambda][theta]);
					if(n2.eta[i] == 0)
						n2.Omega.push_back(i);
				}

				n2.ell++;

				// Get the estimation of the completion time considering the node n2
				int CmaxEst = cMaxEst(m, o, F, p, RW, S, n2.ell, n2.s, n2.w, n2.Cmax, n2.u, n2.v, n2.L, n2.eta, n2.Omega);

				n2.CmaxEst = CmaxEst;

				// Put the candidate node in the list of candidates
				D.push_back(DNid(lambda, theta, Node(n2)));
				// Put in the forbidden list the pair (lambda, theta)
				Forb.push_back(Pair(lambda, theta));
				ForbMat[lambda][theta] = true;
			}

			Dnid.push_back(D);

		}
		
		vector<Node> N2;
		
		// Verify if some of the nodes generated from a 
		// father become equal to another node
		// generated from another father. If that
		// so, mark the second one with CmaxEst = INT_MAX.
		// After this, remove from the candidate nodes
		// those ones marked with CmaxEst = INT_MAX.
		// Finally, put in N2 set, one node from each
		// father that has the smallest CmaxEst compared
		// with the others nodes generated from the same
		// father. Thus, just one child or no one survives
		// for the next generation (level) of the search tree.
		for(int nid1 = 0; nid1 < (int) N.size(); nid1++)
		{
			for(unsigned int l = 0; l < Dnid[nid1].size(); l++)
			{
				DNid *d1 = &Dnid[nid1][l];
				for(int nid2 = nid1+1; nid2 < (int) N.size(); nid2++)
				{
					for(unsigned int l2 = 0; l2 < Dnid[nid2].size(); l2++)
					{
						DNid *d2 = &Dnid[nid2][l2];	

						// If d1->n->s == d2->n->s and d1->n->w == d2->n->w
						if((*d1) == (*d2) || (times + 1 == o && d2->n->Cmax == d1->n->Cmax) )
						{							
							if( d1->lambda < d2-> lambda || ( d1->lambda == d2->lambda && d1->theta < d2->theta ) )
								d2->n->CmaxEst = INT_MAX;
							else
								d1->n->CmaxEst = INT_MAX;
						}
					}
				}
			}
			
			int pos = -1;
			int smallCmaxEst = INT_MAX;
			int smallLambda = INT_MAX;
			int smalltheta = INT_MAX;

			// Remove from Dnid[nid1] the candidate nodes that 
			// CmaxEst is infinite (10e5)
			// It is commented because it is not necessary
			/*for(int l = ((int) Dnid[nid1].size()) - 1; l >= 0; l--)
				if(Dnid[nid1][l].n->CmaxEst == (int) 10e6)
					Dnid[nid1].erase(Dnid[nid1].begin() + l);*/
			
			// Add the node with the smallest CmaxEst. In draw cases,
			// choose the one with the smallest lambda and smallest
			// theta respectively.
			for(unsigned int l = 0; l < Dnid[nid1].size(); l++)
			{
				if ( Dnid[nid1][l].n->CmaxEst != INT_MAX )
				{
					if( smallCmaxEst > Dnid[nid1][l].n->CmaxEst ||
						(smallCmaxEst == Dnid[nid1][l].n->CmaxEst 
							&& smallLambda > Dnid[nid1][l].lambda) ||
						(smallCmaxEst == Dnid[nid1][l].n->CmaxEst 
							&& smallLambda == Dnid[nid1][l].lambda
							&& smalltheta > Dnid[nid1][l].theta) )
					{
						pos = l;
						smallCmaxEst = Dnid[nid1][l].n->CmaxEst;
						smallLambda = Dnid[nid1][l].lambda;
						smalltheta = Dnid[nid1][l].theta;
					}
				}
			}

			// Pos will be equal to -1 if and just if the branch's tree die. 
			// This happen if all the possibilities of node's child become
			// identical to another father's child. This is much better 
			// explain in the paper.
			if(pos != -1)
				//N2.push_back(Node(*Dnid[nid1][pos].n));
				N2.insert(N2.begin(), Node(*Dnid[nid1][pos].n));
		}

		// N receive N2
		N.clear();
		N = N2;
	}
	while(++times < (o - 1));

	int smallestCmax = INT_MAX;
	int pos;

	for(unsigned int z = 0; z < N.size(); z++)
	{
		if(N[z].Cmax < smallestCmax)
		{
			smallestCmax = N[z].Cmax;
			pos = z;
		}
	}

	for(int i = 0; i < o; i++)
	{
		delete [] ForbMat[i];
	}
	delete [] ForbMat;

	return BeamSearchReturn(o, N[pos].s, N[pos].w, N[pos].Cmax);
}

BeamSearchReturn beamSearchIni(int m, int o, vector<int> *F, int **p, vector<int> *S, vector<int> *P, double alpha, double beta, double ksi)
{
	double *pBar = new double[o];

	// Compute pBar
	for(int i = 0; i < o; i++)
	{
		pBar[i] = 0;

		for(unsigned int l = 0; l < F[i].size(); l++)
		{
			int k = F[i][l];

			pBar[i] += p[i][k];
			//pBar[i] = max<double>(pBar[i], p[i][k]);
		}

		pBar[i] /= F[i].size();
	}
	
	vector<int> Q;
	bool *inQ = new bool[o];
	double *RW = new double[o];

	// Put the operations that has no sucessors	in Q
	// and initialize RW of these operations
	for(int i = 0; i < o; i++)
	{
		if( (int) S[i].size() == 0 )
		{
			Q.push_back(i);
			inQ[i] = true;
			RW[i] = pBar[i];
		}
		else
		{
			RW[i] = 0;
			inQ[i] = false;
		}
	}
	
	// Compute RW of the operations that has sucessors
	// According the longest path using a dijkstra modified
	while(Q.size() > 0)
	{
		// Remove the last element z from Q
		int z =	Q.back();
		Q.pop_back();
		inQ[z] = false;

		for(unsigned int e = 0; e < P[z].size(); e++)
		{
			// Set i as the e'th predecessor of the operation z
			int i = P[z][e];
			
			if( RW[z] + pBar[i] > RW[i] )
			{
				RW[i] = RW[z] + pBar[i];
				
				if( ! inQ[i] )
				{
					Q.push_back(i);
					inQ[i] = true;
				}
			}
		}
	}

	int Cmax = 0; int *u = new int[o]; 
	int *v = new int[m]; int *L = new int[m];
	int *s = new int[o]; int *w = new int[o];

	for(int i = 0; i < o; i++)
	{
		u[i] = 0;
		s[i] = INT_MAX;
		w[i] = -1;
	}

	for(int k = 0; k < m; k++)
		v[k] = L[k] = 0;

	// Compute the load L for each machine
	for(int i = 0; i < o; i++)
		for(unsigned int k = 0; k < F[i].size(); k++)
			L[ F[i][k] ] += p[i][ F[i][k] ];

	// Compute eta and initialize Omega
	int *eta = new int[o];
	vector<int> Omega;
	for(int i = 0; i < o; i++)
	{
		// Set eta
		eta[i] = (int) P[i].size();

		// If i has no predecessor, then put it in Omega
		if(eta[i] == 0)
			Omega.push_back(i);
	}

	// Set of initial candidate nodes
	vector<Node> N2;
	int *L2 = new int[m];
	int *eta2 = new int[o];
	
	int *s2 = new int[o];
	int *w2 = new int[o];
	int *u2 = new int[o];
	int *v2 = new int[m];

	for(unsigned int c = 0; c < Omega.size(); c++)
	{
		int lamb = Omega[c];
		
		vector<int> Omega2(Omega);

		for(int i = 0; i < o; i++)
			eta2[i] = eta[i];
		for(int k = 0; k < m; k++)
			L2[k] = L[k];

		// Update the L copy as scheduling the operation lamb
		for(unsigned int b = 0; b < F[lamb].size(); b++)
		{
			int k = F[lamb][b];
			L2[k] = L2[k] - p[lamb][k];
		}

		// Update eta copy and omega copy as scheduling the 
		// operation lamb
		for(unsigned int a = 0; a < S[lamb].size(); a++)
		{
			int i = S[lamb][a];

			eta2[i] = eta2[i] - 1;

			if(eta2[i] == 0)
				Omega2.push_back(i);
		}		

		// Remove lamb from Omega copy
		Omega2.erase(Omega2.begin() + c);

		for(unsigned int b = 0; b < F[lamb].size(); b++)
		{
			int theta = F[lamb][b];			
			int Cmax2 = Cmax;			

			for(int i = 0; i < o; i++)
			{
				s2[i] = s[i];
				w2[i] = w[i];
				u2[i] = u[i];
			}

			for(int k = 0; k < m; k++)
				v2[k] = v[k];

			// Schedule in the copys the operation lamb with machine theta
			s2[lamb] = max(u2[lamb], v2[theta]);
			w2[lamb] = theta;
			Cmax2 = max(Cmax2, s2[lamb] + p[lamb][theta]);
			v2[theta] = s2[lamb] + p[lamb][theta];

			for(unsigned int a = 0; a < S[lamb].size(); a++)
			{
				int i = S[lamb][a];
				u2[i] = max(u2[i], s2[lamb] + p[lamb][theta]);
			}

			// Get the Cmax estimation with operation lamb scheduled on the machine theta
			int CmaxEst = cMaxEst(m, o, F, p, RW, S, 1, s2, w2, Cmax2, u2, v2, L2, eta2, Omega2);

			// Put the candidate node in the set N2. This node consider the operation lamb
			// scheduled on the machine theta
			N2.push_back(Node(o, m, 1, s2, w2, Cmax2, u2, v2, L2, eta2, Omega2, CmaxEst));
		}
	}

	// Calculate betaHat
	int betaHat = (int) ceil( beta * (double) N2.size());

	// Sort N2 set in crescent order on CmaxEst
	std::sort(N2.begin(), N2.end());

	// Get the betaHat-th CmaxEst in the N2 set
	int bTHCmax = N2[ betaHat - 1 ].CmaxEst;
	//int bTHCmax = N2[ min( betaHat, (int)N2.size()-1) ].CmaxEst;

	// New set of nodes that will be used in the beamSearch method
	vector<Node> N; 

	unsigned int iterator = 0;

	// Put the first betaHat-th less candidates nodes with the smallest
	// CmaxEst and the ones that draw with the betaHat-th candidate
	while(iterator < (int) N2.size() && N2[iterator].CmaxEst <= bTHCmax)
	{
		N.push_back( N2[iterator] ); 
		iterator++;
	}
	
	BeamSearchReturn bsr(beamSearch(m, o, F, p, S, RW, alpha, ksi, N));
	
	delete [] pBar;
	delete [] RW;
	delete [] u; 
	delete [] v; 
	delete [] L;
	delete [] s; 
	delete [] w;
	delete [] inQ;
	delete [] eta;
	
	delete [] L2;
	delete [] eta2;	
	delete [] s2;
	delete [] w2;
	delete [] u2;
	delete [] v2;

	return bsr;
}

void saveResults(string patch, int cmax, int o, int *s, int *w, int **p)
{
	std::ofstream resultsFile(patch.c_str(), std::ios::out 
							| std::ios::out);

	resultsFile << cmax << "\n" << o << "\n";
	
	for(int i = 0; i < o; i++)
		resultsFile << w[i] << " ";

	resultsFile << "\n";

	for(int i = 0; i < o; i++)
		resultsFile << s[i] << " ";

	resultsFile << "\n";

	for(int i = 0; i < o; i++)
		resultsFile << p[i][w[i]] << " ";

	resultsFile.close();
}

int main(int argc, char **argv)
{
	if( argc != 6 && argc != 3)
	{
		std::cout << "Arguments are: method type (0 - list scheduling, 1 - beam search), alpha, beta, xi (beam search parameters only) and the instance filename." << std::endl;
		exit(-1);
	}
	
	int method = atoi( argv[1] );
	double alpha;
	double beta;
	double xi;
	string instancePath;
	
	if( method == 1 )
	{
		if( argc != 6 )
		{
			std::cout << "Arguments are: method type (0 to execute the list scheduling or 1 to execute the beam search), alpha, beta, xi (that are parameters only for the beam search method) and the instance path." << std::endl;
			exit(-1);
		}
		
		alpha = atof( argv[2] );
		beta = atof( argv[3] );
		xi = atof( argv[4] );
		instancePath = argv[5];
	}
	else if ( method == 0 )
	{
		if( argc != 3 )
		{
			std::cout << "Arguments are: method type (0 to execute the list scheduling or 1 to execute the beam search), alpha, beta, xi (that are parameters only for the beam search method) and the instance path." << std::endl;
			exit(-1);
		}
		
		instancePath = argv[2];
	}
	else
	{
		std::cout << "Wrong method type. The first argument must be 0 (to execute list scheduling) or 1 (to execute the beam search)." << std::endl;
		exit(-1);
	}

	Instance instance(instancePath);	
		
	float startTime;
	float endTime;
	
	int Cmax;
	int *s = new int[instance.o];
	int *w = new int[instance.o];
	
	if( method == 0 )
	{
		startTime = static_cast<float>(clock()) / CLOCKS_PER_SEC;
		ListSchedulingReturn r = listScheduling(instance.m, instance.o, instance.F, instance.p, instance.S, instance.P);		
		endTime = static_cast<float>(clock()) / CLOCKS_PER_SEC;
		
		Cmax = r.Cmax;
		
		for(int i = 0; i < instance.o; i++)
		{
			s = r.s;
			w = r.w;
		}
	}
	else
	{
		startTime = static_cast<float>(clock()) / CLOCKS_PER_SEC;
		BeamSearchReturn r = beamSearchIni(instance.m, instance.o, instance.F, instance.p, instance.S, instance.P, alpha, beta, xi);		
		endTime = static_cast<float>(clock()) / CLOCKS_PER_SEC;
		
		Cmax = r.Cmax;
		
		for(int i = 0; i < instance.o; i++)
		{
			s = r.s;
			w = r.w;
		}
	}
		
	cout << " - Best Makespan for instance " << instancePath << ": " << 
		Cmax << " in " << endTime - startTime << " seconds.\n";

	std::size_t bar = instancePath.find_first_of("/");
	std::size_t dot = instancePath.find_first_of(".");

	string resultsPatch = instancePath.substr(bar+1, dot-bar-1);
	resultsPatch.insert(0, "solutions/");

	stringstream ss;
	if( method == 0 )
		ss << "_listscheduling.txt";
	else
		ss << "_" << alpha << "_" << beta << "_" << xi << ".txt";

	resultsPatch.append(ss.str());

	saveResults(resultsPatch, Cmax, instance.o, s, w, instance.p);
	
	delete [] s;
	delete [] w;

	return 0;
}
