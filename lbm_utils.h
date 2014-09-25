#ifndef LBM_UTILS_H
#define LBM_UTILS_H

#include <iostream>
using namespace std;

extern "C" void gen_ldc_lnl(int * lnl, int Nx, int Ny, int Nz);
extern "C" void gen_ldc_snl(int * snl, int Nx, int Ny, int Nz);
extern "C" void gen_XYZ_rectLattice(float * X,float * Y, float * Z,const int Nx,
			 const int Ny, const int Nz);


template<class T>
void print_array(T * A, const int nrows, const int ncols){
  cout << endl;
  for(int i=0;i<nrows;i++){
    for(int j=0;j<ncols;j++){
      cout << A[i*ncols+j];
      if(j<(ncols-1))
	cout << ", ";
    }
    cout << endl;
  }
  cout << endl;
}

extern "C"
void initialize_lattice(float * f,const float * w,float rho,
			const int nnodes,const int nspd);


void initialize_lattice_partition(float * f, const float * w, const float rho,
				  const int Nx, const int Ny,
				  const int numSlices,const int nspd);

void initialize_lattice_partitionT(float * f, const float * w, const float rho,
				   const int Nx, const int Ny,
				   const int numSlices, const int nspd);

void initialize_lnl_partition(int * lnl, const int firstSlice,
			      const int numMySlices, const int halo,
			      const int Nx, const int Ny, const int Nz);

void initialize_snl_partition(int * snl, const int firstSlice,
			      const int numMySlices, const int halo,
			      const int Nx, const int Ny, const int Nz);


#endif
