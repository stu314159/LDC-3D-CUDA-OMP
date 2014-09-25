//OpenMP include
#include <omp.h>

//CUDA includes
#include <cuda_runtime.h>

//C++ includes
#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>
#include <string>
#include <sstream>

// My Includes
#include "lbm_utils.h"
#include "lbm_functions_cuda.h"
#include "lbm_functions.h"
#include "ldc_3D_LBGK_ts.h"
#include "lattice_vars.h"

const bool WRITE_OUTPUT=false;
const int TS_RPT=100;
const int HALO=1;
const int size=4; //would like to make this dynamic

using namespace std;

int main(int argc, char * argv[]){

  int ndev;
  omp_set_num_threads(size);

  cudaGetDeviceCount(&ndev);
  if(ndev<size){
    cout << "Not enough devices for all the threads!" << endl;
    return 1;
  }

  
  float * g_in_p_e[size];//<--fOut for even
  float * g_in_m_e[size];//<--fOut for even
  float * g_out_p_e[size];//<--fIn for even
  float * g_out_m_e[size];//<--fIn for even

 
  float * g_in_p_o[size];//<--fIn for odd
  float * g_in_m_o[size];//<--fIn for odd
  float * g_out_p_o[size];//<--fOut for odd
  float * g_out_m_o[size];//<--fOut for odd

  int  numSlices[size];

#pragma omp parallel shared(g_in_p_e,g_out_p_e,g_in_m_e,g_out_m_e,g_in_p_o,g_in_m_o,g_out_p_o,g_out_m_o,numSlices)
  {//<--start of parallel region

    //which thread am I?
    int rank = omp_get_thread_num();
    //select the corresponding CUDA device
    cudaSetDevice(rank);
    //get rank of neighboring processes

    int nd_m,nd_p; //<-- node minus, node plus
    nd_m = rank-1;
    if(nd_m<0)
      nd_m=(size-1);

    nd_p = rank+1;
    if(nd_p==size)
      nd_p=0;

    //enable peer access for adjacent threads
    cudaDeviceEnablePeerAccess(nd_m,0);
    cudaDeviceEnablePeerAccess(nd_p,0);

    //prepare an output file name
    //each process will write their portion of the output density to a 
    //separate file.  Post-processing script will put it all together.
    string outfileName("output_density");
    stringstream rankString;
    rankString << rank;
    outfileName+=rankString.str();
    outfileName+=".lbm";

    //all processes need this input data (better/worse than BCST?)
    int Num_ts;
    int Nx;
    int Ny;
    int Nz;
    int numSpd;
    float u_bc;
    float rho_init;
    float omega;
    int dynamics;
    int impl; //<-- not used this time...

 
    ifstream input_params("params.lbm",ios::in);
    input_params >> Num_ts;
    input_params >> Nx;
    input_params >> Ny;
    input_params >> Nz;
    input_params >> numSpd;
    input_params >> u_bc;
    input_params >> rho_init;
    input_params >> omega;
    input_params >> dynamics;
    input_params >> impl;
    input_params.close();

    // get lattice parameters for appropriate lattice type
 
    float * ex;
    float * ey;
    float * ez;
    float * w;
    int * bb_spd;
    float * M = new float[numSpd*numSpd];
  
    // -- if using MRT, load the matrix operator ----------   
    if(dynamics==3){
      ifstream omega_op("M.lbm",ios::in);
      for(int rows=0;rows<numSpd;rows++){
	for(int cols=0;cols<numSpd;cols++){
	  omega_op >> M[rows*numSpd+cols];
	}
      }
    }
    // -----------------------------------------------------
  
    switch (numSpd){
      //lattice parameters are defined in lattice_vars.h
    case (15):
      ex = ex15;
      ey = ey15;
      ez = ez15;
      w = w15;
      bb_spd = bb15;
      //M = M15;
      break;

    case(19):
      ex = ex19;
      ey = ey19;
      ez = ez19;
      w = w19;
      bb_spd=bb19;
      //M = M19;
      break;

    case(27):
      ex = ex27;
      ey = ey27;
      ez = ez27;
      w = w27;
      bb_spd = bb27;
      //M = NULL; //none implemented for 27-speed model yet.
      break;

    }

    // sort out geometric partition 

    int numMySlices = Nz/size;//<-- floor Nz/size
    if(rank<(Nz%size))//<-- add one starting at rank 0.
      numMySlices+=1;
    

    int numMyNodes=Nx*Ny*numMySlices;//<-- number of lattice points for a given process

    int firstSlice,lastSlice;
    firstSlice=(Nz/size)*rank;//<--starting point
    if((Nz%size)<rank){ //<-- add the minimum of the rank or Nz%size
      firstSlice+=(Nz%size);
    }else{
      firstSlice+=rank;
    }

    lastSlice=firstSlice+numMySlices-1;
    int totalSlices=numMySlices+2*HALO;//<-- this can be used like Nz for a local partition
    numSlices[rank]=totalSlices;


    int nnodes = totalSlices*Nx*Ny; //local value for nnodes.

    //declare and allocate fIn on the host
    float * fIn;
    cudaHostAlloc((void**)&fIn,nnodes*numSpd*sizeof(float),cudaHostAllocPortable);


    if(rank==0)
      cout << "Initializing lattice data...." << endl;

    //initialize fIn
    
    //this needs to be implemented...
    initialize_lattice_partitionT(fIn,w,rho_init,Nx,Ny,totalSlices,numSpd);


 //declare and allocate lnl and snl
    int * lnl = new int[totalSlices*Nx*Ny];
    int * snl = new int[totalSlices*Nx*Ny];

    if(rank==0)
      cout << "Initializing lnl array..." << endl;

 

    initialize_lnl_partition(lnl,firstSlice,numMySlices,HALO,
			     Nx,Ny,Nz);

 
    if(rank==0)
      cout << "Initializing snl array..." << endl;

    initialize_snl_partition(snl,firstSlice,numMySlices,HALO,
			     Nx,Ny,Nz);

   
    int numHALO = (Nx*Ny*HALO);//<--number of values in each HALO buffer.

    //initialize and send data to the GPU
    float * fIn_d;
    float * fOut_d;

    cudaMalloc((void**)&fIn_d,nnodes*numSpd*sizeof(float));
    cudaMalloc((void**)&fOut_d,nnodes*numSpd*sizeof(float));


    //grab the pointers to the first element of the first speed of each array.
    g_in_p_e[rank]=fOut_d+(Nx*Ny*(totalSlices-HALO));
    g_in_p_o[rank]=fIn_d+(Nx*Ny*(totalSlices-HALO));

    g_in_m_e[rank]=fOut_d;
    g_in_m_o[rank]=fIn_d;

    g_out_p_e[rank]=fIn_d+(Nx*Ny*numMySlices);
    g_out_p_o[rank]=fOut_d+(Nx*Ny*numMySlices);

    g_out_m_e[rank]=fIn_d+(Nx*Ny*HALO);
    g_out_m_o[rank]=fOut_d+(Nx*Ny*HALO);


    int * snl_d;
    int * lnl_d;

    cudaMalloc((void**)&snl_d,nnodes*sizeof(int));
    cudaMalloc((void**)&lnl_d,nnodes*sizeof(int));

    cudaMemcpy(fIn_d,fIn,nnodes*numSpd*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(snl_d,snl,nnodes*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(lnl_d,lnl,nnodes*sizeof(int),cudaMemcpyHostToDevice);

#pragma omp barrier
    //time stepping starts here
    for(int ts=0;ts<Num_ts;ts++){
      if((ts+1)%TS_RPT==0){
	if(rank==0){
	  cout << "Executing time step number " << ts+1 << endl;
	}
      }


      if((ts%2)==0){
	ldc_D3Q15_LBGK_ts_cuda(fOut_d,fIn_d,snl_d,lnl_d,u_bc,omega,Nx,Ny,totalSlices);
#pragma omp barrier
	//even-step communications
	for(int spd=0;spd<numSpd;spd++){
	  cudaMemcpyPeer(g_in_p_e[rank]+spd*Nx*Ny*totalSlices,rank,
			 g_out_m_e[nd_p]+spd*Nx*Ny*numSlices[nd_p],nd_p,
			 numHALO*sizeof(float));
	  cudaMemcpyPeer(g_in_m_e[rank]+spd*Nx*Ny*totalSlices,rank,
			 g_out_p_e[nd_m]+spd*Nx*Ny*numSlices[nd_m],nd_m,
			 numHALO*sizeof(float));
	}

#pragma omp barrier
      }else{
	ldc_D3Q15_LBGK_ts_cuda(fIn_d,fOut_d,snl_d,lnl_d,u_bc,omega,Nx,Ny,totalSlices);
#pragma omp barrier
	//odd-step communications
	for(int spd=0;spd<numSpd;spd++){

	  cudaMemcpyPeer(g_in_p_o[rank]+spd*Nx*Ny*totalSlices,rank,
			 g_out_m_o[nd_p]+spd*Nx*Ny*numSlices[nd_p],nd_p,
			 numHALO*sizeof(float));
	  cudaMemcpyPeer(g_in_m_o[rank]+spd*Nx*Ny*totalSlices,rank,
			 g_out_p_o[nd_m]+spd*Nx*Ny*numSlices[nd_m],nd_m,
			 numHALO*sizeof(float));

	}
#pragma omp barrier

      }

      
    }//for(int ts=0...


    if(WRITE_OUTPUT){

      //time stepping complete, get data from the GPU and write to disk
      cudaMemcpy(fIn,fIn_d,nnodes*numSpd*sizeof(float),cudaMemcpyDeviceToHost);

      cout << "rank " << rank << " trying to write output..." << endl;

      float * fIn_m = 
	fIn+(Nx*Ny*HALO);

      ofstream output_dat(outfileName.c_str(),ios::out);
      for(int nd=0;nd<numMyNodes;nd++){
	for(int spd=0;spd<numSpd;spd++){
	  output_dat<<fIn_m[spd*nnodes+nd] << "  ";
	}
	output_dat << endl;
      }
      output_dat.close();
    }

    delete [] M;
    delete [] lnl;
    delete [] snl;
    cudaFreeHost(fIn);

    cudaFree(fIn_d);
    cudaFree(fOut_d);
    cudaFree(lnl_d);
    cudaFree(snl_d);


    
  }//<--end of parallel region
  return 0;
}
