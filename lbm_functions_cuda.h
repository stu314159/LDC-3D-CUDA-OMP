#ifndef LBM_FUNCTIONS_CUDA_H
#define LBM_FUNCTIONS_CUDA_H

void comp_density_cuda(float * rho, const float * fIn,
		       const int nnodes,const int numSpd);
void comp_velocity_cuda(float * ux, float * uy, float * uz,
			const float * fIn, const float * ex,
			const float * ey, const float * ez,
			const int nnodes, const int numSpd);

void velocity_BC_cuda(float * fIn, float * ux, float * uy, float * uz,
		      const int * lnl, const int * snl,const float * ex,
		      const float * ey, const float * ez,
		      const float * rho, const float u_bc,
		      const float * w,const int nnodes,
		      const int numSpd);

void comp_fEq_cuda(float * fEq, const float * rho, const float * w,
		   const float * ux, const float * uy, const float * uz,
		   const float * ex, const float * ey, const float * ez,
		   const int nnodes, const int numSpd);
void collideLBGK_cuda(float * fOut, const float * fIn, const float * fEq,
		      const float omega, const int nnodes, const int numSpd);
void bounceBack_cuda(float * fOut, const float * fIn,const int * snl,
		     const int * bb_spd,const int nnodes, const int numSpd);

void stream_cuda(float * fIn, const float * fOut, const float * ex,
		 const float * ey, const float * ez,const int Nx,
		 const int Ny, const int Nz, const int numSpd);

void comp_density_cuda2D(float * rho, const float * fIn, const int Nx, 
			 const int Ny, const int Nz, const int numSpd);

void comp_velocity_cuda2D(float * ux, float * uy, float * uz,
			  const float * fIn, const float * ex,
			  const float * ey, const float * ez,
			  const int Nx, const int Ny, const int Nz,
			  const int numSpd);

void velocity_BC2D_cuda(float * fIn, float * ux, float * uy, float * uz,
			const int * lnl, const int * snl, const float * ex,
			const float * ey, const float * ez, const float * rho,
			const float u_bc, const float * w, const int Nx,
			const int Ny, const int Nz, const int numSpd);

void comp_fEq2D_cuda(float * fEq,const float * rho, const float * w,
		     const float * ux,const float * uy, const float * uz,
		     const float * ex, const float * ey, const float * ez,
		     const int Nx, const int Ny, const int Nz,const int numSpd);

void collideLBGK2D_cuda(float * fOut, const float * fIn, const float * fEq,
			const float omega, const int Nx,const int Ny,
			const int Nz, const int numSpd);
void bounceBack2D_cuda(float * fOut, const float * fIn, const int * snl,
		       const int * bb_spd, const int Nx, const int Ny,
		       const int Nz, const int numSpd);

void comp_speed_cuda(float * U, const float * ux, const float * uy, 
		     const float * uz, const int nnodes);

void pre_collide_15s_cuda(float * fIn, float * fEq, 
			  float * uxG,float * uyG, float * uzG,
			  const int * snl,
			  const int * lnl, const float u_bc,
			  const float * ex, const float * ey,
			  const float * ez, const float * w,
			  const int nnodes, const int numSpd);

void pre_streamLBGK_15s_cuda(float * fIn, float * fOut, const int * snl,
			     const int * lnl, const float u_bc, 
			     const float omega, const float * ex,
			     const float * ey, const float * ez,
			     const float * w, const int nnodes,
			     const int numSpd);

#endif
