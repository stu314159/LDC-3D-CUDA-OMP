#ifndef LDC_3D_LBGK_TS_H
#define LDC_3D_LBGK_TS_H

/* void ldc_D3Q15_LBGK_ts_cuda(float * fOut, const float * fIn, const int * snl, */
/* 		       const int * lnl, const float u_bc, */
/* 		       const float omega, const float * ex, */
/* 		       const float * ey, const float * ez, */
/* 		       const float * w, const int Nx, */
/* 		       const int Ny, const int Nz); */

void ldc_D3Q15_LBGK_ts_cuda(float * fOut, const float * fIn,
			    const int * snl, const int * lnl,
			    const float u_bc, const float omega,
			    const int Nx, const int Ny, const int Nz);

#endif
