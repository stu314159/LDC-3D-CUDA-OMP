#define TPB2D 8

__global__ void ldc_D3Q15_LBGK_ts(float * fOut, const float * fIn, 
				  const int * snl,
				  const int * lnl, const float u_bc,
				  const float omega,const float * ex,
				  const float * ey, const float * ez,
				  const float * w, const int Nx, 
				  const int Ny, const int Nz){

  int X=threadIdx.x+blockIdx.x*blockDim.x;
  int Y=threadIdx.y+blockIdx.y*blockDim.y;
  int Z=threadIdx.z+blockIdx.z*blockDim.z;
  if((X<Nx)&&(Y<Ny)&&(Z<Nz)){
    int tid=X+Y*Nx+Z*Nx*Ny;
    int dof;
    //load fIn data into shared memory
    __shared__ float fIns[TPB2D][TPB2D][15];
    for(int spd=0;spd<15;spd++){
      for(int y=0;y<TPB2D;y++){
	for(int x=0;x<TPB2D;x++){
	  dof=(blockIdx.x*blockDim.x+x)+
	    (blockIdx.y*blockDim.y+y)*Nx+
	    (blockIdx.z*blockDim.z)*Nx*Ny;
	  fIns[y][x][spd]=fIn[spd*(Nx*Ny*Nz)+dof];
	}
      }
    }

    //compute density and velocity
    float rho=0.; float ux=0.; float uy=0.; float uz=0.; float f_tmp;
    for(int spd=0;spd<15;spd++){
      f_tmp=fIns[threadIdx.y][threadIdx.x][spd];
      rho+=f_tmp;
      ux+=f_tmp*ex[spd];
      uy+=f_tmp*ey[spd];
      uz+=f_tmp*ez[spd];
    }
    ux/=rho; uy/=rho; uz/=rho;

    //check for boundary condition and update
   

    if(lnl[tid]==1){
      for(int spd=1;spd<15;spd++){
	f_tmp=3.0*(ex[spd]*(-ux)+ey[spd]*(u_bc-uy)+ez[spd]*(-uz));
	fIns[threadIdx.y][threadIdx.x][spd]+=w[spd]*rho*f_tmp;
      }
      ux=0.; uy=u_bc;uz=0.;
    }

    if(snl[tid]==1){
      ux=0.;uy=0.;uz=0.;
      //-- bounce-back here as well..
      // 1--2
      f_tmp=fIns[threadIdx.y][threadIdx.x][2];
      fIns[threadIdx.y][threadIdx.x][2]=fIns[threadIdx.y][threadIdx.x][1];
      fIns[threadIdx.y][threadIdx.x][1]=f_tmp;

      //3 -- 4
      f_tmp=fIns[threadIdx.y][threadIdx.x][4];
      fIns[threadIdx.y][threadIdx.x][4]=fIns[threadIdx.y][threadIdx.x][3];
      fIns[threadIdx.y][threadIdx.x][3]=f_tmp;

      //5 -- 6
      f_tmp=fIns[threadIdx.y][threadIdx.x][6];
      fIns[threadIdx.y][threadIdx.x][6]=fIns[threadIdx.y][threadIdx.x][5];
      fIns[threadIdx.y][threadIdx.x][5]=f_tmp;

      //7 -- 14
      f_tmp=fIns[threadIdx.y][threadIdx.x][14];
      fIns[threadIdx.y][threadIdx.x][14]=fIns[threadIdx.y][threadIdx.x][7];
      fIns[threadIdx.y][threadIdx.x][7]=f_tmp;

      //8--13
      f_tmp=fIns[threadIdx.y][threadIdx.x][13];
      fIns[threadIdx.y][threadIdx.x][13]=fIns[threadIdx.y][threadIdx.x][8];
      fIns[threadIdx.y][threadIdx.x][8]=f_tmp;

      //9--12
      f_tmp=fIns[threadIdx.y][threadIdx.x][12];
      fIns[threadIdx.y][threadIdx.x][12]=fIns[threadIdx.y][threadIdx.x][9];
      fIns[threadIdx.y][threadIdx.x][9]=f_tmp;

      //10--11
      f_tmp=fIns[threadIdx.y][threadIdx.x][11];
      fIns[threadIdx.y][threadIdx.x][11]=fIns[threadIdx.y][threadIdx.x][10];
      fIns[threadIdx.y][threadIdx.x][10]=f_tmp;


      //do not do relaxation on solid nodes since the result
      //is annulled with the bounce-back.
    }else{

      //not a solid node, relaxation
      float cu, fEq;
      for(int spd=0;spd<15;spd++){
	cu = 3.0*(ex[spd]*ux+ey[spd]*uy+ez[spd]*uz);
	fEq=rho*w[spd]*(1.+cu+0.5*(cu*cu)-
			(1.5)*(ux*ux+uy*uy+uz*uz));
	fIns[threadIdx.y][threadIdx.x][spd]-=
	  omega*(fIns[threadIdx.y][threadIdx.x][spd]-fEq);
      }


    }

    //now, everybody streams....
    int X_t,Y_t,Z_t;
    for(int spd=0;spd<15;spd++){

      X_t=X+ex[spd];
      Y_t=Y+ey[spd];
      Z_t=Z+ez[spd];
      if(X_t==Nx)
	X_t=0;
      if(Y_t==Ny)
	Y_t=0;
      if(Z_t==Nz)
	Z_t=0;
      if(X_t<0)
	X_t=(Nx-1);
      if(Y_t<0)
	Y_t=(Ny-1);
      if(Z_t<0)
	Z_t=(Nz-1);
      dof=X_t+Y_t*Nx+Z_t*Nx*Ny;
      fOut[spd*Nx*Ny*Nz+dof]=fIns[threadIdx.y][threadIdx.x][spd];

    }



  }//if (X<Nx...
}


void ldc_D3Q15_LBGK_ts_cuda(float * fOut, const float * fIn, const int * snl,
		       const int * lnl, const float u_bc,
		       const float omega, const float * ex,
		       const float * ey, const float * ez,
		       const float * w, const int Nx,
		       const int Ny, const int Nz){

  dim3 BLOCKS(TPB2D,TPB2D,1);
  dim3 GRIDS((Nx+TPB2D-1)/TPB2D,(Ny+TPB2D-1)/TPB2D,Nz);
  ldc_D3Q15_LBGK_ts<<<GRIDS,BLOCKS>>>(fOut,fIn,snl,lnl,u_bc,
					   omega,ex,ey,ez,w,Nx,Ny,Nz);

}
