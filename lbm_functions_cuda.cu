#define TPB1 128
#define TPB2D 16
#define TPBS 92

__global__ void pre_streamLBGK_15s(float * fIn, float * fOut,
				    const int * snl, const int * lnl,
				    const float u_bc, const float omega,
				    const float * ex,const float * ey,
				    const float * ez, const float * w,
				    const int nnodes, const int numSpd){
  int tid=threadIdx.x+blockIdx.x*blockDim.x;
  if(tid<nnodes){
    //load fIn data into shared memory
    __shared__ float fIns[TPBS][15];
    for(int spd=0;spd<numSpd;spd++){
      fIns[threadIdx.x][spd]=fIn[spd*nnodes+tid];
    }
    //compute density and velocity
    float rho = 0.; float ux=0.; float uy=0.; float uz=0.; float f_tmp;
    for(int spd=0;spd<numSpd;spd++){
      f_tmp=fIns[threadIdx.x][spd];
      rho+=f_tmp;
      ux+=f_tmp*ex[spd];
      uy+=f_tmp*ey[spd];
      uz+=f_tmp*ez[spd];
    }
    ux/=rho; uy/=rho; uz/=rho;

    //check for boundary conditions and update fIns
    if(snl[tid]==1){
      ux=0.; uy=0.; uz=0.;
    }
    if(lnl[tid]==1){
      for(int spd=1;spd<numSpd;spd++){
	f_tmp = 3.0*(ex[spd]*(-ux)+ey[spd]*(u_bc-uy)+ez[spd]*(-uz));
	fIns[threadIdx.x][spd]+=w[spd]*rho*f_tmp;
      }
      ux=0.;uy=u_bc;uz=0.;
    }

    //if it's not a bounce-back node, I can compute fEq and relax for each
    //speed
    float cu, fEq;
    
    for(int spd=0;spd<15;spd++){
      cu = 3.0*(ex[spd]*ux+ey[spd]*uy+ez[spd]*uz);
      
      fEq=rho*w[spd]*(1.+cu+0.5*(cu*cu)-
		      (1.5)*(ux*ux+uy*uy+uz*uz));
      fIns[threadIdx.x][spd]=
	fIns[threadIdx.x][spd]-omega*(fIns[threadIdx.x][spd]-fEq);

    }//for(int spd=0...

    if(snl[tid]==1){
      //if it's a solid node, I need to do a swap
      // 1 -- 2
      f_tmp=fIns[threadIdx.x][2];fIns[threadIdx.x][2]=fIns[threadIdx.x][1];
      fIns[threadIdx.x][1]=f_tmp;

      // 3 -- 4
      f_tmp=fIns[threadIdx.x][4];fIns[threadIdx.x][4]=fIns[threadIdx.x][3];
      fIns[threadIdx.x][3]=f_tmp;

      // 5--6
      f_tmp=fIns[threadIdx.x][6];fIns[threadIdx.x][6]=fIns[threadIdx.x][5];
      fIns[threadIdx.x][5]=f_tmp;

      // 7--14
      f_tmp=fIns[threadIdx.x][14];fIns[threadIdx.x][14]=fIns[threadIdx.x][7];
      fIns[threadIdx.x][7]=f_tmp;

      // 8--13
      f_tmp=fIns[threadIdx.x][13];fIns[threadIdx.x][13]=fIns[threadIdx.x][8];
      fIns[threadIdx.x][8]=f_tmp;

      // 9--12
      f_tmp=fIns[threadIdx.x][12];fIns[threadIdx.x][12]=fIns[threadIdx.x][9];
      fIns[threadIdx.x][9]=f_tmp;
  
      // 10--11
      f_tmp=fIns[threadIdx.x][11];fIns[threadIdx.x][11]=fIns[threadIdx.x][10];
      fIns[threadIdx.x][10]=f_tmp; 
    }

    //now write fIns out to fOut
    for(int spd=0;spd<15;spd++){
      fOut[spd*nnodes+tid]=fIns[threadIdx.x][spd];
    }

  }//if(tid<nnodes)...
}


__global__ void pre_collide_15s(float * fIn, float * fEq,
				float * uxG, float * uyG, float * uzG,
				const int * snl,
				const int * lnl,const float u_bc,
				const float * ex,
				const float * ey, const float * ez, 
				const float * w, const int nnodes,
				const int numSpd){

  int tid=threadIdx.x+blockIdx.x*blockDim.x;
  if(tid<nnodes){

    //load fIn data into shared memory
    __shared__ float fIns[TPBS][15];
    for(int spd=0;spd<numSpd;spd++){
      fIns[threadIdx.x][spd]=fIn[spd*nnodes+tid];
    }

    //compute density and velocity
    float rho = 0.; float ux=0.; float uy = 0.; float uz=0.; float f_tmp;
    for(int spd=0;spd<numSpd;spd++){
      f_tmp=fIns[threadIdx.x][spd];
      rho+=f_tmp;
      ux+=f_tmp*ex[spd];
      uy+=f_tmp*ey[spd];
      uz+=f_tmp*ez[spd];
    }
    ux/=rho; uy/=rho; uz/=rho;

    //check for boundary conditions and update fIns
    if(snl[tid]==1){
      ux=0.; uy=0.; uz=0.;
    }
    if(lnl[tid]==1){
      for(int spd=1;spd<numSpd;spd++){
	f_tmp = 3.0*(ex[spd]*(-ux)+ey[spd]*(u_bc-uy)+ez[spd]*(-uz));
	fIns[threadIdx.x][spd]+=w[spd]*rho*f_tmp;
      }
      ux=0.;uy=u_bc;uz=0.;
    }
    uxG[tid]=ux; uyG[tid]=uy; uzG[tid]=uz;

    //compute and store fEq
    for(int spd=0;spd<numSpd;spd++){
      f_tmp = 3.0*(ex[spd]*ux+ey[spd]*uy+ez[spd]*uz);
      fEq[spd*nnodes+tid]=w[spd]*rho*(1. + f_tmp +
				      0.5*(f_tmp*f_tmp)-
				      1.5*(ux*ux+uy*uy+uz*uz));
    }

  }//if(tid<nnodes)...

}

void pre_streamLBGK_15s_cuda(float * fIn, float * fOut,const int * snl,
			     const int * lnl, const float u_bc,
			     const float omega, const float * ex,
			     const float * ey,const float * ez,
			     const float * w, const int nnodes,
			     const int numSpd){
  dim3 BLOCKS(TPBS,1,1);
  dim3 GRIDS((nnodes+TPBS-1)/TPBS,1,1);
  pre_streamLBGK_15s<<<GRIDS,BLOCKS>>>(fIn,fOut,snl,lnl,u_bc,omega,
				       ex,ey,ez,w,nnodes,numSpd);


}

void pre_collide_15s_cuda(float * fIn,float * fEq,
			  float * uxG,float * uyG, float * uzG,
			  const int * snl,const int * lnl,
			  const float u_bc,const float * ex, const float * ey,
			  const float * ez,const float * w,const int nnodes,
			  const int numSpd){

  dim3 BLOCKS(TPBS,1,1);
  dim3 GRIDS((nnodes+TPBS-1)/TPBS,1,1);
  pre_collide_15s<<<GRIDS,BLOCKS>>>(fIn,fEq,uxG,uyG,uzG,snl,lnl,u_bc,
				    ex,ey,ez,w,nnodes,
				    numSpd);

}




__global__ void comp_speed(float * U, const float * ux, const float * uy,
			   const float * uz, const int nnodes){
  int tid=threadIdx.x+blockIdx.x*blockDim.x;
  if(tid<nnodes){
    U[tid]=sqrt(ux[tid]*ux[tid]+uy[tid]*uy[tid]+uz[tid]*uz[tid]);
  }
}

void comp_speed_cuda(float * U, const float * ux, const float * uy,
		     const float * uz, const int nnodes){

  dim3 BLOCKS(TPB1,1,1);
  dim3 GRIDS((nnodes+TPB1-1)/TPB1,1,1);
  comp_speed<<<GRIDS,BLOCKS>>>(U,ux,uy,uz,nnodes);

}

__global__ void comp_density(float * rho, const float * fIn,const int nnodes,
			     const int numSpd){

  int tid=threadIdx.x+blockIdx.x*blockDim.x;
  if(tid<nnodes){
    float rho_tmp = 0.;
    for(int spd=0;spd<numSpd;spd++){
      rho_tmp+=fIn[spd*nnodes+tid];
    }
    rho[tid]=rho_tmp;
  }
}

__global__ void comp_density2D(float * rho, const float * fIn,const int Nx,
			       const int Ny, const int Nz,
			       const int numSpd){

  int X = threadIdx.x+blockIdx.x*blockDim.x;
  int Y = threadIdx.y+blockIdx.y*blockDim.y;
  int Z = threadIdx.z+blockIdx.z*blockDim.z;
  if((X<Nx)&&(Y<Ny)&&(Z<Nz)){
    int tid=X+Y*Nx+Z*Nx*Ny;
    int nnodes=Nx*Ny*Nz;
    float rho_tmp = 0.;
    for(int spd=0;spd<numSpd;spd++){
      rho_tmp+=fIn[spd*nnodes+tid];
    }
    rho[tid]=rho_tmp;
  }
}

void comp_density_cuda2D(float * rho, const float * fIn, const int Nx,
			 const int Ny, const int Nz, const int numSpd){

  dim3 BLOCKS(TPB2D,TPB2D,1);
  dim3 GRIDS((Nx+TPB2D-1)/TPB2D,(Ny+TPB2D-1)/TPB2D,Nz);
  comp_density2D<<<GRIDS,BLOCKS>>>(rho,fIn,Nx,Ny,Nz,numSpd);
}

void comp_density_cuda(float * rho,const float * fIn, const int nnodes,
		       const int numSpd){

  dim3 BLOCKS(TPB1,1,1);
  dim3 GRIDS((nnodes+TPB1-1)/TPB1,1,1);
  comp_density<<<GRIDS,BLOCKS>>>(rho,fIn,nnodes,numSpd);
}

__global__ void comp_velocity(float * ux,float * uy, float * uz,
			      const float * fIn, 
			      const float * ex, const float * ey,
			      const float * ez, const int nnodes,
			      const int numSpd){
  int tid=threadIdx.x+blockIdx.x*blockDim.x;
  if(tid<nnodes){
    float ux_r = 0.;
    float uy_r = 0.;
    float uz_r = 0.;
    float f_tmp; float rho_r = 0.;
    for(int spd=0;spd<numSpd;spd++){
      f_tmp = fIn[spd*nnodes+tid];
      rho_r+=f_tmp;
      ux_r+=ex[spd]*f_tmp;
      uy_r+=ey[spd]*f_tmp;
      uz_r+=ez[spd]*f_tmp;
    }
    ux[tid]=ux_r/rho_r;
    uy[tid]=uy_r/rho_r;
    uz[tid]=uz_r/rho_r;

  }
}


__global__ void comp_velocity2D(float * ux, float * uy, float * uz,
				const float * fIn, const float * ex,
				const float * ey, const float * ez,
				const int Nx, const int Ny, const int Nz,
				const int numSpd){

  int X=threadIdx.x+blockIdx.x*blockDim.x;
  int Y=threadIdx.y+blockIdx.y*blockDim.y;
  int Z=threadIdx.z+blockIdx.z*blockDim.z;
  if((X<Nx)&&(Y<Ny)&&(Z<Nz)){
    int tid=X+Y*Nx+Z*Nx*Ny;
    int nnodes=Nx*Ny*Nz;
    float ux_r = 0.;
    float uy_r = 0.;
    float uz_r = 0.;
    float f_tmp; float rho_r = 0.;
    for(int spd=0;spd<numSpd;spd++){
      f_tmp = fIn[spd*nnodes+tid];
      rho_r+=f_tmp;
      ux_r+=ex[spd]*f_tmp;
      uy_r+=ey[spd]*f_tmp;
      uz_r+=ez[spd]*f_tmp;
    }
    ux[tid]=ux_r/rho_r;
    uy[tid]=uy_r/rho_r;
    uz[tid]=uz_r/rho_r;

  }
}

void comp_velocity_cuda2D(float * ux, float * uy, float * uz,
			  const float * fIn, const float * ex,
			  const float * ey, const float * ez,
			  const int Nx, const int Ny, const int Nz,
			  const int numSpd){

  dim3 BLOCKS(TPB2D,TPB2D,1);
  dim3 GRIDS((Nx+TPB2D-1)/TPB2D,(Ny+TPB2D-1)/TPB2D,Nz);
  comp_velocity2D<<<GRIDS,BLOCKS>>>(ux,uy,uz,fIn,ex,ey,ez,Nx,Ny,Nz,numSpd);

}
void comp_velocity_cuda(float * ux, float * uy, float * uz,
			      const float * fIn, const float * ex,
			      const float * ey, const float * ez,
			      const int nnodes, const int numSpd){

  dim3 BLOCKS(TPB1,1,1);
  dim3 GRIDS((nnodes+TPB1-1)/TPB1,1,1);
  comp_velocity<<<GRIDS,BLOCKS>>>(ux,uy,uz,fIn,ex,ey,ez,nnodes,numSpd);

}


__global__ void velocity_BC(float * fIn, float * ux, float * uy, float * uz,
			    const int * lnl, const int * snl,
			    const float * ex, const float * ey,
			    const float * ez, const float * rho, const float u_bc,
			    const float * w,const int nnodes, const int numSpd){

  int tid=threadIdx.x+blockIdx.x*blockDim.x;
  if(tid<nnodes){
    if(lnl[tid]==1){
      float rho_r = rho[tid];
      float ux_r = ux[tid];
      float uy_r = uy[tid];
      float uz_r = uz[tid];
      float dx = -ux_r;
      float dy = u_bc-uy_r;
      float dz = -uz_r;
      float cu;
      for(int spd=1;spd<numSpd;spd++){
	cu = 3.0*(ex[spd]*dx+ey[spd]*dy+ez[spd]*dz);
	fIn[spd*nnodes+tid]+=w[spd]*rho_r*cu;

      }
      ux[tid]=0.;
      uy[tid]=u_bc;
      uz[tid]=0.;
    }//if(lnl[tid]==1)...
    if(snl[tid]==1){
      ux[tid]=0.;
      uy[tid]=0.;
      uz[tid]=0.;
    }
  }
}


__global__ void velocity_BC2D(float * fIn, float * ux, float * uy, float * uz,
			 const int * lnl, const int * snl,
			 const float * ex, const float * ey, const float * ez,
			 const float * rho, const float u_bc,
			 const float * w, const int Nx, const int Ny, 
			 const int Nz,const int numSpd){

  int X = threadIdx.x+blockIdx.x*blockDim.x;
  int Y = threadIdx.y+blockIdx.y*blockDim.y;
  int Z = threadIdx.z+blockIdx.z*blockDim.z;
  if((X<Nx)&&(Y<Ny)&&(Z<Nz)){
    int tid = X+Y*Nx+Z*Nx*Ny;
    int nnodes=Nx*Ny*Nz;
    if(lnl[tid]==1){
      float rho_r = rho[tid];
      float ux_r = ux[tid];
      float uy_r = uy[tid];
      float uz_r = uz[tid];
      float dx = -ux_r;
      float dy = u_bc-uy_r;
      float dz = -uz_r;
      float cu;
      for(int spd=1;spd<numSpd;spd++){
	cu = 3.0*(ex[spd]*dx+ey[spd]*dy+ez[spd]*dz);
	fIn[spd*nnodes+tid]+=w[spd]*rho_r*cu;

      }
      ux[tid]=0.;
      uy[tid]=u_bc;
      uz[tid]=0.;
    }//if(lnl[tid]==1)...
    if(snl[tid]==1){
      ux[tid]=0.;
      uy[tid]=0.;
      uz[tid]=0.;
    }
  }
}

void velocity_BC2D_cuda(float * fIn, float * ux, float * uy, float * uz,
		   const int *lnl, const int * snl, const float * ex,
		   const float * ey, const float * ez, const float * rho,
		   const float u_bc, const float * w, const int Nx,
		   const int Ny, const int Nz, const int numSpd){

  dim3 BLOCKS(TPB2D,TPB2D,1);
  dim3 GRIDS((Nx+TPB2D-1)/TPB2D,(Ny+TPB2D-1)/TPB2D,Nz);
  velocity_BC2D<<<GRIDS,BLOCKS>>>(fIn,ux,uy,uz,lnl,snl,ex,ey,ez,
				  rho,u_bc,w,Nx,Ny,Nz,numSpd);

}

void velocity_BC_cuda(float * fIn,float * ux, float * uy, float * uz,
		      const int * lnl, const int * snl, const float * ex, 
		      const float * ey,
		      const float * ez,const float * rho, const float u_bc,
		      const float * w, const int nnodes,const int numSpd){

  dim3 BLOCKS(TPB1,1,1);
  dim3 GRIDS((nnodes+TPB1-1)/TPB1,1,1);
  velocity_BC<<<GRIDS,BLOCKS>>>(fIn,ux,uy,uz,lnl,snl,ex,ey,ez,rho,u_bc,w,nnodes,numSpd);


}

__global__ void comp_fEq(float * fEq,const float * rho, const float * w,
			 const float * ux, const float * uy, const float * uz,
			 const float * ex, const float * ey, const float * ez,
			 const int nnodes,const int numSpd){

  int tid=threadIdx.x+blockIdx.x*blockDim.x;
  if(tid<nnodes){
    float ux_r = ux[tid];
    float uy_r = uy[tid];
    float uz_r = uz[tid];
    float rho_r = rho[tid];
    float cu;
    for(int spd=0;spd<numSpd;spd++){
      cu = 3.0*(ex[spd]*ux_r+ey[spd]*uy_r+ez[spd]*uz_r);
      fEq[spd*nnodes+tid]=w[spd]*rho_r*(1.0+cu+(0.5)*(cu*cu)-
					1.5*(ux_r*ux_r+uy_r*uy_r+
					     uz_r*uz_r));

    }
  }
}

__global__ void comp_fEq2D(float * fEq, const float * rho, const float * w,
			   const float * ux, const float * uy, const float * uz,
			   const float * ex, const float * ey, const float * ez,
			   const int Nx, const int  Ny, const int Nz,
			   const int numSpd){

  int X = threadIdx.x+blockIdx.x*blockDim.x;
  int Y = threadIdx.y+blockIdx.y*blockDim.y;
  int Z = threadIdx.z+blockIdx.z*blockDim.z;
  if((X<Nx)&&(Y<Ny)&&(Z<Nz)){
    int tid = X+Y*Nx+Z*Nx*Ny;
    int nnodes=Nx*Ny*Nz;
    float ux_r = ux[tid];
    float uy_r = uy[tid];
    float uz_r = uz[tid];
    float rho_r = rho[tid];
    float cu;
    for(int spd=0;spd<numSpd;spd++){
      cu = 3.0*(ex[spd]*ux_r+ey[spd]*uy_r+ez[spd]*uz_r);
      fEq[spd*nnodes+tid]=w[spd]*rho_r*(1.0+cu+(0.5)*(cu*cu)-
					1.5*(ux_r*ux_r+uy_r*uy_r+
					     uz_r*uz_r));

    }

  }
}

void comp_fEq2D_cuda(float * fEq, const float * rho, const float * w,
		     const float * ux, const float * uy, const float * uz,
		     const float * ex, const float * ey, const float * ez,
		     const int Nx, const int Ny, const int Nz,
		     const int numSpd){
  dim3 BLOCKS(TPB2D,TPB2D,1);
  dim3 GRIDS((Nx+TPB2D-1)/TPB2D,(Ny+TPB2D-1)/TPB2D,Nz);
  comp_fEq2D<<<GRIDS,BLOCKS>>>(fEq,rho,w,ux,uy,uz,ex,ey,ez,Nx,Ny,Nz,numSpd);


}

void comp_fEq_cuda(float * fEq,const float * rho, const float * w,
		   const float * ux, const float *uy, const float * uz,
		   const float * ex,const float * ey, const float * ez,
		   const int nnodes,const int numSpd){

  dim3 BLOCKS(TPB1,1,1);
  dim3 GRIDS((nnodes+TPB1-1)/TPB1,1,1);
  comp_fEq<<<GRIDS,BLOCKS>>>(fEq,rho,w,ux,uy,uz,ex,ey,ez,nnodes,numSpd);

}

__global__ void collideLBGK(float * fOut, const float * fIn, const float * fEq,
			    const float omega,const int nnodes,const int numSpd){

  int tid=threadIdx.x+blockIdx.x*blockDim.x;
  if(tid<nnodes){
    for(int spd=0;spd<numSpd;spd++){
      fOut[spd*nnodes+tid]=fIn[spd*nnodes+tid]-omega*(fIn[spd*nnodes+tid]-fEq[spd*nnodes+tid]);
    }
  }
}

__global__ void collideLBGK2D(float * fOut, const float * fIn, const float * fEq,
			      const float omega, const int Nx, const int Ny,
			      const int Nz, const int numSpd){

  int X = threadIdx.x+blockIdx.x*blockDim.x;
  int Y = threadIdx.y+blockIdx.y*blockDim.y;
  int Z = threadIdx.z+blockIdx.z*blockDim.z;
  if((X<Nx)&&(Y<Ny)&&(Z<Nz)){
    int nnodes=Nx*Ny*Nz;
    int tid=X+Y*Nx+Z*Nx*Ny;
    for(int spd=0;spd<numSpd;spd++){
      fOut[spd*nnodes+tid]=fIn[spd*nnodes+tid]-omega*(fIn[spd*nnodes+tid]-
						      fEq[spd*nnodes+tid]);
    }
  }
}

void collideLBGK2D_cuda(float * fOut, const float * fIn, const float * fEq,
		   const float omega, const int Nx, const int Ny,
		   const int Nz, const int numSpd){
  dim3 BLOCKS(TPB2D,TPB2D,1);
  dim3 GRIDS((Nx+TPB2D-1)/TPB2D,(Ny+TPB2D-1)/TPB2D,Nz);
  collideLBGK2D<<<GRIDS,BLOCKS>>>(fOut,fIn,fEq,omega,Nx,Ny,Nz,numSpd);

}

void collideLBGK_cuda(float * fOut, const float * fIn, const float * fEq,
		      const float omega, const int nnodes,const int numSpd){


  dim3 BLOCKS(TPB1,1,1);
  dim3 GRIDS((nnodes+TPB1-1)/TPB1,1,1);
  collideLBGK<<<GRIDS,BLOCKS>>>(fOut,fIn,fEq,omega,nnodes,numSpd);


}

__global__ void bounceBack(float * fOut, const float * fIn, const int * snl,
			   const int * bb_spd, const int nnodes, const int numSpd){

  int tid=threadIdx.x+blockIdx.x*blockDim.x;
  if(tid<nnodes){
    if(snl[tid]==1){
      for(int spd=1;spd<numSpd;spd++){
	fOut[bb_spd[spd]*nnodes+tid]=fIn[spd*nnodes+tid];
      }
    }
  }
}

__global__ void bounceBack2D(float * fOut, const float * fIn, const int * snl,
			     const int * bb_spd, const int Nx, const int Ny,
			     const int Nz, const int numSpd){

  int X = threadIdx.x+blockIdx.x*blockDim.x;
  int Y = threadIdx.y+blockIdx.y*blockDim.y;
  int Z = threadIdx.z+blockIdx.z*blockDim.z;
  if((X<Nx)&&(Y<Ny)&&(Z<Nz)){
    int tid=X+Y*Nx+Z*Nx*Ny;
    int nnodes=Nx*Ny*Nz;
    if(snl[tid]==1){
      for(int spd=1;spd<numSpd;spd++){
	fOut[bb_spd[spd]*nnodes+tid]=fIn[spd*nnodes+tid];
      }
    }
  }
}

void bounceBack2D_cuda(float * fOut, const float * fIn, const int * snl,
		       const int * bb_spd, const int Nx, const int Ny,
		       const int Nz, const int numSpd){

  dim3 BLOCKS(TPB2D,TPB2D,1);
  dim3 GRIDS((Nx+TPB2D-1)/TPB2D,(Ny+TPB2D-1)/TPB2D,Nz);
  bounceBack2D<<<GRIDS,BLOCKS>>>(fOut,fIn,snl,bb_spd,Nx,Ny,Nz,numSpd);

}

void bounceBack_cuda(float * fOut, const float * fIn, const int * snl,
		     const int * bb_spd,const int nnodes,const int numSpd){

  dim3 BLOCKS(TPB1,1,1);
  dim3 GRIDS((nnodes+TPB1-1)/TPB1,1,1);
  bounceBack<<<GRIDS,BLOCKS>>>(fOut,fIn,snl,bb_spd,nnodes,numSpd);


}

__global__ void stream(float * fIn, const float * fOut, const float * ex,
		       const float * ey, const float * ez,const int Nx,
		       const int Ny, const int Nz,const int numSpd){

  int X = threadIdx.x+blockIdx.x*blockDim.x;
  int Y = threadIdx.y+blockIdx.y*blockDim.y;
  int Z = threadIdx.z+blockIdx.z*blockDim.z;
  if((X<Nx)&&(Y<Ny)&&(Z<Nz)){
    int nnodes=Nx*Ny*Nz;
    int tid=X+Y*Nx+Z*Nx*Ny;
    int tid_t;
    int X_t,Y_t,Z_t;
    for(int spd=0;spd<numSpd;spd++){
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
      tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
      fIn[spd*nnodes+tid_t]=fOut[spd*nnodes+tid];

    }
  }
}



void stream_cuda(float * fIn, const float * fOut, const float * ex,
		 const float * ey, const float * ez, const int Nx,
		 const int Ny, const int Nz, const int numSpd){

  dim3 BLOCKS(TPB2D,TPB2D,1);
  dim3 GRIDS((Nx+TPB2D-1)/TPB2D,(Ny+TPB2D-1)/TPB2D,Nz);
  stream<<<GRIDS,BLOCKS>>>(fIn,fOut,ex,ey,ez,Nx,Ny,Nz,numSpd);


}
