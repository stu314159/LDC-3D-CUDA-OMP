#define TPB 8



__global__ void ldc_D3Q15_LBGK_ts(float * fOut,const float * fIn,
				  const int * snl, const int * lnl,
				  const float u_bc,const float omega,
				  const int Nx, const int Ny,
				  const int Nz){
  int X=threadIdx.x+blockIdx.x*blockDim.x;
  int Y=threadIdx.y+blockIdx.y*blockDim.y;
  int Z=threadIdx.z+blockIdx.z*blockDim.z;
  if((X<Nx)&&(Y<Ny)&&(Z<Nz)){
    int tid=X+Y*Nx+Z*Nx*Ny;
    float f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14;
    float cu;
    //load the data into registers
    f0=fIn[tid]; f1=fIn[Nx*Ny*Nz+tid];
    f2=fIn[2*Nx*Ny*Nz+tid]; f3=fIn[3*Nx*Ny*Nz+tid];
    f4=fIn[4*Nx*Ny*Nz+tid]; f5=fIn[5*Nx*Ny*Nz+tid];
    f6=fIn[6*Nx*Ny*Nz+tid]; f7=fIn[7*Nx*Ny*Nz+tid];
    f8=fIn[8*Nx*Ny*Nz+tid]; f9=fIn[9*Nx*Ny*Nz+tid];
    f10=fIn[10*Nx*Ny*Nz+tid]; f11=fIn[11*Nx*Ny*Nz+tid];
    f12=fIn[12*Nx*Ny*Nz+tid]; f13=fIn[13*Nx*Ny*Nz+tid];
    f14=fIn[14*Nx*Ny*Nz+tid];

    //compute density
    float rho = f0+f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f11+f12+f13+f14;
    float ux=f1-f2+f7-f8+f9-f10+f11-f12+f13-f14; ux/=rho;
    float uy=f3-f4+f7+f8-f9-f10+f11+f12-f13-f14; uy/=rho;
    float uz=f5-f6+f7+f8+f9+f10-f11-f12-f13-f14; uz/=rho;

    //if it's a lid node, update
    if(lnl[tid]==1){

      //speed 1 ex=1 ey=ez=0. w=1./9.
      cu=3.*(1.)*(-ux);
      f1+=(1./9.)*rho*cu;

      //speed 2 ex=-1 ey=ez=0. w=1./9.
      cu=3.*(-1.)*(-ux);
      f2+=(1./9.)*rho*cu;

      //speed 3 ey=1; ex=ez=0; w=1./9.
      cu=3.*(1.)*(u_bc-uy);
      f3+=(1./9.)*rho*cu;

      //speed 4 ey=-1; ex=ez=0; w=1./9.
      cu=3.*(-1.)*(u_bc-uy);
      f4+=(1./9.)*rho*cu;

      //speed 5 ex=ey=0; ez=1; w=1./9.
      cu=3.*(1.)*(-uz);
      f5+=(1./9.)*rho*cu;

      //speed 6 ex=ey=0; ez=-1; w=1./9.
      cu=3.*(-1.)*(-uz);
      f6+=(1./9.)*rho*cu;

      //speed 7 ex=ey=ez=1; w=1./72.
      cu=3.*((1.)*-ux+(1.)*(u_bc-uy)+(1.)*-uz);
      f7+=(1./72.)*rho*cu;

      //speed 8 ex=-1 ey=ez=1; w=1./72.
      cu=3.*((-1.)*-ux+(1.)*(u_bc-uy)+(1.)*-uz);
      f8+=(1./72.)*rho*cu;

      //speed 9 ex=1 ey=-1 ez=1
      cu=3.0*((1.)*-ux+(-1.)*(u_bc-uy)+(1.)*-uz);
      f9+=(1./72.)*rho*cu;

      //speed 10 ex=-1 ey=-1 ez=1
      cu=3.0*((-1.)*-ux+(-1.)*(u_bc-uy)+(1.)*-uz);
      f10+=(1./72.)*rho*cu;

      //speed 11 ex=1 ey=1 ez=-1
      cu=3.0*((1.)*-ux +(1.)*(u_bc-uy)+(-1.)*-uz);
      f11+=(1./72.)*rho*cu;

      //speed 12 ex=-1 ey=1 ez=-1
      cu=3.0*((-1.)*-ux+(1.)*(u_bc-uy)+(-1.)*-uz);
      f12+=(1./72.)*rho*cu;

      //speed 13 ex=1 ey=-1 ez=-1 w=1./72.
      cu=3.0*((1.)*-ux+(-1.)*(u_bc-uy)+(-1.)*-uz);
      f13+=(1./72.)*rho*cu;
      
      //speed 14 ex=ey=ez=-1 w=1./72.
      cu=3.0*((-1.)*-ux + (-1.)*(u_bc-uy) +(-1.)*-uz);
      f14+=(1./72.)*rho*cu;

      ux=0.; uy=u_bc; uz=0.;

    }//if(lnl[tid]==1)...


    if(snl[tid]==1){
      // 1--2
      cu=f2; f2=f1; f1=cu;
      //3--4
      cu=f4; f4=f3; f3=cu;
      //5--6
      cu=f6; f6=f5; f5=cu;
      //7--14
      cu=f14; f14=f7; f7=cu;
      //8--13
      cu=f13; f13=f8; f8=cu;
      //9--12
      cu=f12; f12=f9; f9=cu;
      //10--11
      cu=f11; f11=f10; f10=cu;

    }else{
      //relax
      //speed 0 ex=ey=ez=0 w=2./9.
      float fEq;
      fEq=rho*(2./9.)*(1.-1.5*(ux*ux+uy*uy+uz*uz));
      f0=f0-omega*(f0-fEq);

      //speed 1 ex=1 ey=ez=0 w=1./9.
      cu=3.*(1.*ux);
      fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		       1.5*(ux*ux+uy*uy+uz*uz));
      f1=f1-omega*(f1-fEq);

      //speed 2 ex=-1 ey=ez=0 w=1./9.
      cu=3.*((-1.)*ux);
      fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		       1.5*(ux*ux+uy*uy+uz*uz));
      f2=f2-omega*(f2-fEq);

      //speed 3 ex=0 ey=1 ez=0 w=1./9.
      cu=3.*(1.*uy);
      fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		       1.5*(ux*ux+uy*uy+uz*uz));
      f3=f3-omega*(f3-fEq);

      //speed 4 ex=0 ey=-1 ez=0 w=1./9.
      cu=3.*(-1.*uy);
      fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		       1.5*(ux*ux+uy*uy+uz*uz));
      f4=f4-omega*(f4-fEq);

      //speed 5 ex=ey=0 ez=1 w=1./9.
      cu=3.*(1.*uz);
      fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		       1.5*(ux*ux+uy*uy+uz*uz));
      f5=f5-omega*(f5-fEq);

      //speed 6 ex=ey=0 ez=-1 w=1./9.
      cu=3.*(-1.*uz);
      fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
		       1.5*(ux*ux+uy*uy+uz*uz));
      f6=f6-omega*(f6-fEq);

      //speed 7 ex=ey=ez=1 w=1./72.
      cu=3.*(ux+uy+uz);
      fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			1.5*(ux*ux+uy*uy+uz*uz));
      f7=f7-omega*(f7-fEq);

      //speed 8 ex=-1 ey=ez=1 w=1./72.
      cu=3.*(-ux+uy+uz);
      fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			1.5*(ux*ux+uy*uy+uz*uz));
      f8=f8-omega*(f8-fEq);

      //speed 9 ex=1 ey=-1 ez=1 w=1./72.
      cu=3.*(ux-uy+uz);
      fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			1.5*(ux*ux+uy*uy+uz*uz));
      f9=f9-omega*(f9-fEq);

      //speed 10 ex=-1 ey=-1 ez=1 w=1/72
      cu=3.*(-ux-uy+uz);
      fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			1.5*(ux*ux+uy*uy+uz*uz));
      f10=f10-omega*(f10-fEq);

      //speed 11 ex=1 ey=1 ez=-1 w=1/72
      cu=3.*(ux+uy-uz);
      fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			1.5*(ux*ux+uy*uy+uz*uz));
      f11=f11-omega*(f11-fEq);

      //speed 12 ex=-1 ey=1 ez=-1 w=1/72
      cu=3.*(-ux+uy-uz);
      fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			1.5*(ux*ux+uy*uy+uz*uz));
      f12=f12-omega*(f12-fEq);

      //speed 13 ex=1 ey=ez=-1 w=1/72
      cu=3.*(ux-uy-uz);
      fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			1.5*(ux*ux+uy*uy+uz*uz));
      f13=f13-omega*(f13-fEq);

      //speed 14 ex=ey=ez=-1 w=1/72
      cu=3.*(-ux-uy-uz);
      fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			1.5*(ux*ux+uy*uy+uz*uz));
      f14=f14-omega*(f14-fEq);

    }//if/else snl


    //now, everybody streams...
    int X_t, Y_t, Z_t;
    int tid_t;

    //speed 0 ex=ey=ez=0
    fOut[tid]=f0;

    //speed 1 ex=1 ey=ez=0
    X_t=X+1; Y_t=Y; Z_t=Z;
    if(X_t==Nx) X_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[Nx*Ny*Nz+tid_t]=f1;

    //speed 2 ex=-1 ey=ez=0;
    X_t=X-1; Y_t=Y; Z_t=Z;
    if(X_t<0) X_t=(Nx-1);
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[2*Nx*Ny*Nz+tid_t]=f2;

    //speed 3 ex=0 ey=1 ez=0
    X_t=X; Y_t=Y+1; Z_t=Z;
    if(Y_t==Ny) Y_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[3*Nx*Ny*Nz+tid_t]=f3;

    //speed 4 ex=0 ey=-1 ez=0
    X_t=X; Y_t=Y-1; Z_t=Z;
    if(Y_t<0) Y_t=(Ny-1);
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[4*Nx*Ny*Nz+tid_t]=f4;

    //speed 5 ex=ey=0 ez=1
    X_t=X; Y_t=Y; Z_t=Z+1;
    if(Z_t==Nz) Z_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[5*Nx*Ny*Nz+tid_t]=f5;

    //speed 6 ex=ey=0 ez=-1
    X_t=X; Y_t=Y; Z_t=Z-1;
    if(Z_t<0) Z_t=(Nz-1);
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[6*Nx*Ny*Nz+tid_t]=f6;

    //speed 7 ex=ey=ez=1
    X_t=X+1; Y_t=Y+1; Z_t=Z+1;
    if(X_t==Nx) X_t=0;
    if(Y_t==Ny) Y_t=0;
    if(Z_t==Nz) Z_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[7*Nx*Ny*Nz+tid_t]=f7;

    //speed 8 ex=-1 ey=1 ez=1
    X_t=X-1; Y_t=Y+1; Z_t=Z+1;
    if(X_t<0) X_t=(Nx-1);
    if(Y_t==Ny) Y_t=0;
    if(Z_t==Nz) Z_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[8*Nx*Ny*Nz+tid_t]=f8;

    //speed 9 ex=1 ey=-1 ez=1
    X_t=X+1; Y_t=Y-1; Z_t=Z+1;
    if(X_t==Nx) X_t=0;
    if(Y_t<0) Y_t=(Ny-1);
    if(Z_t==Nz) Z_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[9*Nx*Ny*Nz+tid_t]=f9;

    //speed 10 ex=-1 ey=-1 ez=1
    X_t=X-1; Y_t=Y-1; Z_t=Z+1;
    if(X_t<0) X_t=(Nx-1);
    if(Y_t<0) Y_t=(Ny-1);
    if(Z_t==Nz) Z_t=0;
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[10*Nx*Ny*Nz+tid_t]=f10;

    //speed 11 ex=1 ey=1 ez=-1
    X_t=X+1; Y_t=Y+1; Z_t=Z-1;
    if(X_t==Nx) X_t=0;
    if(Y_t==Ny) Y_t=0;
    if(Z_t<0) Z_t=(Nz-1);
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[11*Nx*Ny*Nz+tid_t]=f11;

    //speed 12 ex=-1 ey=1 ez=-1
    X_t=X-1; Y_t=Y+1; Z_t=Z-1;
    if(X_t<0) X_t=(Nx-1);
    if(Y_t==Ny) Y_t=0;
    if(Z_t<0) Z_t=(Nz-1);
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[12*Nx*Ny*Nz+tid_t]=f12;

    //speed 13 ex=1 ey=-1 ez=-1
    X_t=X+1; Y_t=Y-1; Z_t=Z-1;
    if(X_t==Nx) X_t=0;
    if(Y_t<0) Y_t=(Ny-1);
    if(Z_t<0) Z_t=(Nz-1);
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[13*Nx*Ny*Nz+tid_t]=f13;

    //speed 14 ex=ey=ez=-1
    X_t=X-1; Y_t=Y-1; Z_t=Z-1;
    if(X_t<0) X_t=(Nx-1);
    if(Y_t<0) Y_t=(Ny-1);
    if(Z_t<0) Z_t=(Nz-1);
    tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
    fOut[14*Nx*Ny*Nz+tid_t]=f14;



  }//if(X<Nx...
}


void ldc_D3Q15_LBGK_ts_cuda(float * fOut, const float * fIn,
			    const int * snl, const int * lnl,
			    const float u_bc, const float omega,
			    const int Nx, const int Ny,
			    const int Nz){
  dim3 BLOCKS(TPB,TPB,1);
  dim3 GRIDS((Nx+TPB-1)/TPB,(Ny+TPB-1)/TPB,Nz);
  ldc_D3Q15_LBGK_ts<<<GRIDS,BLOCKS>>>(fOut,fIn,snl,lnl,u_bc,
				      omega,Nx,Ny,Nz);


}
