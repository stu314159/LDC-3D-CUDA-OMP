
void comp_density_C(float * rho, const float * fIn, const int nnodes,
		    const int numSpd){
  for(int tid=0;tid<nnodes;tid++){
    float rho_tmp=0.;
    for(int spd=0;spd<numSpd;spd++){
      rho_tmp+=fIn[spd*nnodes+tid];
    }
    rho[tid]=rho_tmp;
  }
}

// void comp_velocity_C(float * ux, float * uy, float * uz,
// 		     const float * fIn, const float * ex,
// 		     const float * ey, const float * ez,
// 		     const int nnodes, const int numSpd){

//   float ux_r = 0.;
//   float uy_r = 0.;
//   float uz_r = 0.;
//   float f_tmp;
//   float rho_r= 0.;
  
