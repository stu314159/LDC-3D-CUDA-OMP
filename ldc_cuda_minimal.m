%ldc_cuda.m


% LBGK version only at first...

clear
clc
close('all');

initialization = 0;
% 0 = initialize fIn to zero speed
% 1 = initialize fIn to Poiseuille profile <--not used for this problem

impl = 1;
% 0 = C/C++ LBM implementation
% 1 = C/C++ + CUDA LBM implementation

dynamics = 1;
% 1 = LBGK
% 2 = TRT 
% 3 = MRT <--- D3Q15 and D3Q19 only

lattice_selection = 1;
% 1 = D3Q15
% 2 = D3Q19
% 3 = D3Q27

fluid = 1;
% 1 = glycerin
% 2 = glycol
% 3 = water

Num_ts = 50000;
ts_rep_freq = 100;
plot_freq = Num_ts-1;
Re = 150;

Lx_p = 1;
Ly_p = 1;
Lz_p = 1;

switch fluid
    case 1
        rho_p = 1260;
        nu_p = 1.49/rho_p;
        
    case 2
        rho_p = 965.3;
        nu_p = 0.06/rho_p;
        
    case 3
        rho_p = 1000;
        nu_p = 1e-3/rho_p;
        
end

Lo = Ly_p;
Uavg = nu_p*Re/Lo;

Uo = Uavg;
To = Lo/Uo;

Ld = 1; Td = 1; Ud = (To/Lo)*Uavg;
nu_d = 1/Re;

% convert to LBM units
dt = 5e-4;
Ny_divs = 128;
dx = 1/(Ny_divs-1);
u_lbm = (dt/dx)*Ud;
nu_lbm=(dt/(dx^2))*nu_d;
omega = get_BGK_Omega(nu_lbm);

u_conv_fact = (dt/dx)*(To/Lo);
t_conv_fact = (dt*To);
l_conv_fact = (dx*Lo);
f_conv_fact = (l_conv_fact^2)/(u_conv_fact^2);

Ny = ceil((Ny_divs-1)*(Ly_p/Lo))+1;
Nx = ceil((Ny_divs-1)*(Lx_p/Lo))+1;
Nz = ceil((Ny_divs-1)*(Lz_p/Lo))+1;
nnodes=Nx*Ny*Nz;

switch lattice_selection
    
    case 1
        [w,ex,ey,ez,bb_spd]=D3Q15_lattice_parameters();
        lattice = 'D3Q15';
    case 2
        [w,ex,ey,ez,bb_spd]=D3Q19_lattice_parameters();
        lattice = 'D3Q19';
    case 3
        [w,ex,ey,ez,bb_spd]=D3Q27_lattice_parameters();
        lattice = 'D3Q27';
        
end
numSpd = length(w);

switch dynamics
    
    case 1% BGK
        fEq = zeros(nnodes,numSpd);
    case 2 % TRT
        fEq = zeros(nnodes,numSpd);
        fNEq = zeros(nnodes,numSpd);
        fEven = zeros(nnodes,numSpd);
        fOdd = zeros(nnodes,numSpd);
        
    case 3 % MRT
        fEq = zeros(nnodes,numSpd);
        M = getMomentMatrix(lattice);
        S = getEwMatrixMRT(lattice,omega);
        omega_op = M\(S*M);
        
        
end

rho_lbm = rho_p;

xm = 0; xp = Lx_p;
ym = 0; yp = Ly_p;
zm = 0; zp = Lz_p;
[gcoord,~,faces]=Brick3Dr2(xm,xp,ym,yp,zm,zp,Nx,Ny,Nz);
[nnodes,~]=size(gcoord);
snl = [faces.zx_m; faces.zx_p; faces.xy_m; faces.xy_p; faces.zy_p];
snl = unique(snl);

% lid-node-list is the zy plane where x is at a minimum
% may seem like an odd choice, but it obviously doesn't matter
lnl = faces.zy_m; 
lnl = setxor(lnl,intersect(lnl,snl)); % eliminate solid nodes from inl

ux_p_lid = zeros(length(lnl),1);
uy_p_lid = u_lbm*ones(length(lnl),1);
uz_p_lid = ux_p_lid;


x_space = linspace(xm,xp,Nx);
y_space = linspace(ym,yp,Ny);
z_space = linspace(zm,zp,Nz);
z_pln = z_space(ceil(Nz/2));
vis_nodes = find(gcoord(:,3)==z_pln);


fprintf('Number of Lattice-points = %d.\n',nnodes);
fprintf('Number of time-steps = %d. \n',Num_ts);
%fprintf('Predicted execution time = %g.\n', predicted_ex_time);

fprintf('LBM viscosity = %g. \n',nu_lbm);
fprintf('LBM relaxation parameter (omega) = %g. \n',omega);
fprintf('LBM flow Mach number = %g. \n',u_lbm);

input_string = sprintf('Do you wish to continue? [Y/n] \n');

run_dec = input(input_string,'s');

if ((run_dec ~= 'n') && (run_dec ~= 'N'))
    
    fprintf('Ok! Cross your fingers!! \n');
    
    % write parameters to disk
    % Num_ts
    % Nx
    % Ny
    % Nz
    % numSpd
    % u_bc
    % rho_init
    
    % --- for LBGK version.. --
    % omega
    
    s = system('rm *.lbm');
    
    params = fopen('params.lbm','w');
    fprintf(params,'%d \n',Num_ts);
    fprintf(params,'%d \n',Nx);
    fprintf(params,'%d \n',Ny);
    fprintf(params,'%d \n',Nz);
    fprintf(params,'%d \n',numSpd);
    fprintf(params,'%f \n',u_lbm);
    fprintf(params,'%f \n',rho_lbm);
    fprintf(params,'%f \n',omega);
    fprintf(params,'%d \n',dynamics);
    fprintf(params,'%d \n',impl);
    
    fclose(params);
    
    if(dynamics==3)
        save('M.lbm','omega_op','-ascii');
    end
    
        
    %invoke the executable
    tic;
    system('./clbm_ldc3D');
    run_time = toc;
    fprintf('Lattice Point Updates per second = %g.\n',Nx*Ny*Nz*Num_ts/run_time);
    
    
    % load resuts and visualize (add details later...)
     % read the results
    fIn = load('output_density.lbm','-ascii');
    
    % visualize the data
    % compute density
    rho = sum(fIn,2);
    
    % compute velocities
    ux = (fIn*ex')./rho;
    uy = (fIn*ey')./rho;
    uz = (fIn*ez')./rho;
    
    % set macroscopic and Microscopic Dirichlet-type boundary
    % conditions
    
    % macroscopic BCs
    ux(lnl)=ux_p_lid;
    uy(lnl)=uy_p_lid;
    uz(lnl)=uz_p_lid;
    
    
    % velocity magnitude on mid-box slice
    ux_vp = ux(vis_nodes);
    uy_vp = uy(vis_nodes);
    uz_vp = uz(vis_nodes);
    u_vp = sqrt(ux_vp.^2+uy_vp.^2+uz_vp.^2)./u_conv_fact;
    u_vp =reshape(u_vp,[Ny Nx]);
    %imagesc(u_vp);
    contourf(u_vp,30);
    colorbar
    title('Velocity contour at mid-cavity slice');
    xlabel('x');
    ylabel('y');
    axis equal off
    
    figure(2)
    % density magnitude on mid-box slice
    rho_p = rho(vis_nodes);
    rho_p = reshape(rho_p,[Ny Nx]);
    contourf(rho_p,150);
    colorbar
    title('Density contour at mid-cavity slice');
    xlabel('x');
    ylabel('y');
    axis equal off
    
    % for figure 3, include pressure on bottom plane
    figure(3)
    rho_p = rho(faces.zy_p);
    rho_p = reshape(rho_p,[Nz Ny]);
    contourf(rho_p,50)
    colorbar
    title('Density Contour along bottom plane');
    xlabel('z');
    ylabel('y');
    axis equal off
    colorbar
    
    drawnow
    
    
    
    
else
    fprintf('Run aborted.  Better luck next time!\n');
end

