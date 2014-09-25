% visual_check.m

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
    
    