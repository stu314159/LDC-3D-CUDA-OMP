x_sl = logspace(xm,xp,6);
y_sl = logspace(ym,yp,6);
z_sl = logspace(zm,zp,6);

[Xsl,Ysl,Zsl]=meshgrid(x_sl,y_sl,z_sl);

Xsl=Xsl(:);
Ysl=Ysl(:);
Zsl=Zsl(:);

Usl = reshape(ux,[Nx Ny Nz]);
Vsl = reshape(uy,[Nx Ny Nz]);
Wsl = reshape(uz,[Nx Ny Nz]);

XYZ = stream3(Usl,Vsl,Wsl,Xsl,Ysl,Zsl);