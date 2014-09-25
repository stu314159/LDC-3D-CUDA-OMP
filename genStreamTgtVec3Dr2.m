function stv = genStreamTgtVec3Dr2(Nx,Ny,Nz,ex,ey,ez)

ind = 1:(Nx*Ny*Nz); ind = reshape(ind,[Nx Ny Nz]);

numSpd=length(ex);
nnodes = Nx*Ny*Nz;

stv = zeros(nnodes,numSpd);

for spd = 1:numSpd
   t = circshift(ind,[-ex(spd) -ey(spd) -ez(spd)]); t = t(:);
   stv(:,spd)=t;
        
end