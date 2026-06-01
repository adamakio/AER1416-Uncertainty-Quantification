%
%  PlotP3_3D(sol)
%  Create 3D plots of the solution of problem P3 
% (steady-state stochastic diffusion equation)
%
   function PlotP3_3D(sol)
%
% INPUT: sol: solution vector (length 143). This could be a vector that contains the
%        nominal solution, mean or variance obtained by solving Au = b
%

   load P3grid;

   x=p(1,:)';
   y=p(2,:)';
   tri=delaunay(x,y);
   sz=size(p,2);
   z=zeros(sz,1); %Here we assume zero values on the boundary
   z(ind_interior)=sol; %Plug the values of 'sol' at the interior nodes
   trisurf(tri, x, y, z);
   colormap cool;
   colorbar;
   grid off;

