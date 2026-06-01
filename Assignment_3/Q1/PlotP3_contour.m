%
%  PlotP3_contour(sol)
%  Create contour plots of the solution of problem P3 
% (steady-state stochastic diffusion equation)
%
   function PlotP3_contour(sol)
%
% INPUT: sol: solution vector (length 143). This could be a vector that contains the
%        nominal solution, mean or variance obtained by solving Au = b
%
% NOTE: This function loads P3grid.mat and calls the function in tricontour.m.
%

   levelcontour = 0.005;
   load P3grid;

   x=p(1,:)';
   y=p(2,:)';
   tri=delaunay(x,y);
   sz=size(p,2);
   z=zeros(sz,1); % Here we assume zero values on the boundary (Dirichlet BCs)
   z(ind_interior)=sol; % Insert the values of 'sol' at the interior nodes
   v=0:levelcontour:max(sol)+0.1;
   [C,H]=tricontour(tri, x, y, z, v); %Plot z on the mesh 
   colormap cool;
   colorbar;
