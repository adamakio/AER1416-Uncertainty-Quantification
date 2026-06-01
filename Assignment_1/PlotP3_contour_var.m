function PlotP3_contour_var(sol)
% PlotP3_contour_var:
%   Same as PlotP3_contour but fixes the contour level vector v so it
%   spans the actual data range of sol, rather than always starting at 0.
%   This prevents tricontour from erroring on small-valued fields like
%   variance, where max(sol) << 0.005 (the hardcoded levelcontour step).

    n_levels = 20;   % number of contour lines to draw
    load P3grid;
    x   = p(1,:)';
    y   = p(2,:)';
    tri = delaunay(x, y);
    sz  = size(p, 2);
    z   = zeros(sz, 1);
    z(ind_interior) = sol;

    % Fix: build v from the actual data range instead of 0:0.005:max(sol)
    v = linspace(min(z), max(z), n_levels + 2);
    v = v(2:end-1);   % exclude exact min/max as tricontour drops boundary levels

    [C, H] = tricontour(tri, x, y, z, v);
    colormap cool;
    colorbar;
end