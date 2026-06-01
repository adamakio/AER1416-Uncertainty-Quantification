% =========================================================================
%  Q3.m
%  Sparse Clenshaw-Curtis (Smolyak) quadrature for P1, P2, P3
%
%  Folder layout expected:
%    Q3/
%      Q3.m                          <- this file
%      sparse_grid_cc.m              <- provided
%      sparse_grid_cc_index.m        <- provided
%      sparse_grid_cc_weights.m      <- provided
%      sparse_grid_cfn_size.m        <- provided
%      sparse_grid_ccs_size.m        <- provided
%      sparse_grid_cc_size_old.m     <- provided
%      abscissa_level_closed_nd.m    <- provided
%      cc_abscissa.m                 <- provided
%      cc_weights.m                  <- provided
%      comp_next.m                   <- provided
%      i4_choose.m                   <- provided
%      i4_modp.m                     <- provided
%      i4_mop.m                      <- provided
%      index_to_level_closed.m       <- provided
%      level_to_order_closed.m       <- provided
%      multigrid_index0.m            <- provided
%      multigrid_scale_closed.m      <- provided
%      product_weights_cc.m          <- provided
%      r8vec_direct_product2.m       <- provided
%      r8mat_write.m                 <- provided
%      vec_colex_next2.m             <- provided
%    P3/
%      P3.mat
%      P3grid.mat
%      PlotP3_3D.m
%      PlotP3_contour.m
%      tricontour.m
%
%  Run from the Q3/ directory:
%      cd Q3
%      Q3
% =========================================================================

clear; clc; close all;
addpath('../P3');

% -------------------------------------------------------------------------
%  Problem definitions
% -------------------------------------------------------------------------
% P1: d=10, xi ~ U[-1,1]  (CC nodes already on [-1,1], no rescaling needed)
% P2: d=5,  params ~ U[lb2, ub2]  (rescale from [-1,1])
% P3: d=3,  theta ~ U[-3,3]       (rescale from [-1,1])

lb2 = [0.9, 0.8, 10,  0.1, 0.9];
ub2 = [1.1, 1.2, 12,  0.2, 1.1];
lb3 = -3*ones(1,3);
ub3 =  3*ones(1,3);

% -------------------------------------------------------------------------
%  Level ranges
% -------------------------------------------------------------------------
levels_P1 = 0:4;
levels_P2 = 0:6;
levels_P3 = 0:6;

% =========================================================================
%  P1 sweep
% =========================================================================
fprintf('=== P1 (10-D Quadratic, xi ~ U[-1,1]) ===\n');
fprintf('%6s  %8s  %14s  %14s\n','Level','Pts','Mean','Variance');

d1 = 10;
mu_P1  = zeros(size(levels_P1));
var_P1 = zeros(size(levels_P1));
pts_P1 = zeros(size(levels_P1));

for ki = 1:length(levels_P1)
    lev = levels_P1(ki);
    N   = sparse_grid_cfn_size(d1, lev);
    pts_P1(ki) = N;
    [w, x] = sparse_grid_cc(d1, lev, N);  % x: d1xN, w: 1xN

    % Evaluate P1 at each node (nodes already in [-1,1]^d)
    fvals = zeros(1,N);
    for i = 1:N
        fvals(i) = evalP1(x(:,i));
    end

    % Mean and variance under uniform on [-1,1]^d: divide by 2^d
    vol = 2^d1;
    mu_P1(ki)  = (w * fvals') / vol;
    mu2        = (w * (fvals.^2)') / vol;
    var_P1(ki) = mu2 - mu_P1(ki)^2;

    fprintf('%6d  %8d  %14.6f  %14.6e\n', lev, N, mu_P1(ki), var_P1(ki));
end

% =========================================================================
%  P2 sweep
% =========================================================================
fprintf('\n=== P2 (5-D Mass-Spring-Damper) ===\n');
fprintf('%6s  %8s  %14s  %14s\n','Level','Pts','Mean','Variance');

d2 = 5;
mu_P2  = zeros(size(levels_P2));
var_P2 = zeros(size(levels_P2));
pts_P2 = zeros(size(levels_P2));

% Volume of parameter space
vol2 = prod(ub2 - lb2);

for ki = 1:length(levels_P2)
    lev = levels_P2(ki);
    N   = sparse_grid_cfn_size(d2, lev);
    pts_P2(ki) = N;
    [w, x] = sparse_grid_cc(d2, lev, N);

    % Rescale nodes from [-1,1] to [lb2,ub2]
    theta = 0.5*(ub2' + lb2') + 0.5*(ub2' - lb2') .* x;  % d2 x N

    % Scale factor for weights: prod((bi-ai)/2)
    wscale = prod((ub2 - lb2)/2);

    fvals = zeros(1,N);
    for i = 1:N
        fvals(i) = evalP2(theta(:,i));
    end

    mu_P2(ki)  = (w * fvals') * wscale / vol2;
    mu2        = (w * (fvals.^2)') * wscale / vol2;
    var_P2(ki) = mu2 - mu_P2(ki)^2;

    fprintf('%6d  %8d  %14.6e  %14.6e\n', lev, N, mu_P2(ki), var_P2(ki));
end

% =========================================================================
%  P3 sweep
% =========================================================================
fprintf('\n=== P3 (3-D Random Linear System, u1, theta ~ U[-3,3]) ===\n');
fprintf('%6s  %8s  %14s  %14s\n','Level','Pts','Mean','Variance');

d3 = 3;
mu_P3  = zeros(size(levels_P3));
var_P3 = zeros(size(levels_P3));
pts_P3 = zeros(size(levels_P3));

vol3 = prod(ub3 - lb3);

for ki = 1:length(levels_P3)
    lev = levels_P3(ki);
    N   = sparse_grid_cfn_size(d3, lev);
    pts_P3(ki) = N;
    [w, x] = sparse_grid_cc(d3, lev, N);

    % Rescale from [-1,1] to [-3,3]
    theta = 0.5*(ub3' + lb3') + 0.5*(ub3' - lb3') .* x;

    wscale = prod((ub3 - lb3)/2);

    fvals = zeros(1,N);
    for i = 1:N
        fvals(i) = evalP3(theta(:,i));
    end

    mu_P3(ki)  = (w * fvals') * wscale / vol3;
    mu2        = (w * (fvals.^2)') * wscale / vol3;
    var_P3(ki) = mu2 - mu_P3(ki)^2;

    fprintf('%6d  %8d  %14.6e  %14.6e\n', lev, N, mu_P3(ki), var_P3(ki));
end

% =========================================================================
%  CONVERGENCE PLOTS  (one figure per problem per moment)
% =========================================================================
set(groot,'defaultAxesFontSize',14,'defaultAxesFontName','Helvetica',...
         'defaultLineLineWidth',2,'defaultAxesLineWidth',1.2,...
         'defaultAxesTickLength',[0.015 0.015]);
figure; plot(levels_P1, mu_P1, 'o-b','LineWidth',2.5,'MarkerSize',7);
xlabel('Level','Interpreter','latex'); ylabel('$\mu$','Interpreter','latex'); grid on;
saveas(gcf, 'SCC_P1_mean.png');

figure; plot(levels_P1, var_P1, 's-r','LineWidth',2.5,'MarkerSize',7);
xlabel('Level','Interpreter','latex'); ylabel('$\sigma^2$','Interpreter','latex'); grid on;
saveas(gcf, 'SCC_P1_var.png');

figure; plot(levels_P2, mu_P2, 'o-b','LineWidth',2.5,'MarkerSize',7);
xlabel('Level','Interpreter','latex'); ylabel('$\mu$','Interpreter','latex'); grid on;
saveas(gcf, 'SCC_P2_mean.png');

figure; plot(levels_P2, var_P2, 's-r','LineWidth',2.5,'MarkerSize',7);
xlabel('Level','Interpreter','latex'); ylabel('$\sigma^2$','Interpreter','latex'); grid on;
saveas(gcf, 'SCC_P2_var.png');

figure; plot(levels_P3, mu_P3, 'o-b','LineWidth',2.5,'MarkerSize',7);
xlabel('Level','Interpreter','latex'); ylabel('$\mu$','Interpreter','latex'); grid on;
saveas(gcf, 'SCC_P3_mean_conv.png');

figure; plot(levels_P3, var_P3, 's-r','LineWidth',2.5,'MarkerSize',7);
xlabel('Level','Interpreter','latex'); ylabel('$\sigma^2$','Interpreter','latex'); grid on;
saveas(gcf, 'SCC_P3_var_conv.png');

% =========================================================================
%  P3 FULL FIELD  (highest level)
% =========================================================================
fprintf('\nComputing P3 full field at level 7...\n');
Sdata = load('../P3/P3.mat');
A0 = Sdata.A0; A1 = Sdata.A1; A2 = Sdata.A2; A3 = Sdata.A3; b_vec = Sdata.b;

lev_full = 7;
N_full   = sparse_grid_cfn_size(d3, lev_full);
[w_full, x_full] = sparse_grid_cc(d3, lev_full, N_full);
theta_full = 0.5*(ub3' + lb3') + 0.5*(ub3' - lb3') .* x_full;
wscale3    = prod((ub3 - lb3)/2);

n_nodes   = length(b_vec);
mean_field = zeros(n_nodes,1);
var_field  = zeros(n_nodes,1);

for i = 1:N_full
    th = theta_full(:,i);
    A  = A0 + th(1)*A1 + th(2)*A2 + th(3)*A3;
    u  = A \ b_vec;
    mean_field = mean_field + w_full(i) * u;
    var_field  = var_field  + w_full(i) * u.^2;
end
mean_field = mean_field * wscale3 / vol3;
var_field  = var_field  * wscale3 / vol3 - mean_field.^2;

% Mean field plots
figure('Name','P3 Mean 3D');
PlotP3_3D(mean_field);
saveas(gcf,'SCC_P3_mean_3D.png');

figure('Name','P3 Mean Contour');
PlotP3_contour(mean_field);
saveas(gcf,'SCC_P3_mean_contour.png');

% Variance field plots
figure('Name','P3 Variance 3D');
PlotP3_3D(var_field);
saveas(gcf,'SCC_P3_var_3D.png');

figure('Name','P3 Variance Contour');
load('../P3/P3grid.mat');
x_p = p(1,:)';  y_p = p(2,:)';
tri = delaunay(x_p, y_p);
sz  = size(p,2);
z_v = zeros(sz,1);
z_v(ind_interior) = var_field;
n_lev = 20;
v_lev = linspace(min(z_v), max(z_v), n_lev+2);
v_lev = v_lev(2:end-1);
tricontour(tri, x_p, y_p, z_v, v_lev);
colormap cool; colorbar;
saveas(gcf,'SCC_P3_var_contour.png');

fprintf('Done. All figures saved.\n');

% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================

function f = evalP1(x)
    d     = length(x);
    a0    = 1;
    a     = 1 ./ sqrt(1:d)';
    A     = zeros(d,d);
    for i = 1:d
        for j = 1:d
            if i == j
                A(i,j) = 1/i;
            else
                A(i,j) = min(i,j) / (max(i,j)^2 * (i+j));
            end
        end
    end
    gamma = 0.5;
    f = a0 + a'*x + gamma * x'*A*x;
end

function u_T = evalP2(theta)
    m  = theta(1);  c  = theta(2);  k  = theta(3);
    om = theta(4);  F0 = theta(5);
    T  = 10;
    odefun = @(t,y) [y(2); (F0*cos(om*t) - c*y(2) - k*y(1))/m];
    [~,Y]  = ode45(odefun, [0 T], [0; 0]);
    u_T    = Y(end,1);
end

function u1 = evalP3(theta)
    persistent A0 A1 A2 A3 b_vec
    if isempty(A0)
        S     = load('../P3/P3.mat');
        A0    = S.A0; A1 = S.A1; A2 = S.A2; A3 = S.A3;
        b_vec = S.b;
    end
    A  = A0 + theta(1)*A1 + theta(2)*A2 + theta(3)*A3;
    u  = A \ b_vec;
    u1 = u(1);
end