% =========================================================================
%  Q2.m
%  Quasi-Monte Carlo using Sobol sequences for P1, P2, P3
%
%  Folder layout expected:
%    Q2/
%      Q2.m                     <- this file
%      i4_sobol_generate.m      <- provided
%      i4_sobol.m               <- provided
%      i4_bit_hi1.m             <- provided
%      i4_bit_lo0.m             <- provided
%      prime_ge.m               <- provided
%      i4_uniform.m             <- provided
%      r4_uniform_01.m          <- provided
%      tau_sobol.m              <- provided
%      timestamp.m              <- provided
%    P3/
%      P3.mat
%      P3grid.mat
%      PlotP3_3D.m
%      PlotP3_contour.m
%      tricontour.m
%
%  Run from the Q2/ directory:
%      cd Q2
%      Q2
% =========================================================================

clear; clc; close all;
addpath('../P3');

% -------------------------------------------------------------------------
%  Sample sizes to sweep
% -------------------------------------------------------------------------
N_vals = [1000, 5000, 10000, 20000, 50000, 100000, 200000];
nN     = length(N_vals);

% -------------------------------------------------------------------------
%  Pre-allocate result arrays
% -------------------------------------------------------------------------
mu_P1  = zeros(nN,1);  var_P1  = zeros(nN,1);
mu_P2  = zeros(nN,1);  var_P2  = zeros(nN,1);
mu_P3  = zeros(nN,1);  var_P3  = zeros(nN,1);

% =========================================================================
%  LOOP OVER SAMPLE SIZES
% =========================================================================
for k = 1:nN
    N = N_vals(k);
    fprintf('N = %d\n', N);

    % --- P1: 10-D quadratic, xi ~ U[-1,1] --------------------------------
    d1  = 10;
    lb1 = -ones(1,d1);  ub1 = ones(1,d1);
    % Sobol gives points in [0,1]^d; rescale to [lb,ub]
    S1  = i4_sobol_generate(d1, N, 0);       % d1 x N in [0,1]
    X1  = lb1' + S1 .* (ub1 - lb1)';         % d1 x N in [-1,1]

    vals1 = zeros(1,N);
    for i = 1:N
        vals1(i) = evalP1(X1(:,i));
    end
    mu_P1(k)  = mean(vals1);
    var_P1(k) = var(vals1);

    % --- P2: 5-D mass-spring-damper --------------------------------------
    d2  = 5;
    lb2 = [0.9, 0.8, 10,  0.1, 0.9];
    ub2 = [1.1, 1.2, 12,  0.2, 1.1];
    S2  = i4_sobol_generate(d2, N, 0);
    X2  = lb2' + S2 .* (ub2 - lb2)';

    vals2 = zeros(1,N);
    for i = 1:N
        vals2(i) = evalP2(X2(:,i));
    end
    mu_P2(k)  = mean(vals2);
    var_P2(k) = var(vals2);

    % --- P3: 3-D random linear system, theta ~ U[-3,3] -------------------
    d3  = 3;
    lb3 = -3*ones(1,d3);  ub3 = 3*ones(1,d3);
    S3  = i4_sobol_generate(d3, N, 0);
    X3  = lb3' + S3 .* (ub3 - lb3)';

    vals3 = zeros(1,N);
    for i = 1:N
        vals3(i) = evalP3(X3(:,i));
    end
    mu_P3(k)  = mean(vals3);
    var_P3(k) = var(vals3);
end

% =========================================================================
%  PRINT RESULTS TABLE
% =========================================================================
fprintf('\n=== P1 (10-D Quadratic) ===\n');
fprintf('%10s  %14s  %14s\n','N','Mean','Variance');
for k = 1:nN
    fprintf('%10d  %14.6f  %14.6e\n', N_vals(k), mu_P1(k), var_P1(k));
end

fprintf('\n=== P2 (Mass-Spring-Damper) ===\n');
fprintf('%10s  %14s  %14s\n','N','Mean','Variance');
for k = 1:nN
    fprintf('%10d  %14.6e  %14.6e\n', N_vals(k), mu_P2(k), var_P2(k));
end

fprintf('\n=== P3 (Random Linear System, u1) ===\n');
fprintf('%10s  %14s  %14s\n','N','Mean','Variance');
for k = 1:nN
    fprintf('%10d  %14.6e  %14.6e\n', N_vals(k), mu_P3(k), var_P3(k));
end

% =========================================================================
%  CONVERGENCE PLOTS  (one figure per problem per moment)
% =========================================================================
set(groot,'defaultAxesFontSize',14,'defaultAxesFontName','Helvetica',...
         'defaultLineLineWidth',2,'defaultAxesLineWidth',1.2,...
         'defaultAxesTickLength',[0.015 0.015]);
figure; semilogx(N_vals, mu_P1, 'o-b','LineWidth',2.5,'MarkerSize',7);
xlabel('$N$','Interpreter','latex'); ylabel('$\mu$','Interpreter','latex'); grid on;
saveas(gcf, 'Sobol_P1_mean.png');

figure; semilogx(N_vals, var_P1, 's-r','LineWidth',2.5,'MarkerSize',7);
xlabel('$N$','Interpreter','latex'); ylabel('$\sigma^2$','Interpreter','latex'); grid on;
saveas(gcf, 'Sobol_P1_var.png');

figure; semilogx(N_vals, mu_P2, 'o-b','LineWidth',2.5,'MarkerSize',7);
xlabel('$N$','Interpreter','latex'); ylabel('$\mu$','Interpreter','latex'); grid on;
saveas(gcf, 'Sobol_P2_mean.png');

figure; semilogx(N_vals, var_P2, 's-r','LineWidth',2.5,'MarkerSize',7);
xlabel('$N$','Interpreter','latex'); ylabel('$\sigma^2$','Interpreter','latex'); grid on;
saveas(gcf, 'Sobol_P2_var.png');

figure; semilogx(N_vals, mu_P3, 'o-b','LineWidth',2.5,'MarkerSize',7);
xlabel('$N$','Interpreter','latex'); ylabel('$\mu$','Interpreter','latex'); grid on;
saveas(gcf, 'Sobol_P3_mean_conv.png');

figure; semilogx(N_vals, var_P3, 's-r','LineWidth',2.5,'MarkerSize',7);
xlabel('$N$','Interpreter','latex'); ylabel('$\sigma^2$','Interpreter','latex'); grid on;
saveas(gcf, 'Sobol_P3_var_conv.png');

% =========================================================================
%  P3 FULL FIELD  (use largest N)
% =========================================================================
fprintf('\nComputing P3 full field at N = %d ...\n', N_vals(end));
Sdata  = load('../P3/P3.mat');
A0 = Sdata.A0; A1 = Sdata.A1; A2 = Sdata.A2; A3 = Sdata.A3; b_vec = Sdata.b;

d3    = 3;
lb3   = -3*ones(1,d3);  ub3 = 3*ones(1,d3);
N_big = N_vals(end);
S3big = i4_sobol_generate(d3, N_big, 0);
X3big = lb3' + S3big .* (ub3 - lb3)';

n_nodes  = length(b_vec);
sol_sum  = zeros(n_nodes,1);
sol_sum2 = zeros(n_nodes,1);
for i = 1:N_big
    th = X3big(:,i);
    A  = A0 + th(1)*A1 + th(2)*A2 + th(3)*A3;
    u  = A \ b_vec;
    sol_sum  = sol_sum  + u;
    sol_sum2 = sol_sum2 + u.^2;
end
mean_field = sol_sum  / N_big;
var_field  = sol_sum2 / N_big - mean_field.^2;

% --- Mean field ---
figure('Name','P3 Mean 3D');
PlotP3_3D(mean_field);
saveas(gcf,'Sobol_P3_mean_3D.png');

figure('Name','P3 Mean Contour');
PlotP3_contour(mean_field);
saveas(gcf,'Sobol_P3_mean_contour.png');

% --- Variance field ---
figure('Name','P3 Variance 3D');
PlotP3_3D(var_field);
saveas(gcf,'Sobol_P3_var_3D.png');

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
saveas(gcf,'Sobol_P3_var_contour.png');

fprintf('Done. All figures saved to Q2/ folder.\n');

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
        S   = load('../P3/P3.mat');
        A0  = S.A0; A1 = S.A1; A2 = S.A2; A3 = S.A3;
        b_vec = S.b;
    end
    A  = A0 + theta(1)*A1 + theta(2)*A2 + theta(3)*A3;
    u  = A \ b_vec;
    u1 = u(1);
end