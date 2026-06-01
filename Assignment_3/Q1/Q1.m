%==========================================================================
%  Q1.m -- Non-Intrusive Polynomial-Chaos Projection for Problem P3
%  AER1416  Assignment 3
%
%  Approximates the first two statistical moments of the solution u(theta)
%  to the random linear system
%             A(theta) u = b,  A(theta) = A0 + sum_{i=1}^3 theta_i A_i,
%  with theta_i ~ U[-3,3] i.i.d., using a total-order Legendre PC basis of
%  degree p = 1 and p = 2.  Projection integrals are evaluated with a
%  tensor-product Gauss-Legendre rule of (p+1) nodes per dimension.
%
%  Notation follows the course lecture slides:
%      u(xi) ~ sum_{i=0}^P  y_i  phi_i(xi),    P+1 = (d+p)! / (d! p!)
%      y_i  =  <u phi_i> / <phi_i^2>           (slide 8)
%      mu   =  y_0,   sigma^2 = sum_{i=1..P} y_i^2 <phi_i^2>    (slide 10)
%  Here d = 3 (number of random inputs) and n = 143 (spatial DOFs).
%
%  Required data files in the working directory:
%     P3.mat        -- contains A0, A1, A2, A3, b
%     P3grid.mat    -- used by the plotting helpers
%     PlotP3_3D.m, PlotP3_contour.m, PlotP3_contour_var.m, tricontour.m
%==========================================================================

clear; close all; clc;

%% -----------------------  1.  Load problem data  -----------------------
fprintf('Loading P3 problem data ...\n');
S  = load('P3.mat');
A0 = S.A0;  A1 = S.A1;  A2 = S.A2;  A3 = S.A3;  b  = S.b;
n_int = length(b);                       % number of interior DOFs (143)
fprintf('System size n = %d\n', n_int);

fig_dir = 'Q1_figs';
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% ---------------  2.  Problem-specific constants / references  ---------
d      = 3;           % number of random inputs
scale  = 3;           % theta_i = scale * xi_i,  xi_i ~ U[-1,1]

% Reference values (Sobol QMC N=2e5, from Assignment 2) for u_1:
ref_mean_u1 = 6.361e-2;
ref_var_u1  = 7.221e-5;

results = struct();   % container for summary table

%% ------------------------  3.  Main loop over p  -----------------------
orders = [1, 2];

for ip = 1:numel(orders)
    p = orders(ip);
    fprintf('\n=========================================================\n');
    fprintf('   Polynomial-Chaos Order  p = %d\n', p);
    fprintf('=========================================================\n');

    % --- 3a. Total-order multi-index set ---------------------------------
    alpha = build_total_order_multiindex(d, p);   % (P+1) x d
    Pp1   = size(alpha, 1);
    fprintf('Number of basis functions  P+1 = %d\n', Pp1);

    % ||phi_i||^2 = <phi_i^2> = prod_j 1/(2*alpha_{i,j}+1)  (lecture slide 10)
    phi_sq_avg = prod(1 ./ (2*alpha + 1), 2);     % (P+1) x 1

    % --- 3b. Tensor-product Gauss-Legendre nodes/weights -----------------
    nq1           = p + 1;                        % 1-D nodes
    [eta, omega]  = gauss_legendre_prob(nq1);     % weights sum to 1

    [X1, X2, X3] = ndgrid(eta, eta, eta);
    [W1, W2, W3] = ndgrid(omega, omega, omega);

    xi_nodes = [X1(:), X2(:), X3(:)];             % Nq x d
    w_nodes  = W1(:) .* W2(:) .* W3(:);           % Nq x 1
    Nq       = size(xi_nodes, 1);
    fprintf('Number of forward solves = %d\n', Nq);

    % --- 3c. Basis matrix Phi(q,i) = phi_i(xi^(q)) -----------------------
    Phi = eval_legendre_basis(xi_nodes, alpha);   % Nq x (P+1)

    % --- 3d. Forward solves ----------------------------------------------
    fprintf('Performing %d linear solves ...\n', Nq);
    U = zeros(n_int, Nq);
    for q = 1:Nq
        th     = scale * xi_nodes(q, :);
        Aq     = A0 + th(1)*A1 + th(2)*A2 + th(3)*A3;
        U(:,q) = Aq \ b;
    end

    % --- 3e. Discrete projection (lecture slide 8) -----------------------
    %  y_{j,i} = (1/<phi_i^2>) * sum_q W_q * u_j(xi_q) * phi_i(xi_q)
    Y = U * (w_nodes .* Phi);            % n x (P+1)
    Y = Y ./ phi_sq_avg.';               % divide each column by <phi_i^2>

    % --- 3f. First two moments (lecture slide 10) ------------------------
    %  mu_{u_j}     = y_{j,0}
    %  sigma^2_{u_j} = sum_{i=1..P} y_{j,i}^2 * <phi_i^2>
    mean_u = Y(:, 1);                                  % n x 1
    var_u  = (Y(:, 2:end).^2) * phi_sq_avg(2:end);     % n x 1

    % --- 3g. Report ------------------------------------------------------
    fprintf('\nStatistics at component j = 1:\n');
    fprintf('   Mean     = %.7e\n', mean_u(1));
    fprintf('   Variance = %.7e\n', var_u(1));
    fprintf('   Reference (Sobol N=2e5): mean %.6e, var %.6e\n', ...
        ref_mean_u1, ref_var_u1);
    fprintf('   Rel. error mean : %.3f%%\n', ...
        100*abs(mean_u(1)-ref_mean_u1)/abs(ref_mean_u1));
    fprintf('   Rel. error var  : %.3f%%\n', ...
        100*abs(var_u(1)-ref_var_u1)/abs(ref_var_u1));

    fprintf('\nField statistics:\n');
    fprintf('   Mean field     : min = %.4e, max = %.4e\n', ...
        min(mean_u), max(mean_u));
    fprintf('   Variance field : min = %.4e, max = %.4e\n', ...
        min(var_u),  max(var_u));

    % Save for final summary
    results(ip).p       = p;
    results(ip).Nq      = Nq;
    results(ip).Pp1     = Pp1;
    results(ip).mean_u1 = mean_u(1);
    results(ip).var_u1  = var_u(1);
    results(ip).mean_u  = mean_u;
    results(ip).var_u   = var_u;

    % --- 3h. Visualisations ----------------------------------------------
    fprintf('\nSaving figures to %s/ ...\n', fig_dir);
    make_and_save(mean_u, 'mean',    '3D',      p, fig_dir);
    make_and_save(mean_u, 'mean',    'contour', p, fig_dir);
    make_and_save(var_u,  'var',     '3D',      p, fig_dir);
    make_and_save(var_u,  'var',     'contour', p, fig_dir);
end

%% --------------------------  4.  Summary table  ------------------------
fprintf('\n=========================================================\n');
fprintf('   Summary: u_1 statistics across methods\n');
fprintf('   (reference values = smallest cost at which each method\n');
fprintf('    converged to 3 sig. fig. in Assignments 1-2)\n');
fprintf('=========================================================\n');
fprintf('%-28s %8s %16s %16s\n', 'Method', 'Cost', 'Mean', 'Variance');
fprintf('%-28s %8d %16.6e %16.6e\n', ...
    sprintf('PC  p=1 (P+1=%d)', results(1).Pp1), ...
    results(1).Nq, results(1).mean_u1, results(1).var_u1);
fprintf('%-28s %8d %16.6e %16.6e\n', ...
    sprintf('PC  p=2 (P+1=%d)', results(2).Pp1), ...
    results(2).Nq, results(2).mean_u1, results(2).var_u1);
fprintf('%-28s %8d %16.6e %16.6e\n', 'Sparse CC (level 3)', 69, ...
    6.361e-2, 7.221e-5);
fprintf('%-28s %8d %16.6e %16.6e\n', 'Sobol QMC (N=5,000)',  5000, ...
    6.361e-2, 7.224e-5);
fprintf('%-28s %8d %16.6e %16.6e\n', 'LHS (N=5,000)',        5000, ...
    6.361e-2, 7.220e-5);
fprintf('%-28s %8d %16.6e %16.6e\n', 'Monte Carlo (N=50,000)', 50000, ...
    6.366e-2, 7.222e-5);
fprintf('\nDone.\n');

%==========================================================================
%                               HELPERS
%==========================================================================

function alpha = build_total_order_multiindex(d, p)
% All multi-indices alpha in N^d with |alpha|_1 <= p, rows = (P+1) x d.
    alpha = zeros(0, d);
    for deg = 0:p
        alpha = [alpha; partitions_exact(d, deg)]; %#ok<AGROW>
    end
end

function A = partitions_exact(d, deg)
% Non-negative integer d-tuples summing exactly to deg.
    if d == 1
        A = deg;
        return;
    end
    A = zeros(0, d);
    for k = 0:deg
        sub = partitions_exact(d-1, deg-k);
        A   = [A; k*ones(size(sub,1),1), sub]; %#ok<AGROW>
    end
end

function [nodes, weights] = gauss_legendre_prob(n)
% Golub-Welsch: Gauss-Legendre on [-1,1] with weights summing to 1
% (i.e. including the 1/2 uniform-density factor).
    k     = 1:(n-1);
    beta  = k ./ sqrt(4*k.^2 - 1);
    T     = diag(beta, 1) + diag(beta, -1);
    [V,D] = eig(T);
    [nodes, idx] = sort(diag(D));
    V       = V(:, idx);
    weights = V(1, :).'.^2;        % probabilistic weights (sum to 1)
end

function Psi = eval_legendre_basis(xi, alpha)
% Evaluate the tensor-product Legendre basis at a set of nodes.
%   xi     : Nq x d  (quadrature nodes in [-1,1]^d)
%   alpha  : Nb x d  (multi-indices)
% Returns  Psi : Nq x Nb  with Psi(q,k) = prod_i L_{alpha(k,i)}(xi(q,i)).

    [Nq, d] = size(xi);
    Nb      = size(alpha, 1);
    maxdeg  = max(alpha(:));

    % 1-D Legendre values L_j(xi_{q,i}) via Bonnet's recursion:
    %   (j+1) L_{j+1}(x) = (2j+1) x L_j(x) - j L_{j-1}(x)
    L = zeros(Nq, d, maxdeg + 1);
    L(:, :, 1) = 1;                           % L_0
    if maxdeg >= 1
        L(:, :, 2) = xi;                      % L_1
    end
    for j = 1:(maxdeg - 1)
        L(:, :, j+2) = ((2*j+1) * xi .* L(:, :, j+1) ...
                       - j * L(:, :, j)) / (j + 1);
    end

    Psi = ones(Nq, Nb);
    for k = 1:Nb
        for i = 1:d
            Psi(:, k) = Psi(:, k) .* L(:, i, alpha(k,i) + 1);
        end
    end
end

function make_and_save(sol, kind, style, p, fig_dir)
% Wrap a PlotP3_3D / PlotP3_contour call, set a title, save as png.
% For contour plots of small-magnitude fields (variance), route to
% PlotP3_contour_var which builds the level vector from the actual data
% range; the stock PlotP3_contour hardcodes a step of 0.005 that misses
% the variance data entirely.
    h = figure('Position', [100 100 600 450], 'Color', 'w');
    if strcmp(style, '3D')
        PlotP3_3D(sol);
    elseif strcmp(kind, 'var')
        PlotP3_contour_var(sol);
    else
        PlotP3_contour(sol);
    end
    if strcmp(kind, 'mean')
        ttl = sprintf('\\mu_u  (PC, p = %d)', p);
    else
        ttl = sprintf('\\sigma^2_u  (PC, p = %d)', p);
    end
    title(ttl, 'Interpreter', 'tex');
    fname = sprintf('PC_p%d_%s_%s.png', p, kind, style);
    saveas(h, fullfile(fig_dir, fname));
end