function run_Q3()
% run_Q3:
%   Monte Carlo estimation of the mean and variance of the solution u(theta)
%   to the random linear system arising from a stochastic diffusion equation:
%
%       A(theta) * u(theta) = b
%
%   where  A(theta) = A0 + sum_{i=1}^{d} theta_i * A_i,
%   and each theta_i ~ U[-3, +3] independently (d = 3).
%
%   Sweeps over multiple sample sizes N to assess convergence.
%   Reports 99% confidence intervals for u_1 (the first solution component).
%   Finishes with a large-N run to visualize the full mean and variance fields.

    % -------------------------------------------------------
    % Load problem data: A0, A1, A2, A3 (each 143x143) and b (143x1)
    % -------------------------------------------------------
    load('P3.mat', 'A0', 'A1', 'A2', 'A3', 'b');

    n = size(A0, 1);   % spatial degrees of freedom (n = 143)
    d = 3;             % number of independent random inputs

    % -------------------------------------------------------
    % Settings
    % -------------------------------------------------------
    sample_sizes       = [1000, 5000, 10000, 50000, 200000];
    z_99               = 2.58;   % z-score for 99% confidence intervals
    report_component   = 1;      % index of u to display in the table (u_1)

    fprintf('Q3: Stochastic system  A(theta)*u(theta) = b  with d = %d.\n', d);
    fprintf('    theta_i ~ U[-3, +3],  i = 1, ..., %d.\n', d);
    fprintf('    Approximating E[u] and Var[u] via Monte Carlo.\n\n');

    % -------------------------------------------------------
    % Table header
    % -------------------------------------------------------
    fprintf('%s\n', repmat('-', 1, 80));
    fprintf('  N        E[u_1]       SE(mean)   ±99%% CI     Var[u_1]     SE(var)    ±99%% CI    Time\n');
    fprintf('%s\n', repmat('-', 1, 80));

    % -------------------------------------------------------
    % Sweep over sample sizes
    % -------------------------------------------------------
    for N = sample_sizes

        t_start = tic;

        % Draw N samples: theta is d x N, each column = [theta_1; ...; theta_d]
        % theta_i ~ U[-3, +3]  =>  theta_i = 6*rand - 3
        theta = 6*rand(d, N) - 3;

        % Solve A(theta^(k)) * u^(k) = b for each sample k = 1, ..., N
        u = zeros(n, N);
        for k = 1:N
            A_theta = A0 + theta(1,k)*A1 + theta(2,k)*A2 + theta(3,k)*A3;
            u(:, k) = A_theta \ b;
        end

        % Estimate mean and variance across samples (per component)
        E_u   = mean(u, 2);          % n x 1  sample mean
        Var_u = var(u, 0, 2);        % n x 1  unbiased sample variance

        % Extract statistics for the reported component (u_1)
        E_u1   = E_u(report_component);
        Var_u1 = Var_u(report_component);

        % Standard errors and 99% confidence interval half-widths
        SE_mean = sqrt(Var_u1 / N);           % SE of sample mean
        SE_var  = sqrt(2 / N) * Var_u1;       % SE of sample variance (normal approx)

        elapsed = toc(t_start);

        fprintf('%6d   %11.5e  %9.3e ±%9.3e  %11.5e  %9.3e ±%9.3e  %.2fs\n', ...
            N, E_u1, SE_mean, z_99*SE_mean, Var_u1, SE_var, z_99*SE_var, elapsed);
    end

    % -------------------------------------------------------
    % Large run for full-field mean and variance visualization
    % -------------------------------------------------------
    N_large = 200000;
    fprintf('\nLarge run (N = %d) for full mean and variance field plots...\n', N_large);

    theta_large = 6*rand(d, N_large) - 3;
    u_large     = zeros(n, N_large);

    t_large = tic;
    for k = 1:N_large
        A_theta = A0 + theta_large(1,k)*A1 + theta_large(2,k)*A2 + theta_large(3,k)*A3;
        u_large(:, k) = A_theta \ b;
    end
    fprintf('   (Elapsed: %.2f s)\n', toc(t_large));

    E_u_field   = mean(u_large, 2);
    Var_u_field = var(u_large, 0, 2);

    % -------------------------------------------------------
    % Visualize mean field
    % -------------------------------------------------------
    fprintf('Plotting mean field...\n');
    PlotP3_3D(E_u_field);
    title('E[u(\theta)] — mean solution (3D)');
    set(gcf, 'Color', 'w'); set(gca, 'Color', 'w');
    saveas(gcf, 'q3_outputs/q3_mean_3d.png')

    figure; PlotP3_contour(E_u_field);
    title('E[u(\theta)] — mean solution (contour)');
    set(gcf, 'Color', 'w'); set(gca, 'Color', 'w');
    saveas(gcf, 'q3_outputs/q3_mean_contour.png')

    % -------------------------------------------------------
    % Visualize variance field
    % -------------------------------------------------------
    fprintf('Plotting variance field...\n');
    figure; PlotP3_3D(Var_u_field);
    title('Var[u(\theta)] — variance field (3D)');
    set(gcf, 'Color', 'w'); set(gca, 'Color', 'w');
    saveas(gcf, 'q3_outputs/q3_var_3d.png')

    figure;
    PlotP3_contour_var(Var_u_field);
    title('Var[u(\theta)] — variance field (contour)');
    set(gcf, 'Color', 'w'); set(gca, 'Color', 'w');
    saveas(gcf, 'q3_outputs/q3_var_contour.png')

end