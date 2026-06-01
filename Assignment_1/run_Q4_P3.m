function run_Q4_P3()
% run_Q4_P3:
%   First-order Taylor approximation of the solution u(theta) to the
%   random linear system  A(theta)*u(theta) = b, where
%       A(theta) = A0 + theta_1*A1 + theta_2*A2 + theta_3*A3,
%       theta_i ~ U[-3, +3]  (d = 3 independent inputs).
%
%   Expansion point: theta = 0.
%   Nominal solution:  u0 = A0^{-1} * b.
%   Sensitivity vectors: beta_i = -A0^{-1} * Ai * u0,  i = 1,2,3.
%
%   Linear approximation:
%       E[u_tilde]   = u0
%       Var[u_tilde] = Var(theta_i) * sum_{i=1}^{3} beta_i * beta_i^T
%                    = 3 * sum_{i=1}^{3} beta_i * beta_i^T
%   since Var(U[-3,+3]) = (6^2)/12 = 3.

    % -----------------------------------------------------------
    % Load system matrices and right-hand side
    % -----------------------------------------------------------
    load('P3.mat', 'A0', 'A1', 'A2', 'A3', 'b');
    sys_dim = size(A0, 1);   % n = 143

    % -----------------------------------------------------------
    % Compute nominal solution u0 = A0^{-1} * b
    % -----------------------------------------------------------
    A0_inv = inv(A0);
    u0 = A0_inv * b;         % 143 x 1

    % -----------------------------------------------------------
    % Compute sensitivity vectors beta_i = -A0^{-1} * Ai * u0
    % These are the partial derivatives du/d(theta_i) at theta=0
    % -----------------------------------------------------------
    beta_1 = -A0_inv * A1 * u0;   % 143 x 1
    beta_2 = -A0_inv * A2 * u0;   % 143 x 1
    beta_3 = -A0_inv * A3 * u0;   % 143 x 1

    % -----------------------------------------------------------
    % Variance of theta_i ~ U[-3, +3]:
    %   Var(theta_i) = (b - a)^2 / 12 = 6^2 / 12 = 3
    % -----------------------------------------------------------
    theta_var = (6^2) / 12;   % = 3

    % Componentwise variance: diag of the 143x143 variance matrix
    %   Var[u_tilde] = theta_var * (beta_1*beta_1' + beta_2*beta_2' + beta_3*beta_3')
    var_matrix = theta_var * (beta_1*beta_1' + beta_2*beta_2' + beta_3*beta_3');
    var_components = diag(var_matrix);   % 143 x 1

    % -----------------------------------------------------------
    % Report results for the first component
    % -----------------------------------------------------------
    fprintf('==========================================\n');
    fprintf('Q4 P3: First-Order Linear Approximation\n');
    fprintf('==========================================\n');
    fprintf('System dimension n = %d\n\n', sys_dim);

    fprintf('Nominal solution u0 = A0^{-1}*b\n');
    fprintf('  First 5 components of u0:\n');
    fprintf('    u0(%d) = %.8e\n', [1:5; u0(1:5)']);

    fprintf('\nSensitivity vectors beta_i = -A0^{-1}*Ai*u0\n');
    fprintf('  beta_1(1) = %.6e\n', beta_1(1));
    fprintf('  beta_2(1) = %.6e\n', beta_2(1));
    fprintf('  beta_3(1) = %.6e\n', beta_3(1));

    fprintf('\nVar(theta_i) = %.4f  [U(-3,+3)]\n', theta_var);

    fprintf('\nLinear approximation for component 1:\n');
    fprintf('  E[u_tilde(1)]   = u0(1)        = %.8e\n', u0(1));
    fprintf('  Var[u_tilde(1)]               = %.8e\n', var_components(1));

    fprintf('\nFirst 5 component variances:\n');
    fprintf('  Var[u_tilde(%d)] = %.6e\n', [1:5; var_components(1:5)']);
end