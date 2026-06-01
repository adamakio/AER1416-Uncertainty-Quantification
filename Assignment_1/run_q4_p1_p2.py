import numpy as np
from scipy.integrate import solve_ivp

# ===========================================================================
# Q4: First-Order (Linear) Approximation for P1 and P2
# ===========================================================================

# ---------------------------------------------------------------------------
# P1: Quadratic function f(x) with d=10, gamma=0.5, x_i ~ N(0,1)
# ---------------------------------------------------------------------------

# Compute the harmonic sum needed for the analytical mean
d = 10
harmonic_sum = sum(1.0 / i for i in range(1, d + 1))
gamma = 0.5

# Build coefficient matrices
a0 = 1.0
a_lin = np.array([1.0 / np.sqrt(i) for i in range(1, d + 1)])   # linear coefficients

def build_A_matrix(dim):
    """Build the symmetric coefficient matrix A with entries a_ij."""
    A = np.zeros((dim, dim))
    for i in range(1, dim + 1):
        for j in range(1, dim + 1):
            if i == j:
                A[i-1, j-1] = 1.0 / i
            else:
                A[i-1, j-1] = min(i, j) / (max(i, j)**2 * (i + j))
    return A

A_mat = build_A_matrix(d)

# --- Analytical mean and variance (Gaussian case) ---
# E[f] = a0 + gamma * sum(a_ii)  (off-diagonal E[xi*xj]=0, diagonal E[xi^2]=1)
analytical_mean = a0 + gamma * np.sum(np.diag(A_mat))

# Var[f] = sum(a_i^2) + 2*gamma^2 * sum_ij(a_ij^2)
analytical_var  = np.sum(a_lin**2) + 2 * gamma**2 * np.sum(A_mat**2)

print("=" * 55)
print("P1: First-Order Approximation  (d=10, gamma=0.5)")
print("=" * 55)
print(f"  Analytical mean     : {analytical_mean:.6f}")
print(f"  Analytical variance : {analytical_var:.6f}")

# --- First-order Taylor expansion about z=0 (the mean of x) ---
# f(x) ~ f(0) + sum_i (df/dx_i)|_0 * x_i
# f(0) = a0 = 1,   df/dx_i|_0 = a_i  (linear term only; quadratic vanishes at x=0)
# => f_tilde(x) = 1 + sum_i a_i * x_i

# For a linear function of independent zero-mean unit-variance Gaussians:
#   E[f_tilde] = f(0) = 1
#   Var[f_tilde] = sum_i a_i^2 * Var(x_i) = sum_i (1/i)
linear_mean_p1 = a0                          # = 1
linear_var_p1  = np.sum(a_lin**2)            # = sum(1/i, i=1..10)

print(f"\n  Linear approx mean  : {linear_mean_p1:.6f}")
print(f"  Linear approx var   : {linear_var_p1:.6f}")
print(f"\n  Error in mean       : {abs(linear_mean_p1 - analytical_mean):.6f}")
print(f"  Error in variance   : {abs(linear_var_p1  - analytical_var ):.6f}")


# ===========================================================================
# P2: Mass-spring-damper ODE
#   m*u'' + c*u' + k*u = F0*cos(w*t),   u(0)=u'(0)=0
# Nominal (mean) parameters:
#   m ~ U[0.9,1.1]  => mean=1.0,  Var=0.2^2/12
#   c ~ U[0.8,1.2]  => mean=1.0,  Var=0.4^2/12
#   k ~ U[10,12]    => mean=11.0, Var=2.0^2/12
#   w ~ U[0.1,0.2]  => mean=0.15, Var=0.1^2/12
#   F0~ U[0.9,1.1]  => mean=1.0,  Var=0.2^2/12
# ===========================================================================

# Nominal parameter values (means)
m_nom  = 1.0
c_nom  = 1.0
k_nom  = 11.0
w_nom  = 0.15
F0_nom = 1.0
T      = 10.0   # final time

# Uniform variances: Var(U[a,b]) = (b-a)^2 / 12
param_variances = {
    'm' : (1.1  - 0.9)**2 / 12.0,
    'c' : (1.2  - 0.8)**2 / 12.0,
    'k' : (12.0 - 10.0)**2 / 12.0,
    'w' : (0.2  - 0.1)**2 / 12.0,
    'F0': (1.1  - 0.9)**2 / 12.0,
}

def msd_rhs(t, state, m, c, k, F0, w):
    """Right-hand side of the mass-spring-damper system written as first-order form."""
    displacement, velocity = state
    accel = (F0 * np.cos(w * t) - c * velocity - k * displacement) / m
    return [velocity, accel]

def displacement_at_T(m, c, k, F0, w, t_end=10.0):
    """Integrate the MSD system and return displacement u(t_end)."""
    result = solve_ivp(
        msd_rhs,
        [0.0, t_end],
        [0.0, 0.0],
        args=(m, c, k, F0, w),
        t_eval=[t_end],
        rtol=1e-8, atol=1e-10
    )
    return result.y[0, -1]

def compute_gradients(m0, c0, k0, F0_0, w0, step=1e-4, t_end=10.0):
    """
    Forward-difference approximation of partial derivatives of u(T)
    with respect to each parameter at the nominal point.
    Returns (du/dm, du/dc, du/dk, du/dF0, du/dw).
    """
    u0 = displacement_at_T(m0, c0, k0, F0_0, w0, t_end)

    grad_m  = (displacement_at_T(m0+step, c0,      k0,      F0_0,      w0,      t_end) - u0) / step
    grad_c  = (displacement_at_T(m0,      c0+step, k0,      F0_0,      w0,      t_end) - u0) / step
    grad_k  = (displacement_at_T(m0,      c0,      k0+step, F0_0,      w0,      t_end) - u0) / step
    grad_F0 = (displacement_at_T(m0,      c0,      k0,      F0_0+step, w0,      t_end) - u0) / step
    grad_w  = (displacement_at_T(m0,      c0,      k0,      F0_0,      w0+step, t_end) - u0) / step

    return u0, grad_m, grad_c, grad_k, grad_F0, grad_w

u_nominal, g_m, g_c, g_k, g_F0, g_w = compute_gradients(
    m_nom, c_nom, k_nom, F0_nom, w_nom
)

# Linear approximation:
#   E[u(T)] = u_nominal  (each parameter has zero-mean deviation)
#   Var[u(T)] = sum_alpha  (du/d_alpha)^2 * Var(alpha)
linear_mean_p2 = u_nominal
linear_var_p2  = (
    g_m**2  * param_variances['m']  +
    g_c**2  * param_variances['c']  +
    g_k**2  * param_variances['k']  +
    g_F0**2 * param_variances['F0'] +
    g_w**2  * param_variances['w']
)

print("\n" + "=" * 55)
print("P2: First-Order Approximation  (t = 10 s)")
print("=" * 55)
print(f"  Nominal displacement u(10)  : {u_nominal:.6f}")
print(f"  du/dm  = {g_m:+.6f}   Var(m)  = {param_variances['m']:.6f}")
print(f"  du/dc  = {g_c:+.6f}   Var(c)  = {param_variances['c']:.6f}")
print(f"  du/dk  = {g_k:+.6f}   Var(k)  = {param_variances['k']:.6f}")
print(f"  du/dF0 = {g_F0:+.6f}   Var(F0) = {param_variances['F0']:.6f}")
print(f"  du/dw  = {g_w:+.6f}   Var(w)  = {param_variances['w']:.6f}")
print(f"\n  Linear approx mean  : {linear_mean_p2:.6f}")
print(f"  Linear approx var   : {linear_var_p2:.6f}")