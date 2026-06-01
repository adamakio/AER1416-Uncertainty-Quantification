import os
import numpy as np
import matplotlib.pyplot as plt
from p1_model import build_coeffs_p1, eval_f_p1, analytic_mean_var_p1

np.random.seed(42)

Z_99  = 2.58
MU_G  = 1.0   # analytical mean of the linear control variate g(x)


def draw_gaussian_samples(N: int, d: int) -> np.ndarray:
    """Draw N i.i.d. samples from N(0, I_d), returning shape (N, d)."""
    return np.random.randn(N, d)


def eval_control_variate(X: np.ndarray, a0: float, a_vec: np.ndarray) -> np.ndarray:
    """
    Evaluate the linear control variate g(x) = a0 + sum_i a_i * x_i.
    E[g] = 1 since each x_i has zero mean.
    Returns array of length N.
    """
    return a0 + X @ a_vec


def optimal_cv_coefficient(f_vals: np.ndarray, g_vals: np.ndarray) -> float:
    """
    Compute lambda* = Cov(f, g) / Var(g) using sample estimates.
    This minimises the variance of the CV estimator f - lambda*(g - mu_g).
    """
    mu_f  = np.mean(f_vals)
    mu_g  = np.mean(g_vals)
    cov   = np.mean((f_vals - mu_f) * (g_vals - mu_g))
    var_g = np.mean((g_vals - mu_g) ** 2)
    return cov / var_g if var_g > 1e-14 else 0.0


def compute_statistics(values: np.ndarray) -> dict:
    """
    Compute mean, variance, standard errors and 99% CI half-widths
    for an array of scalar sample values.
    """
    N      = len(values)
    mu     = np.mean(values)
    sigma2 = np.var(values, ddof=1)
    se_mu  = np.sqrt(sigma2 / N)
    se_var = np.sqrt(2.0 / N) * sigma2
    return dict(mean=mu, var=sigma2,
                se_mean=se_mu,  ci_mean=Z_99 * se_mu,
                se_var=se_var,  ci_var=Z_99 * se_var)


def run_cv_experiment(N: int, d: int, gamma: float,
                      a0: float, a_vec: np.ndarray, A: np.ndarray) -> dict:
    """
    Single MC run of size N: evaluates standard and CV estimators,
    returning statistics for both along with lambda* and the variance ratio.
    """
    X       = draw_gaussian_samples(N, d)
    f_vals  = eval_f_p1(X, gamma, a0, a_vec, A)
    g_vals  = eval_control_variate(X, a0, a_vec)

    lam     = optimal_cv_coefficient(f_vals, g_vals)
    cv_vals = f_vals - lam * (g_vals - MU_G)

    stats_mc = compute_statistics(f_vals)
    stats_cv = compute_statistics(cv_vals)
    var_ratio = stats_cv['var'] / stats_mc['var'] \
                if stats_mc['var'] > 1e-14 else 1.0

    return dict(N=N, lam_star=lam,
                mc=stats_mc, cv=stats_cv,
                var_ratio=var_ratio)


def repeated_mean_estimates(N: int, d: int, gamma: float,
                             a0: float, a_vec: np.ndarray, A: np.ndarray,
                             n_repeats: int = 200) -> tuple:
    """
    Run n_repeats independent MC experiments of size N.
    Collect the mean estimate from each run for standard and CV MC —
    used to visualise how much the CV tightens the estimator distribution.
    """
    mc_means = np.zeros(n_repeats)
    cv_means = np.zeros(n_repeats)
    for r in range(n_repeats):
        result       = run_cv_experiment(N, d, gamma, a0, a_vec, A)
        mc_means[r]  = result['mc']['mean']
        cv_means[r]  = result['cv']['mean']
    return mc_means, cv_means


def print_convergence_table(d: int, gamma: float,
                             a0: float, a_vec: np.ndarray, A: np.ndarray,
                             sample_sizes: list) -> None:
    """Print mean/variance estimates with 99% CIs for standard and CV MC."""
    header = (f"{'N':>8}  {'E[f] MC':>10} {'±CI':>8}  "
              f"{'Var MC':>10} {'±CI':>8}  "
              f"{'E[f] CV':>10} {'±CI':>8}  "
              f"{'Var CV':>10} {'±CI':>8}  "
              f"{'VarRatio':>9}")
    print(header)
    print("-" * 108)
    for N in sample_sizes:
        r  = run_cv_experiment(N, d, gamma, a0, a_vec, A)
        mc = r['mc']
        cv = r['cv']
        print(f"{N:>8d}  "
              f"{mc['mean']:>10.5f} ±{mc['ci_mean']:>7.5f}  "
              f"{mc['var']:>10.4e} ±{mc['ci_var']:>8.4e}  "
              f"{cv['mean']:>10.5f} ±{cv['ci_mean']:>7.5f}  "
              f"{cv['var']:>10.4e} ±{cv['ci_var']:>8.4e}  "
              f"{r['var_ratio']:>9.4f}")


def plot_mean_distributions(mc_means: np.ndarray, cv_means: np.ndarray,
                             N: int, gamma: float) -> None:
    """
    Overlay histograms of mean estimates from repeated runs to show
    how the CV reduces the spread of the estimator distribution.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(mc_means, bins=30, alpha=0.5, density=True,
             color='steelblue', label='Standard MC')
    plt.hist(cv_means, bins=30, alpha=0.5, density=True,
             color='seagreen', label='Control Variate MC')
    plt.axvline(np.mean(mc_means), color='steelblue', linestyle='--',
                label=f'MC mean = {np.mean(mc_means):.4f}')
    plt.axvline(np.mean(cv_means), color='seagreen', linestyle='--',
                label=f'CV mean = {np.mean(cv_means):.4f}')
    plt.title(f'Mean estimate distributions over 200 runs '
              f'(N={N}, $\\gamma$={gamma})')
    plt.xlabel('Estimated mean of $f$')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    out_dir = "q5_outputs"
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f'q5_histogram_gamma{gamma}.png')
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"  Histogram saved as {fname}")


def run_q5():
    d            = 10
    sample_sizes = [1000, 5000, 10000, 50000, 200000]
    N_hist       = 1000
    n_repeats    = 200

    for gamma in [0.5, 0.1]:
        a0, a_vec, A          = build_coeffs_p1(d)
        exact_mean, exact_var = analytic_mean_var_p1(d, gamma)

        print(f"\n{'='*108}")
        print(f"Q5: Control Variate MC  |  d={d},  gamma={gamma}  |  "
              f"Analytical mean={exact_mean:.6f},  var={exact_var:.6f}")
        print(f"{'='*108}")

        print_convergence_table(d, gamma, a0, a_vec, A, sample_sizes)

        # Lambda* and variance ratio at largest N
        result_large = run_cv_experiment(sample_sizes[-1], d, gamma,
                                         a0, a_vec, A)
        print(f"\n  lambda* at N={sample_sizes[-1]:,}: "
              f"{result_large['lam_star']:.6f}")
        print(f"  Var(CV) / Var(MC) at N={sample_sizes[-1]:,}: "
              f"{result_large['var_ratio']:.6f}")

        # Histogram of repeated mean estimates
        print(f"\n  Running {n_repeats} repeated experiments at "
              f"N={N_hist} to visualise variance reduction...")
        mc_means, cv_means = repeated_mean_estimates(
            N_hist, d, gamma, a0, a_vec, A, n_repeats)
        plot_mean_distributions(mc_means, cv_means, N_hist, gamma)

    print("\nFinished Q5.")


if __name__ == "__main__":
    run_q5()