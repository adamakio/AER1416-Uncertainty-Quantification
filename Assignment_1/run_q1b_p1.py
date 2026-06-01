# run_q1b_p1.py
from __future__ import annotations
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from p1_model import build_coeffs_p1, eval_f_p1, analytic_mean_var_p1
from mc_stats import mean_ci_clt_99, var_ci_sample_variance_99, cdf_with_clt_ci

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def main() -> None:
    d = 10
    gamma = 0.5
    seed = 12345

    # Convergence sizes (edit if desired)
    N_list = [5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]

    out_dir = "q1b_outputs"
    ensure_dir(out_dir)

    a0, a_vec, A = build_coeffs_p1(d)
    mu_true, var_true = analytic_mean_var_p1(d, gamma)

    print(f"Analytic mean  = {mu_true:.12f}")
    print(f"Analytic var   = {var_true:.12f}")

    rng = np.random.default_rng(seed)

    rows = []
    all_values_for_cdf = None

    for N in N_list:
        X = rng.standard_normal(size=(N, d))
        fvals = eval_f_p1(X, gamma, a0, a_vec, A)

        mean_ci = mean_ci_clt_99(fvals)
        var_ci = var_ci_sample_variance_99(fvals)

        rows.append({
            "N": N,

            "mean_hat": mean_ci.est,
            "mean_se": mean_ci.se,
            "mean_ci_low": mean_ci.ci_low,
            "mean_ci_high": mean_ci.ci_high,
            "mean_ci_halfwidth": 0.5 * (mean_ci.ci_high - mean_ci.ci_low),
            "mean_abs_err": abs(mean_ci.est - mu_true),

            "var_hat": var_ci.est,
            "var_se": var_ci.se,
            "var_ci_low": var_ci.ci_low,
            "var_ci_high": var_ci.ci_high,
            "var_ci_halfwidth": 0.5 * (var_ci.ci_high - var_ci.ci_low),
            "var_abs_err": abs(var_ci.est - var_true),
        })

        print(
            f"N={N:>8}  mean={mean_ci.est:.6f}  var={var_ci.est:.6f}  "
            f"mean99=[{mean_ci.ci_low:.6f},{mean_ci.ci_high:.6f}]  "
            f"var99=[{var_ci.ci_low:.6f},{var_ci.ci_high:.6f}]"
        )
        all_values_for_cdf = fvals

    # Save convergence CSV
    csv_path = os.path.join(out_dir, "p1_q1b_convergence.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Minimal plots (you can choose what to include in report)
    N_arr = np.array([r["N"] for r in rows], dtype=float)

    # CDF using largest N
    assert all_values_for_cdf is not None
    vals = all_values_for_cdf
    z_lo = np.quantile(vals, 0.0005)
    z_hi = np.quantile(vals, 0.9995)
    z_grid = np.linspace(z_lo, z_hi, 400)
    cdf_pts = cdf_with_clt_ci(vals, z_grid)

    # plot without CI (CI too tight to see at N=1e6)
    plt.figure()
    z = np.array([p[0] for p in cdf_pts])
    F = np.array([p[1] for p in cdf_pts])
    plt.plot(z, F, linewidth=2, label="Empirical CDF")
    plt.xlabel("z")
    plt.ylabel("P(f ≤ z)")
    plt.title(f"P1: CDF estimate (N={len(vals)})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cdf.png"), dpi=200)

    print("\nSaved outputs to:", out_dir)
    print(" - p1_q1b_convergence.csv")
    print(" - p1_q1b_cdf.csv")
    print(" - cdf.png")

if __name__ == "__main__":
    main()