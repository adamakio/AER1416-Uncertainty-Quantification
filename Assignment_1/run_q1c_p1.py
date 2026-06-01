# run_q1c_p1.py
from __future__ import annotations
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from p1_model import build_coeffs_p1, eval_f_p1, analytic_mean_var_p1

Z_99 = 2.5758293035489004  # Phi^{-1}(0.995)

def tcrit_99(df: int) -> float:
    """t_{0.995,df} with SciPy if available; else normal approx."""
    if df <= 0:
        return Z_99
    try:
        from scipy.stats import t  # type: ignore
        return float(t.ppf(0.995, df))
    except Exception:
        return Z_99

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

import math

def normality_metrics(z: np.ndarray) -> dict:
    """Return mean, std, skewness, excess kurtosis of z."""
    z = np.asarray(z, dtype=float).ravel()
    m = float(z.mean())
    s = float(z.std(ddof=1))
    if s == 0 or not np.isfinite(s):
        return {"mean": m, "std": s, "skew": float("nan"), "exkurt": float("nan")}
    x = (z - m) / s
    skew = float(np.mean(x**3))
    exkurt = float(np.mean(x**4) - 3.0)
    return {"mean": m, "std": s, "skew": skew, "exkurt": exkurt}


def simulate_batch_variances(d: int, gamma: float, m: int, B: int, seed: int):
    """Simulate B batches of size m; return var_b, zscores, v_bar, se_vbar, CI."""
    a0, a_vec, A = build_coeffs_p1(d)
    rng = np.random.default_rng(seed)

    var_b = np.empty(B, dtype=float)
    for b in range(B):
        X = rng.standard_normal(size=(m, d))
        fvals = eval_f_p1(X, gamma, a0, a_vec, A)
        mu_hat = float(fvals.mean())
        m2_hat = float(np.mean(fvals**2))
        var_b[b] = m2_hat - mu_hat**2

    v_bar = float(var_b.mean())
    s_v = float(var_b.std(ddof=1))
    se_vbar = s_v / math.sqrt(B)
    tcrit = tcrit_99(B - 1)
    ci_low = v_bar - tcrit * se_vbar
    ci_high = v_bar + tcrit * se_vbar

    zscores = (var_b - v_bar) / s_v
    return var_b, zscores, v_bar, se_vbar, (ci_low, ci_high)


def tune_m_B(
    d: int,
    gamma: float,
    candidates: list[tuple[int, int]],
    seed: int,
    Nmax: int = 4_000_000,
    targets: dict | None = None,
):
    """
    Evaluate candidate (m,B) pairs and pick one.
    targets: dict with thresholds, e.g.
      {"abs_skew":0.15, "abs_exkurt":0.30, "mean_tol":0.05, "std_tol":0.05}
    """
    if targets is None:
        targets = {"abs_skew": 0.15, "abs_exkurt": 0.30, "mean_tol": 0.05, "std_tol": 0.05}

    results = []
    for m, B in candidates:
        Ntot = m * B
        if Ntot > Nmax:
            continue

        _, z, v_bar, se, (lo, hi) = simulate_batch_variances(d, gamma, m, B, seed)
        met = normality_metrics(z)

        # "normality score": smaller is better
        score = (
            abs(met["mean"]) / targets["mean_tol"]
            + abs(met["std"] - 1.0) / targets["std_tol"]
            + abs(met["skew"]) / targets["abs_skew"]
            + abs(met["exkurt"]) / targets["abs_exkurt"]
        )

        results.append({
            "m": m, "B": B, "N_total": Ntot,
            "z_mean": met["mean"], "z_std": met["std"],
            "z_skew": met["skew"], "z_exkurt": met["exkurt"],
            "v_bar": v_bar, "se_vbar": se, "ci_low": lo, "ci_high": hi,
            "ci_halfwidth": 0.5*(hi-lo),
            "score": score,
        })

    # pick: first those meeting thresholds, lowest N_total; else lowest score
    def meets(r):
        return (abs(r["z_mean"]) <= targets["mean_tol"]
                and abs(r["z_std"] - 1.0) <= targets["std_tol"]
                and abs(r["z_skew"]) <= targets["abs_skew"]
                and abs(r["z_exkurt"]) <= targets["abs_exkurt"])

    feasible = [r for r in results if meets(r)]
    if feasible:
        best = min(feasible, key=lambda r: r["N_total"])
    else:
        best = min(results, key=lambda r: r["score"])

    return best, results

def main() -> None:
    d = 10
    gamma = 0.5
    seed = 2026

    # Batch MC settings (N_total = B * m)
        # --- Tuning grid (keep B reasonably large; increase m to improve CLT) ---
    candidates = [
        (5000, 200),    # 1e6 (your current)
        (10000, 200),   # 2e6 (usually noticeably better)
        (20000, 200),   # 4e6 (often much better)
        (10000, 300),   # 3e6
        (20000, 150),   # 3e6
        (20000, 100),   # 2e6 but fewer batches
    ]

    best, allres = tune_m_B(d, gamma, candidates, seed=seed, Nmax=4_000_000)
    m = best["m"]
    B = best["B"]

    print("Chosen (m,B) after tuning:")
    print(best)

    # Now run the final simulation using the selected (m,B)
    var_b, zscores, v_bar, se_vbar, (ci_low, ci_high) = simulate_batch_variances(d, gamma, m, B, seed+1)
    out_dir = "q1c_outputs"
    ensure_dir(out_dir)

    a0, a_vec, A = build_coeffs_p1(d)
    mu_true, var_true = analytic_mean_var_p1(d, gamma)

    print(f"Analytic var = {var_true:.12f}")
    print(f"Batch MC: B={B}, m={m}, N_total={B*m}")

    rng = np.random.default_rng(seed)

    mu_b = np.zeros(B)
    m2_b = np.zeros(B)
    var_b = np.zeros(B)

    # Each batch produces a variance estimate as a function of sample means:
    #   v_hat = mean(f^2) - mean(f)^2
    # This is CLT-friendly: it's a smooth function of sample means.
    for b in range(B):
        X = rng.standard_normal(size=(m, d))
        fvals = eval_f_p1(X, gamma, a0, a_vec, A)
        mu_hat = fvals.mean()
        m2_hat = np.mean(fvals**2)
        v_hat = m2_hat - mu_hat**2

        mu_b[b] = mu_hat
        m2_b[b] = m2_hat
        var_b[b] = v_hat

    # Treat {var_b} as i.i.d. batch estimates; CLT applies to their average as B grows
    v_bar = float(var_b.mean())
    s_v = float(var_b.std(ddof=1))
    se_vbar = s_v / np.sqrt(B)
    tcrit = tcrit_99(B - 1)
    ci_low = v_bar - tcrit * se_vbar
    ci_high = v_bar + tcrit * se_vbar

    print(f"Batch mean variance estimate = {v_bar:.6f}")
    print(f"99% CI (t, B-1) = [{ci_low:.6f}, {ci_high:.6f}]")
    print(f"Abs error vs analytic = {abs(v_bar - var_true):.6f}")

    # Save batch samples
    zscores = (var_b - v_bar) / s_v
    batch_csv = os.path.join(out_dir, "p1_q1c_batch_variances.csv")
    with open(batch_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["batch", "mu_hat", "m2_hat", "var_hat", "zscore"])
        for i in range(B):
            w.writerow([i, mu_b[i], m2_b[i], var_b[i], zscores[i]])

    # Save summary
    summary_csv = os.path.join(out_dir, "p1_q1c_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["B", "m", "N_total", "var_bar", "se", "ci_low", "ci_high", "var_true", "abs_err"])
        w.writerow([B, m, B*m, v_bar, se_vbar, ci_low, ci_high, var_true, abs(v_bar-var_true)])

    # CLT demonstration: standardized batch estimates should look ~ N(0,1)
    plt.figure()
    plt.hist(zscores, bins=30, density=True)
    xs = np.linspace(-4, 4, 400)
    pdf = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * xs**2)
    plt.plot(xs, pdf, linewidth=2, label="N(0,1) pdf")
    plt.xlabel("Standardized batch estimates (z-score)")
    plt.ylabel("Density")
    plt.title("P1: CLT check using batch variance estimates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "clt_hist_zscores.png"), dpi=200)

    # Optional QQ plot if SciPy exists
    try:
        from scipy.stats import norm  # type: ignore
        p = (np.arange(1, B+1) - 0.5) / B
        theo = norm.ppf(p)
        emp = np.sort(zscores)

        plt.figure()
        plt.plot(theo, emp, marker="o", linestyle="none")
        plt.plot([theo.min(), theo.max()], [theo.min(), theo.max()], linewidth=2)
        plt.xlabel("Theoretical N(0,1) quantiles")
        plt.ylabel("Empirical quantiles of z-scores")
        plt.title("P1: QQ plot of standardized batch variance estimates")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "clt_qq_zscores.png"), dpi=200)
        qq_made = True
    except Exception:
        qq_made = False

    tuning_csv = os.path.join(out_dir, "p1_q1c_tuning.csv")
    with open(tuning_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allres[0].keys()))
        w.writeheader()
        w.writerows(allres)
    print("Saved tuning summary:", tuning_csv)

    print("\nSaved outputs to:", out_dir)
    print(" - p1_q1c_batch_variances.csv")
    print(" - p1_q1c_summary.csv")
    print(" - clt_hist_zscores.png")
    if qq_made:
        print(" - clt_qq_zscores.png")

if __name__ == "__main__":
    main()