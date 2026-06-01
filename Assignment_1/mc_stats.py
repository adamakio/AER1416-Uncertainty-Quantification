# mc_stats.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

Z_99 = 2.5758293035489004  # Phi^{-1}(0.995)

def _t_crit_99(df: int) -> float:
    """
    99% two-sided => 0.995 quantile.
    Uses SciPy if available; otherwise falls back to normal approx (OK when df is large).
    """
    if df <= 0:
        return Z_99
    try:
        from scipy.stats import t  # type: ignore
        return float(t.ppf(0.995, df))
    except Exception:
        # Normal approximation: fine once df ~ 30+
        return Z_99


@dataclass
class EstimateCI:
    est: float
    se: float
    ci_low: float
    ci_high: float


@dataclass
class MCScalarSummary:
    N: int
    batches: int
    batch_size: int
    mean: EstimateCI
    var: EstimateCI


def summarize_mean_var(values: np.ndarray, max_batches: int = 200, min_batch_size: int = 2000) -> MCScalarSummary:
    """
    Returns mean and variance estimates with standard errors + 99% CI.
    - If N is large enough, uses batch estimates (recommended).
    - Otherwise uses direct CLT for mean and a batch-with-few-batches fallback for variance.
    """
    values = np.asarray(values, dtype=float).ravel()
    N = int(values.size)

    # Always compute point estimates from all data
    mean_hat = float(values.mean())
    var_hat = float(values.var(ddof=1)) if N > 1 else float("nan")

    # Choose number of batches B so each batch has at least min_batch_size when possible
    B = min(max_batches, N // min_batch_size) if N >= min_batch_size else 0
    if B < 30 and N >= 30 * min_batch_size:
        B = 30
    if B > 0:
        m = N // B
        used = B * m
        v = values[:used].reshape(B, m)

        batch_means = v.mean(axis=1)
        batch_vars = v.var(axis=1, ddof=1)

        se_mean = float(batch_means.std(ddof=1) / np.sqrt(B))
        se_var = float(batch_vars.std(ddof=1) / np.sqrt(B))

        tcrit = _t_crit_99(B - 1)

        mean_ci = (mean_hat - tcrit * se_mean, mean_hat + tcrit * se_mean)
        var_ci = (var_hat - tcrit * se_var, var_hat + tcrit * se_var)

        return MCScalarSummary(
            N=N,
            batches=B,
            batch_size=m,
            mean=EstimateCI(mean_hat, se_mean, mean_ci[0], mean_ci[1]),
            var=EstimateCI(var_hat, se_var, var_ci[0], var_ci[1]),
        )

    # Fallback: direct CLT SE for mean; for variance we provide a rough SE via batching into ~10 chunks if possible
    s = float(values.std(ddof=1)) if N > 1 else float("nan")
    se_mean = s / np.sqrt(N) if N > 1 else float("nan")
    mean_ci = (mean_hat - Z_99 * se_mean, mean_hat + Z_99 * se_mean)

    # Variance SE fallback: make up to 10 batches if possible
    B2 = min(10, max(2, N // 200)) if N >= 400 else 0
    if B2 >= 2:
        m2 = N // B2
        used2 = B2 * m2
        v2 = values[:used2].reshape(B2, m2)
        batch_vars2 = v2.var(axis=1, ddof=1)
        se_var = float(batch_vars2.std(ddof=1) / np.sqrt(B2))
        tcrit = _t_crit_99(B2 - 1)
        var_ci = (var_hat - tcrit * se_var, var_hat + tcrit * se_var)
    else:
        se_var = float("nan")
        var_ci = (float("nan"), float("nan"))

    return MCScalarSummary(
        N=N,
        batches=B2,
        batch_size=(N // B2) if B2 else 0,
        mean=EstimateCI(mean_hat, se_mean, mean_ci[0], mean_ci[1]),
        var=EstimateCI(var_hat, se_var, var_ci[0], var_ci[1]),
    )


@dataclass
class CDFPoint:
    z: float
    Fhat: float
    ci_low: float
    ci_high: float


def cdf_with_agresti_ci(values: np.ndarray, z_grid: np.ndarray, zcrit: float = Z_99) -> list[CDFPoint]:
    """
    Empirical CDF with Agresti-Coull (approx) 99% CI at each z.
    Good behavior even near 0/1 compared to plain normal approx.

    For each z:
      k = #{f <= z}, n = N
      p_tilde = (k + z^2/2) / (n + z^2)
      half = z * sqrt(p_tilde(1-p_tilde)/(n+z^2))
      CI = p_tilde ± half, clipped to [0,1]
    """
    values = np.asarray(values, dtype=float).ravel()
    z_grid = np.asarray(z_grid, dtype=float).ravel()
    N = int(values.size)

    vals_sorted = np.sort(values)
    out: list[CDFPoint] = []
    z2 = zcrit**2
    denom = N + z2

    for z in z_grid:
        k = int(np.searchsorted(vals_sorted, z, side="right"))
        p_tilde = (k + 0.5 * z2) / denom
        half = zcrit * np.sqrt(p_tilde * (1.0 - p_tilde) / denom)
        lo = max(0.0, p_tilde - half)
        hi = min(1.0, p_tilde + half)
        out.append(CDFPoint(float(z), float(k / N), float(lo), float(hi)))

    return out


def cdf_with_clt_ci(values: np.ndarray, z_grid: np.ndarray, zcrit: float = Z_99):
    values = np.asarray(values, dtype=float).ravel()
    z_grid = np.asarray(z_grid, dtype=float).ravel()
    N = int(values.size)
    vals_sorted = np.sort(values)

    out = []
    for z in z_grid:
        k = int(np.searchsorted(vals_sorted, z, side="right"))
        phat = k / N
        se = np.sqrt(phat * (1 - phat) / N) if N > 0 else float("nan")
        lo = max(0.0, phat - zcrit * se)
        hi = min(1.0, phat + zcrit * se)
        out.append((float(z), float(phat), float(lo), float(hi)))
    return out


def mean_ci_clt_99(values: np.ndarray) -> EstimateCI:
    """
    99% CI for the mean using CLT:
      mean ± z_0.995 * s/sqrt(N)
    """
    values = np.asarray(values, dtype=float).ravel()
    N = int(values.size)
    mu = float(values.mean())
    s = float(values.std(ddof=1)) if N > 1 else float("nan")
    se = s / np.sqrt(N) if N > 1 else float("nan")
    lo = mu - Z_99 * se
    hi = mu + Z_99 * se
    return EstimateCI(mu, se, lo, hi)


def var_ci_sample_variance_99(values: np.ndarray) -> EstimateCI:
    """
    99% CI for the variance using asymptotic SE of the unbiased sample variance.

    For i.i.d. data with central moments mu2=σ^2 and mu4=E[(X-μ)^4],
    Var(s^2) = (1/N) * (mu4 - ((N-3)/(N-1)) * σ^4).

    We plug in:
      mu4 ≈ m4 = mean((x-mean)^4)
      σ^4 ≈ (s^2)^2
    and use a normal approximation CI: s^2 ± z_0.995 * SE.
    """
    x = np.asarray(values, dtype=float).ravel()
    N = int(x.size)
    if N < 4:
        return EstimateCI(float("nan"), float("nan"), float("nan"), float("nan"))

    mu = x.mean()
    s2 = x.var(ddof=1)  # unbiased sample variance
    centered = x - mu
    m4 = float(np.mean(centered**4))

    # Plug-in asymptotic variance of s^2
    var_s2 = (m4 - ((N - 3) / (N - 1)) * (s2**2)) / N
    var_s2 = float(max(var_s2, 0.0))  # numerical guard
    se = float(np.sqrt(var_s2))

    lo = max(0.0, float(s2 - Z_99 * se))
    hi = float(s2 + Z_99 * se)

    return EstimateCI(float(s2), se, lo, hi)