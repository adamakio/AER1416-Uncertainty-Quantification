import time
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# -----------------------------
# Model + parameters
# -----------------------------

@dataclass(frozen=True)
class MSDInputs:
    """Random inputs for a single mass-spring-damper run."""
    mass: float
    damping: float
    stiffness: float
    force_amp: float
    omega: float


class MassSpringDamper:
    """Mass-spring-damper with harmonic forcing: m u¨ + c u˙ + k u = F0 cos(ω t)."""

    def __init__(self, inputs: MSDInputs) -> None:
        self.p = inputs

    def rhs(self, t: float, state: np.ndarray) -> Tuple[float, float]:
        """
        State is [u, u_dot]. Returns [u_dot, u_ddot].
        """
        u, u_dot = state
        u_ddot = (
            self.p.force_amp * np.cos(self.p.omega * t)
            - self.p.damping * u_dot
            - self.p.stiffness * u
        ) / self.p.mass
        return (u_dot, u_ddot)


class IVPSolver:
    """Small wrapper around solve_ivp so the rest of the code stays tidy."""

    def __init__(self, method: str = "RK45") -> None:
        self.method = method

    def solve_displacement_at(self, system: MassSpringDamper, t_end: float) -> float:
        """Integrate and return u(t_end) only."""
        y0 = np.array([0.0, 0.0], dtype=float)

        sol = solve_ivp(
            fun=system.rhs,
            t_span=(0.0, t_end),
            y0=y0,
            t_eval=[t_end],
            method=self.method,
        )
        return float(sol.y[0, -1])

    def solve_displacement_series(self, system: MassSpringDamper, t_grid: np.ndarray) -> np.ndarray:
        """Integrate and return u(t) on the provided time grid."""
        y0 = np.array([0.0, 0.0], dtype=float)

        sol = solve_ivp(
            fun=system.rhs,
            t_span=(float(t_grid[0]), float(t_grid[-1])),
            y0=y0,
            t_eval=t_grid,
            method=self.method,
        )
        return sol.y[0].astype(float)


# -----------------------------
# Sampling utilities
# -----------------------------

class UniformInputSampler:
    """Draws statistically independent inputs from the given uniform ranges."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def draw_many(self, count: int) -> List[MSDInputs]:
        m = self.rng.uniform(0.9, 1.1, size=count)
        k = self.rng.uniform(10.0, 12.0, size=count)
        c = self.rng.uniform(0.8, 1.2, size=count)
        w = self.rng.uniform(0.1, 0.2, size=count)
        f0 = self.rng.uniform(0.9, 1.1, size=count)

        return [
            MSDInputs(
                mass=float(m[i]),
                damping=float(c[i]),
                stiffness=float(k[i]),
                force_amp=float(f0[i]),
                omega=float(w[i]),
            )
            for i in range(count)
        ]


# -----------------------------
# Stats helpers
# -----------------------------

@dataclass(frozen=True)
class MCScalarSummary:
    mean: float
    variance: float
    se_mean: float
    ci_halfwidth_mean: float
    se_var: float
    ci_halfwidth_var: float


class ConfidenceIntervals:
    def __init__(self, z_value: float = 2.58) -> None:
        # z ≈ 2.58 for 99% (normal approximation)
        self.z = z_value

    def summarize_scalar(self, samples: np.ndarray) -> MCScalarSummary:
        n = int(samples.size)
        mu = float(np.mean(samples))
        s2 = float(np.var(samples, ddof=1))
        s = float(np.sqrt(s2))

        se_mu = s / np.sqrt(n)
        hw_mu = self.z * se_mu

        # Common approximation used in many MC notes: SE(var) ≈ sqrt(2/N) * var
        se_s2 = np.sqrt(2.0 / n) * s2
        hw_s2 = self.z * se_s2

        return MCScalarSummary(
            mean=mu,
            variance=s2,
            se_mean=se_mu,
            ci_halfwidth_mean=hw_mu,
            se_var=float(se_s2),
            ci_halfwidth_var=float(hw_s2),
        )


class RunningVectorMoments:
    """
    Online mean/variance for vectors using Welford's method.
    Avoids storing all trajectories (big memory win).
    """

    def __init__(self, length: int) -> None:
        self.n = 0
        self.mean = np.zeros(length, dtype=float)
        self.M2 = np.zeros(length, dtype=float)

    def update(self, x: np.ndarray) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.n < 2:
            raise ValueError("Need at least two samples to compute variance.")
        var = self.M2 / (self.n - 1)
        return self.mean.copy(), var


# -----------------------------
# Experiment runner (high-level)
# -----------------------------

class MonteCarloMSDStudy:
    def __init__(self, sampler: UniformInputSampler, solver: IVPSolver, ci: ConfidenceIntervals) -> None:
        self.sampler = sampler
        self.solver = solver
        self.ci = ci

    def run_final_time_sweep(self, counts: Iterable[int], t_end: float) -> None:
        print("Monte Carlo study: m u¨ + c u˙ + k u = F0 cos(ω t)")
        print(f"Reporting displacement at t = {t_end} with ~99% confidence intervals.\n")
        print("-" * 110)
        print("  Samples |   mean(u)    SE(mean)   ±CI(mean) |   var(u)     SE(var)    ±CI(var)   | time")
        print("-" * 110)

        for n in counts:
            t0 = time.time()
            draws = self.sampler.draw_many(n)
            u_end = self._simulate_final_displacements(draws, t_end)
            summary = self.ci.summarize_scalar(u_end)
            dt = time.time() - t0

            print(
                f"{n:9d} | "
                f"{summary.mean:9.6f}  {summary.se_mean:9.6f}  ±{summary.ci_halfwidth_mean:9.6f} | "
                f"{summary.variance:9.6f}  {summary.se_var:9.6f}  ±{summary.ci_halfwidth_var:9.6f} | "
                f"{dt:5.2f}s"
            )

    def _simulate_final_displacements(self, draws: List[MSDInputs], t_end: float) -> np.ndarray:
        out = np.zeros(len(draws), dtype=float)
        for i, params in enumerate(draws):
            system = MassSpringDamper(params)
            out[i] = self.solver.solve_displacement_at(system, t_end)
        return out

    def run_time_history(self, n_batches: int, batch_size: int, t_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        total = n_batches * batch_size
        print(f"\nTime-history run: {n_batches} batches × {batch_size} each = {total} trajectories")
        print("Computing mean/variance over time (online; no huge trajectory matrix stored).")

        draws = self.sampler.draw_many(total)
        tracker = RunningVectorMoments(length=len(t_grid))

        t0 = time.time()
        for params in draws:
            system = MassSpringDamper(params)
            u_series = self.solver.solve_displacement_series(system, t_grid)
            tracker.update(u_series)
        dt = time.time() - t0
        print(f"Integration time: {dt:.2f}s")

        mean_u, var_u = tracker.finalize()
        return mean_u, var_u

    def plot_bands(
        self,
        t_grid: np.ndarray,
        mean_u: np.ndarray,
        var_u: np.ndarray,
        z_value: float,
        n_samples: int,
    ) -> None:
        std_u = np.sqrt(var_u)

        # Mean CI
        se_mean = std_u / np.sqrt(n_samples)
        lo_mean = mean_u - z_value * se_mean
        hi_mean = mean_u + z_value * se_mean

        # Variance CI (approx)
        se_var = np.sqrt(2.0 / n_samples) * var_u
        lo_var = var_u - z_value * se_var
        hi_var = var_u + z_value * se_var

        plt.figure(figsize=(7, 5))
        plt.plot(t_grid, mean_u, label="E[u(t)]")
        plt.fill_between(t_grid, lo_mean, hi_mean, alpha=0.2, label="~99% CI for mean")
        plt.title(f"Mean displacement vs time (N={n_samples})")
        plt.xlabel("t")
        plt.ylabel("mean u(t)")
        plt.grid(True)
        plt.legend()
        plt.show()

        plt.figure(figsize=(7, 5))
        plt.plot(t_grid, var_u, label="Var[u(t)]")
        plt.fill_between(t_grid, lo_var, hi_var, alpha=0.2, label="~99% CI for variance")
        plt.title(f"Variance of displacement vs time (N={n_samples})")
        plt.xlabel("t")
        plt.ylabel("var u(t)")
        plt.grid(True)
        plt.legend()
        plt.show()


def run():
    # Simulation parameters
    final_time = 10.0
    sample_counts = [1000, 5000, 10000, 50000, 200000]

    z_99 = 2.58  # normal approx for 99%
    sampler = UniformInputSampler(seed=0)
    solver = IVPSolver(method="RK45")
    ci = ConfidenceIntervals(z_value=z_99)

    study = MonteCarloMSDStudy(sampler=sampler, solver=solver, ci=ci)

    # Part A: scalar output u(t_end) for multiple N
    study.run_final_time_sweep(sample_counts, t_end=final_time)

    # Part B: time history (mean/var over t)
    n_batches = 200
    batch_size = 1000
    n_total = n_batches * batch_size
    t_grid = np.linspace(0.0, final_time, 201)

    mean_u, var_u = study.run_time_history(n_batches, batch_size, t_grid)
    study.plot_bands(t_grid, mean_u, var_u, z_value=z_99, n_samples=n_total)

    print(f"\nAt t={final_time:.1f}: mean[u(t)] ≈ {mean_u[-1]:.6f}, var[u(t)] ≈ {var_u[-1]:.6f}")


if __name__ == "__main__":
    run()