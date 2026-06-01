# AER1416 Assignment 2 — UQ Methods

This repository contains MATLAB scripts for uncertainty quantification of three benchmark problems (P1, P2, P3) using three sampling/quadrature methods.

## Repository Layout

```
├── Q1/          # Latin Hypercube Sampling (LHS)
├── Q2/          # Quasi-Monte Carlo via Sobol sequences
├── Q3/          # Sparse Clenshaw-Curtis (Smolyak) quadrature
└── P3/          # Shared data and plotting utilities for P3
```

## Benchmark Problems

| Problem | Description | Dimensions | Random variables |
|---------|-------------|:----------:|-----------------|
| P1 | 10-D quadratic function | 10 | xi ~ U[-1, 1] |
| P2 | Mass-spring-damper (ODE, solved with `ode45`) | 5 | (m, c, k, omega, F0) ~ U[lb, ub] |
| P3 | Random linear system (FEM), QoI = u1 | 3 | theta ~ U[-3, 3] |

## Prerequisites

- MATLAB (any recent version with the ODE solver `ode45`)
- No additional toolboxes required — all quadrature/sampling utilities are bundled in each question folder

## Running the Scripts

Each script must be run **from inside its own folder** so that relative paths to `../P3/` resolve correctly.

### Q1 — Latin Hypercube Sampling

```matlab
cd Q1
Q1
```

**What it does:**
- Draws LHS samples at N = 1000, 5000, 10000, 20000, 50000, 100000, 200000
- Estimates mean and variance of P1, P2, and P3 at each sample size
- Prints a convergence table to the command window
- Saves convergence plots: `LHS_P{1,2,3}_mean.png`, `LHS_P{1,2,3}_var.png`
- Computes and saves the full P3 mean/variance field at N = 200000:
  `LHS_P3_mean_3D.png`, `LHS_P3_mean_contour.png`, `LHS_P3_var_3D.png`, `LHS_P3_var_contour.png`

**Dependencies in `Q1/`:** `lhsamp.m`

---

### Q2 — Quasi-Monte Carlo (Sobol Sequences)

```matlab
cd Q2
Q2
```

**What it does:**
- Generates Sobol sequence points at the same sample sizes as Q1
- Estimates mean and variance of P1, P2, and P3
- Prints a convergence table and saves plots prefixed `Sobol_`
- Computes the full P3 field at N = 200000 (plots prefixed `Sobol_P3_`)

**Dependencies in `Q2/`:** `i4_sobol_generate.m` and its supporting routines (`i4_sobol.m`, `i4_bit_hi1.m`, `i4_bit_lo0.m`, `prime_ge.m`, `i4_uniform.m`, `r4_uniform_01.m`, `tau_sobol.m`, `timestamp.m`)

---

### Q3 — Sparse Clenshaw-Curtis (Smolyak) Quadrature

```matlab
cd Q3
Q3
```

**What it does:**
- Sweeps Smolyak levels 0–4 for P1 (d=10) and levels 0–6 for P2 (d=5) and P3 (d=3)
- Computes exact (quadrature-based) mean and variance at each level
- Prints a convergence table (level, number of quadrature points, mean, variance)
- Saves convergence plots prefixed `SCC_`
- Computes the full P3 field at level 7 and saves field plots prefixed `SCC_P3_`

**Dependencies in `Q3/`:** `sparse_grid_cc.m` and all supporting routines (listed in the header of `Q3.m`)

---

## Shared Data (`P3/`)

All three scripts add `../P3` to the MATLAB path automatically. Do not move or rename files in this folder.

| File | Contents |
|------|----------|
| `P3.mat` | System matrices A0, A1, A2, A3 and RHS vector b |
| `P3grid.mat` | FEM mesh data (node coordinates `p`, interior indices `ind_interior`) |
| `PlotP3_3D.m` | 3-D surface plot utility |
| `PlotP3_contour.m` | Contour plot utility |
| `tricontour.m` | Triangle-based contour renderer |

## Output Files

All figures are saved as `.png` files in the folder of the script that produced them (i.e. `Q1/`, `Q2/`, or `Q3/`).
