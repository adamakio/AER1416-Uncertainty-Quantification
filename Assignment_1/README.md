# AER 1416 Assignment 1: Uncertainty Quantification

**Author:** Zouhair Adam Hamaimou (1004891986)

---

## Overview

This directory contains solutions for Assignment 1 in AER 1416 (Uncertainty Quantification).
---

## Environment Requirements

### Python Environment

**Python Version:** 3.8 or higher

**Required Python Packages:**
- `numpy` – numerical computing
- `matplotlib` – plotting and visualization
- `scipy` – scientific computing (optional, for improved statistical functions)

Install dependencies with:
```bash
pip install numpy matplotlib scipy
```

### MATLAB/Octave Environment (for Q3 and Q4_P3)

- **MATLAB** with core libraries

---

## Directory Structure

The working directory should contain the following files before running the scripts:

### Core Model Files (Required for Python scripts)
```
p1_model.py          – Defines the P1 quadratic model and coefficient builder
mc_stats.py          – Statistical utilities for Monte Carlo analysis
```

### Data Files (Required for Q3 and Q4_P3)
```
P3.mat               – MATLAB data file containing problem parameters
P3grid.mat           – Grid data for contour plotting
```

### Plotting Helper (Required for Q3)
```
tricontour.m         – Helper function for triangulated contour plotting
PlotP3_3D.m          – 3D surface plotting script
PlotP3_contour.m     – Contour plot script
PlotP3_contour_var.m – Variance contour plot script
```

### Output Directories
The following directories will be **automatically created** by their respective scripts:
```
q1b_outputs/         – Outputs from run_q1b_p1.py (auto-created)
q1c_outputs/         – Outputs from run_q1c_p1.py (auto-created)
q5_outputs/          – Outputs from run_q5.py (auto-created)
```

Optional (create manually if needed for organization):
```
q2_outputs/          – For saving output from run_q2.py (console/plots only)
q3_outputs/          – For saving figures from run_Q3.m
```

Note: `run_q4_p1_p2.py` and `run_Q4_P3.m` produce console output only.

---

## How to Run the Scripts

### Python Scripts

**All Python scripts should be run from the directory containing this README.**

- `run_q1b_p1.py` and `run_q1c_p1.py` automatically create their output directories
- `run_q2.py` and `run_q4_p1_p2.py` do not create output directories; they print to console or display interactive plots

#### Q1b: Monte Carlo Convergence Study (P1)

**File:** `run_q1b_p1.py`

**Purpose:**
Demonstrates convergence of Monte Carlo estimation for the mean and variance of a quadratic function under increasing sample sizes. Uses the Central Limit Theorem (CLT) and computes 99% confidence intervals.

**Run with:**
```bash
python run_q1b_p1.py
```

**Outputs saved to `q1b_outputs/`:**
- `p1_q1b_convergence.csv` – Convergence table with estimates, errors, and CI widths for N = [5k, 10k, 50k, 100k, 500k, 1M]
- `cdf.png` – Empirical CDF plot

---

#### Q1c: Batch Monte Carlo and CLT Verification (P1)

**File:** `run_q1c_p1.py`

**Purpose:**
Validates the Central Limit Theorem by examining the distribution of batch variance estimates. The script:
1. Tunes the batch size ($m$) and number of batches ($B$) to optimize normality of batch estimates
2. Verifies that properly scaled batch estimates are approximately $N(0,1)$
3. Uses t-distribution for confidence intervals (more accurate than normal approximation)

**Run with:**
```bash
python run_q1c_p1.py
```

**Outputs saved to `q1c_outputs/`:**
- `p1_q1c_summary.csv` – Summary statistics for the chosen (m, B) configuration
- `p1_q1c_batch_variances.csv` – Individual batch estimates and z-scores
- `p1_q1c_tuning.csv` – Results of tuning grid search
- `clt_hist_zscores.png` – Histogram of standardized batch estimates vs. $N(0,1)$ density
- `clt_qq_zscores.png` – Q-Q plot for normality assessment (if SciPy is available)

---

#### Q2: Mass-Spring-Damper System with Uncertain Parameters

**File:** `run_q2.py`

**Purpose:**
Analyzes the uncertainty propagation through an ODE system:
$$m\ddot{u} + c\dot{u} + ku = F_0\cos(\omega t), \quad u(0)=\dot{u}(0)=0$$

The script computes mean and variance of displacement at $t=10$ seconds under uniformly-distributed uncertain parameters. Features:
- Uncertainty in mass, damping, stiffness, forcing amplitude, and frequency
- Online variance tracking (Welford's method) to avoid storing large trajectory matrices
- Time-history statistics of mean displacement and variance over time

**Run with:**
```bash
python run_q2.py
```

**Console Output:**
- Convergence table for mean and variance at $u(T)$ across multiple sample sizes
- Time integration statistics
- Mean displacement and variance at final time

*Note:* Visualization is interactive (requires display). Comment out `plt.show()` calls if running on a headless system.

---

#### Q4 (P1 & P2): First-Order Taylor Approximation for Uncertainty

**File:** `run_q4_p1_p2.py`

**Purpose:**
Applies first-order (linear) uncertainty propagation to:
1. **P1 (Quadratic function):** Compares analytical mean/variance to first-order linear approximation
2. **P2 (Mass-spring-damper):** Uses finite-difference gradients to estimate sensitivities and propagates uncertainties using the linear approximation formula:
$$\text{Var}[u(T)] = \sum_{\alpha} \left(\frac{\partial u}{\partial \alpha}\right)^2 \text{Var}(\alpha)$$

**Run with:**
```bash
python run_q4_p1_p2.py
```

**Console Output:**
- Comparison of analytical vs. linear approximation errors for P1
- Partial derivatives (sensitivities) for each parameter in P2
- Linear approximation mean and variance for P2

*Note:* This is primarily a computational script; no CSV or plot files are generated (output is printed to console).

---

#### Q5: Control Variate Variance Reduction

**File:** `run_q5.py`

**Purpose:**
Demonstrates variance reduction through control variates. For the quadratic function P1, uses the linear part as a control variate:
$$\hat{\mu}_{CV} = \hat{\mu}_f - \lambda^* (\hat{\mu}_g - E[g])$$
where $\lambda^*$ is the optimal coefficient minimizing the estimator variance.

The script runs experiments at two levels of nonlinearity ($\gamma = 0.5$ and $\gamma = 0.1$).

**Run with:**
```bash
python run_q5.py
```

**Outputs saved to `q5_outputs/`:**
- `q5_histogram_gamma0.5.png` – Overlay of MC and CV mean estimate distributions (200 repeated runs, N=1000)
- `q5_histogram_gamma0.1.png` – Same for lower nonlinearity

**Console Output:**
- Convergence tables showing mean/variance estimates and 99% CIs for both MC and CV methods
- Variance reduction ratio: $\text{Var}(CV) / \text{Var}(MC)$
- Optimal control variate coefficient $\lambda^*$

---

### MATLAB/Octave Scripts

#### Q3: Surrogate Modeling via Thin-Plate Splines (P3)

**File:** `run_Q3.m`

**Purpose:**
Constructs a thin-plate spline surrogate model for a 2D problem using the data in `P3.mat`. The surrogate provides fast emulation of an expensive simulator.

**Data files required:**
- `P3.mat` – Contains sample locations and function values
- `P3grid.mat` – Evaluation grid for contour plots

**Helper files:**
- `tricontour.m` – Triangulation-based contour plotting

**Run in MATLAB/Octave:**
```matlab
run_Q3
```

**Typical outputs:**
- Plots of the surrogate fit and cross-validation errors
- Contour plots saved as figures (format depends on MATLAB settings)

---

#### Q4 (P3): Linear Approximation and Sensitivity (P3)

**File:** `run_Q4_P3.m`

**Purpose:**
Applies first-order Taylor expansion to the 2D surrogate model from Q3:
- Builds the Hessian matrix of the surrogate at a nominal point
- Computes uncertainty propagation for small parameter perturbations
- Estimates the variance of the output under uncertain inputs

**Data files required:**
- `P3.mat` – Problem parameters

**Run in MATLAB/Octave:**
```matlab
run_Q4_P3
```

**Console Output:**
- Nominal function value and gradient
- Hessian matrix
- Linear approximation variance estimate
- Condition number of the Hessian

---

## File Dependencies Summary

| Script | Depends On | Data Files | Output Type | Auto-created |
|--------|-----------|-----------|-----------|-----------|
| `run_q1b_p1.py` | `p1_model.py`, `mc_stats.py` | — | CSV + PNG | ✓ `q1b_outputs/` |
| `run_q1c_p1.py` | `p1_model.py` | — | CSV + PNG | ✓ `q1c_outputs/` |
| `run_q2.py` | — | — | Console + interactive plots | — |
| `run_q4_p1_p2.py` | `p1_model.py` | — | Console output | — |
| `run_q5.py` | `p1_model.py` | — | PNG + console | ✓ `q5_outputs/` |
| `run_Q3.m` | `tricontour.m` | `P3.mat`, `P3grid.mat` | MATLAB figures | — |
| `run_Q4_P3.m` | — | `P3.mat` | Console output | — |

---

## Notes

### Python Specific

- **NumPy random number generation:** Scripts use `np.random.default_rng(seed)` for reproducibility. Set `seed` parameter to get consistent results across runs.
- **CSV output:** All outputs use standard CSV format; compatible with Excel, pandas, or any spreadsheet software.
- **Optional SciPy:** Some scripts (e.g., `run_q1c_p1.py`) optionally use SciPy's `t.ppf()` for more accurate t-critical values. The code gracefully falls back to normal approximation if SciPy is unavailable.

### MATLAB/Octave Specific

- Ensure `P3.mat` and `P3grid.mat` are in the working directory before running Q3/Q4_P3 scripts.
- Thin-plate spline fitting uses built-in MATLAB functions (`tpaps` or equivalent Octave routines).
- Plotting may behave differently in Octave vs. MATLAB; adjust as needed.

---

## Questions?

Email zouhair.hamaimou@mail.utoronto.ca for prompt answers about any of the scripts.
