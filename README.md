# AER1416 — Uncertainty Quantification

**University of Toronto (UTIAS) | MEng Aerospace Science & Engineering | Winter 2026**

A graduate course on numerical methods for propagating and quantifying uncertainty through engineering models — progressing from standard Monte Carlo through quasi-random sampling, sparse quadrature, and spectral surrogate methods.

---

## Assignments

| Folder | Topic | Methods | Languages |
|--------|-------|---------|-----------|
| [Assignment_1](Assignment_1/) | Monte Carlo fundamentals | Crude MC, CLT convergence, control variates, linear sensitivity, thin-plate spline surrogate | Python + MATLAB |
| [Assignment_2](Assignment_2/) | Variance reduction & quasi-MC | Latin Hypercube Sampling (LHS), Sobol sequences, Sparse Clenshaw-Curtis (Smolyak) quadrature | MATLAB |
| [Assignment_3](Assignment_3/) | Spectral surrogates | Polynomial Chaos (PC) expansion — mean and variance fields on 3 benchmark problems | MATLAB |

---

## Benchmark Problems

All three assignments share the same set of test problems, enabling direct comparison of methods:

| Problem | Description | Dimension |
|---------|-------------|:---------:|
| **P1** | 10-D quadratic function | 10 |
| **P2** | Mass-spring-damper ODE (parameters m, c, k, ω, F₀ ~ Uniform) | 5 |
| **P3** | Random FEM linear system; QoI = first displacement component | 3 |

---

## Assignment Summaries

### Assignment 1 — Monte Carlo Foundations

Implements and studies fundamental UQ methods in Python and MATLAB:

- **Q1 (P1):** Crude MC convergence — mean/variance estimates vs. sample size with 99% CI; CLT verification via batch variance histograms
- **Q2 (P2):** MC uncertainty propagation through a mass-spring-damper ODE (Welford online variance)
- **Q3 (P3):** Thin-plate spline surrogate fit; MC integration over the surrogate for 2D mean/variance fields
- **Q4:** First-order Taylor (linear) uncertainty propagation for P1, P2, and P3 — finite-difference sensitivities
- **Q5 (P1):** Control variate variance reduction — optimal coefficient λ\*, variance reduction ratios at two nonlinearity levels

### Assignment 2 — Quasi-MC and Sparse Quadrature

Benchmarks three variance-reduction strategies on P1, P2, P3 across sample sizes N = 1k–200k:

- **Q1 — LHS:** Stratified sampling via Latin Hypercube; convergence plots for mean and variance
- **Q2 — Sobol:** Quasi-random Sobol sequences (i4_sobol library); faster convergence than crude MC
- **Q3 — Sparse Clenshaw-Curtis:** Smolyak sparse-grid quadrature; convergence vs. Smolyak level (0–7); exact integral for smooth integrands

### Assignment 3 — Polynomial Chaos

Builds a Polynomial Chaos expansion surrogate for each benchmark problem:

- **Q1 (P1/P2/P3):** PC expansion coefficients via sparse-grid quadrature; mean and variance surfaces as 3D and contour plots; comparison with MC reference solutions

---

## Tools

Python · NumPy · SciPy · Matplotlib · MATLAB
