# p1_model.py
from __future__ import annotations
import numpy as np

def build_coeffs_p1(d: int) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Build coefficients for P1:
      a0 = 1
      ai = 1/sqrt(i), i=1..d
      aij = 1/i if i=j
            min(i,j) / (max(i,j)^2 * (i+j)) if i!=j
    Returns (a0, a_vec shape (d,), A shape (d,d)).
    """
    a0 = 1.0
    i = np.arange(1, d + 1, dtype=float)  # 1..d
    a_vec = 1.0 / np.sqrt(i)

    # Build A with 1-indexed formula using broadcasting
    I, J = np.meshgrid(i, i, indexing="ij")  # both float, shape (d,d)
    A = np.empty((d, d), dtype=float)

    diag = (I == J)
    A[diag] = 1.0 / I[diag]

    # off-diagonal
    minIJ = np.minimum(I, J)
    maxIJ = np.maximum(I, J)
    A[~diag] = minIJ[~diag] / (maxIJ[~diag] ** 2 * (I[~diag] + J[~diag]))

    return a0, a_vec, A


def eval_f_p1(X: np.ndarray, gamma: float, a0: float, a_vec: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Evaluate f(x) for a batch of samples X of shape (N,d):
      f(x) = a0 + sum_i ai xi + gamma * sum_{i,j} aij xi xj
           = a0 + X @ a_vec + gamma * (x^T A x)
    Returns f values shape (N,).
    """
    lin = X @ a_vec
    quad = np.einsum("bi,ij,bj->b", X, A, X)  # vectorized x^T A x
    return a0 + lin + gamma * quad


def analytic_mean_var_p1(d: int, gamma: float) -> tuple[float, float]:
    """
    Analytical mean and variance for P1 with xi ~ N(0,1) independent:

      E[f] = a0 + gamma * sum_i a_ii = 1 + gamma * sum_{i=1}^d (1/i)

      Var[f] = Var(L) + Var(Q)
             = sum_i a_i^2 + 2*gamma^2 * sum_{i,j} a_ij^2
             = sum_{i=1}^d (1/i) + 2*gamma^2 * sum_{i,j} a_ij^2

    Returns (mean, variance).
    """
    a0, a_vec, A = build_coeffs_p1(d)

    H_d = np.sum(1.0 / np.arange(1, d + 1, dtype=float))
    mean = a0 + gamma * H_d 

    var_L = H_d     
    var_Q = 2.0 * (gamma ** 2) * np.sum(A ** 2)
    var = var_L + var_Q
    return float(mean), float(var)