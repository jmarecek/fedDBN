"""Scalable synthetic dynamic Bayesian networks (for d beyond ExDBN's generator).

ExDBN's generator (CausalNex-derived) simulates node by node in Python and rejects series
whose magnitude exceeds 1e3; with ExDBN's intra-slice weights in [0.5, 2] the rejection rate
explodes beyond d of about 25.  Here the intra-slice DAG is drawn with psiDAG's generator
(data_generator.generate_DAG, weights in +-[0.05, 1]), the inter-slice matrix follows ExDBN
(ER with d edges per lag, |weights| in [0.2, 0.4], decay 1.1 per lag), and A is rescaled if
needed so that the VAR is stable.  The series is simulated in vectorised form
    x_t = (sum_k x_{t-k} A_k + z_t) (I - W)^{-1}.
"""
import importlib.util
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]


def _psidag_generator():
    spec = importlib.util.spec_from_file_location("psidag_data_generator", _ROOT / "data_generator.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def simulate_dbn(d, n, p=1, intra_degree=4, w_inter=(0.2, 0.4), decay=1.1, noise=1.0,
                 rho_max=0.9, burn_in=200, seed=0):
    gen = _psidag_generator()
    B_ranges = ((-1.0, -0.05), (0.05, 1.0))
    W = gen.generate_DAG(d, "ER", intra_degree, B_ranges, hd=False, seed=seed).B
    rs = np.random.RandomState(seed + 7919)
    A = []
    for k in range(p):
        mask = np.zeros((d, d))
        idx = rs.choice(d * d, size=d, replace=False)
        mask.flat[idx] = 1.0
        U = rs.uniform(*w_inter, size=(d, d)) * rs.choice([-1.0, 1.0], size=(d, d)) / decay ** k
        A.append(mask * U)
    G = np.linalg.inv(np.eye(d) - W)
    # companion-matrix stability check for the reduced-form VAR x_t = sum_k x_{t-k} A_k G + e_t
    comp = np.zeros((p * d, p * d))
    for k in range(p):
        comp[k * d:(k + 1) * d, :d] = A[k] @ G
    if p > 1:
        comp[:-d, d:] = np.eye((p - 1) * d)
    rho = np.max(np.abs(np.linalg.eigvals(comp)))
    if rho > rho_max:
        A = [a * rho_max / rho for a in A]
    T = n + burn_in + p
    Xs = np.zeros((T, d))
    E = rs.normal(scale=noise, size=(T, d))
    AG = [a @ G for a in A]
    for t in range(p, T):
        Xs[t] = E[t] @ G + sum(Xs[t - k - 1] @ AG[k] for k in range(p))
    Xs = Xs[burn_in:]
    X = Xs[p:]
    Y = np.stack([Xs[p - k - 1: len(Xs) - k - 1] for k in range(p)])
    return dict(X=X, Y=Y, W_true=W, A_true=np.stack(A), p=p, rho=rho)
