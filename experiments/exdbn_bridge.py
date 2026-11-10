"""Thin bridge to the ExDBN code base (https://github.com/pavelrt/ExDBN).

We call ExDBN's own data generator (``load_problem``), its MILP solver
(``solve_milp.solve``) and its evaluation code (``find_optimal_threshold_for_shd``,
``count_accuracy``) so that the comparison is on exactly the data and the metrics
used in the ExDBN paper.  Set EXDBN_PATH to a checkout of ExDBN.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import time

import numpy as np

EXDBN_PATH = os.environ.get("EXDBN_PATH")
if not EXDBN_PATH or not os.path.isdir(EXDBN_PATH):
    raise RuntimeError("Set EXDBN_PATH to a checkout of https://github.com/pavelrt/ExDBN")
if EXDBN_PATH not in sys.path:
    sys.path.insert(0, EXDBN_PATH)

import utils as _ex_utils  # noqa: E402  (ExDBN's utils, not psiDAG's)
from data_generation_loading_utils import load_problem  # noqa: E402
from dagsolver_utils import (ExDagDataException, apply_threshold,  # noqa: E402
                             count_accuracy, find_optimal_threshold_for_shd)

assert hasattr(_ex_utils, "simulate_linear_sem"), "wrong `utils` module picked up"

# ExDBN's experimental grid (run_experiments.py): (p, intra_edge_ratio, w_min_inter,
# w_max_inter, noise_scale, noise_scale_variance)
CONFIGS = {
    "ER3-p1": ("er", (1, 3, 0.2, 0.4, 1.0, None)),
    "ER3-p1-hetnoise": ("er", (1, 3, 0.2, 0.4, 0.8, 0.4)),
    "ER2-p2": ("er", (2, 2, 0.3, 0.5, 1.0, None)),
    "SF3-p1": ("sf", (1, 3, 0.2, 0.4, 1.0, None)),
}


def generate(config: str, d: int, n: int, seed: int):
    """Exactly ExDBN's start_experiment() data path for 'dynamic' problems."""
    variant, (p, ier, wmin, wmax, ns, nsv) = CONFIGS[config]
    _ex_utils.set_random_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        W_true, _, B_true, _, A_true, X, Y, *_ = load_problem(
            "dynamic", variant, d, n, p, ier, 1, wmax, wmin, 1.1, ns, nsv)
    return dict(X=X, Y=np.stack(Y), W_true=W_true, A_true=np.stack(A_true), p=p)


def solve_exdbn(X, Y, time_limit: float, threads: int):
    import gurobipy as gp
    import solve_milp
    gp.setParam("OutputFlag", 0)
    gp.setParam("Threads", threads)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        W, A, gap, lazy, _ = solve_milp.solve(X, 0.1, 0.1, 0, Y=list(Y), B_ref=None,
                                              tabu_edges=None, time_limit=time_limit)
    dt = time.perf_counter() - t0
    if W is None:
        return None
    return dict(W=W, A=np.stack(A), gap=gap, lazy=lazy, time=dt)


def evaluate(W_true, A_true, W_est, A_est, fixed_thresholds=(0.1,)):
    """SHD/F1 with ExDBN's metric code: at the SHD-optimal threshold (as reported by ExDBN)
    and at fixed thresholds."""
    B_true = W_true != 0
    A_true = [a for a in A_true]
    A_est = [a for a in A_est]
    zeros = np.zeros_like(W_true)
    out = {}
    best_t, _ = find_optimal_threshold_for_shd(B_true, W_est, A_true, A_est, zeros, zeros)
    for name, t in [("best", best_t)] + [(f"t{t}", t) for t in fixed_thresholds]:
        W_t = apply_threshold(W_est, t)
        A_t = [apply_threshold(a, t) for a in A_est]
        if not _ex_utils.is_dag(W_t):
            out[f"shd_{name}"] = np.nan
            continue
        acc = count_accuracy(B_true.astype(int), (W_t != 0).astype(int), A_true, A_t)
        tp, fp, fn = acc["true_pos"], acc["false_pos"], acc["false_neg"]
        out[f"shd_{name}"] = acc["shd"]
        out[f"shdW_{name}"] = acc["shd_w"]
        out[f"shdA_{name}"] = acc["shd_a"]
        out[f"f1_{name}"] = 2 * tp / max(2 * tp + fp + fn, 1)   # standard F1
        out[f"tpr_{name}"] = acc["tpr"]
        out[f"fdr_{name}"] = acc["fdr"]
    out["threshold_best"] = best_t
    out["n_true_edges"] = int(B_true.sum() + sum((a != 0).sum() for a in A_true))
    return out


def solve_exdbn_gram(X, Y, time_limit: float, threads: int, lambda1=0.1, weights_bound=100.0,
                     mip_gap=1e-4):
    """ExDBN's MIQP written in terms of sufficient statistics.

    Same variables, big-M constraints, l0 penalties, 2-cycle constraints and lazy cycle cuts as
    ``solve_milp.solve`` (loss 'l2', constraints 'weights', reg 'l1' on W and A indicator
    variables), but the least-squares term sum_{i,j} (X_ij - Z_i theta_j)^2 / n is written as
    sum_j theta_j' S theta_j - 2 c_j' theta_j + const with S = Z'Z/n, c_j = Z'x_j/n.  The model
    size no longer grows with n, and S can be computed federatedly by secure aggregation of
    the clients' local Gram matrices in a single round.
    """
    import gurobipy as gp
    from gurobipy import GRB
    import solve_milp as sm
    n, d = X.shape
    p = len(Y)
    Z = np.concatenate([X, *Y], axis=1)
    S = Z.T @ Z / n
    C = Z.T @ X / n
    t0 = time.perf_counter()
    gp.setParam("OutputFlag", 0)
    gp.setParam("Threads", threads)
    m = gp.Model()
    Wv, Ww = sm.construct_matrix_vars(m, d, "W", "weights", weights_bound, None)
    for v1 in range(d):
        for v2 in range(v1):
            m.addConstr(Wv[v2, v1] + Wv[v1, v2] <= 1)
    Av, Aw = [], []
    for t in range(p):
        a_v, a_w = sm.construct_matrix_vars(m, d, f"A_{t}", "", weights_bound, None, diagonal=True)
        Av.append(a_v)
        Aw.append(a_w)
    obj = gp.QuadExpr()
    for j in range(d):
        rows = [k for k in range(d) if k != j] + [d + t * d + k for t in range(p) for k in range(d)]
        vars_j = [Ww[k, j] for k in range(d) if k != j] + [Aw[t][k, j] for t in range(p) for k in range(d)]
        Sj = S[np.ix_(rows, rows)]
        mv = gp.MVar.fromlist(vars_j)
        obj += mv @ Sj @ mv - 2.0 * (C[rows, j] @ mv) + float(S[j, j])
    reg = gp.quicksum(Wv.values()) + gp.quicksum(v for a in Av for v in a.values())
    m.setObjective(obj + lambda1 * reg / d, GRB.MINIMIZE)
    m.Params.lazyConstraints = 1
    m.Params.MIPGap = mip_gap
    m.Params.TimeLimit = time_limit
    m._edges_vars, m._edges_weights = Wv, Ww
    m._lazy_count, m._last_time_stats, m._B_ref, m._stats = 0, 0, None, []
    m._d, m._callback_mode = d, "all_cycles"
    build = time.perf_counter() - t0
    m.optimize(sm.check_for_cycles)
    total = time.perf_counter() - t0
    res = dict(build=build, time=total, status=m.Status, n_bin=m.NumBinVars)
    if m.SolCount == 0:
        res.update(failed=True, gap=np.inf)
        m.dispose()
        return res
    W = sm.extract_adj_matrix(m.getAttr("x", Wv), m.getAttr("x", Ww), d)
    A = np.stack([sm.extract_adj_matrix_no_vars(m.getAttr("x", a), d) for a in Aw])
    res.update(failed=False, W=W, A=A, gap=m.MIPGap, obj=m.ObjVal, bound=m.ObjBound)
    m.dispose()
    return res
