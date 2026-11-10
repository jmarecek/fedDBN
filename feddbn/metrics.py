"""Vectorised re-implementation of ExDBN's evaluation (dagsolver_utils.calculate_shd et al.).

ExDBN's conventions are kept so that numbers are comparable with the ExDBN paper:
  * SHD is computed on unordered pairs {i, j}, i > j, of each matrix; a reversed pair costs 0.5,
    any other mismatch costs 1.  For the lag matrices A_k the same pairwise rule is applied
    (so self-lag entries A_k[i, i] are not counted in SHD).
  * F1 is computed entrywise on the stacked [W; A_1; ...; A_p] support (standard F1).
  * "best" = the single threshold on |weights| minimising SHD (oracle threshold, as in ExDBN).
The ExDBN implementation is pure Python and O(d^2) per threshold; this one is NumPy.
"""
import numpy as np


def pair_shd(T: np.ndarray, E: np.ndarray) -> float:
    T = T != 0
    E = E != 0
    low = np.tril(np.ones(T.shape, dtype=bool), -1)
    diff = (E != T) | (E.T != T.T)
    rev = diff & (E == T.T) & (E.T == T)
    return float((0.5 * rev + 1.0 * (diff & ~rev))[low].sum())


def shd(W_true, A_true, W_est, A_est) -> float:
    return pair_shd(W_true, W_est) + sum(pair_shd(a, b) for a, b in zip(A_true, A_est))


def f1(W_true, A_true, W_est, A_est) -> float:
    T = np.concatenate([W_true, *A_true]) != 0
    E = np.concatenate([W_est, *A_est]) != 0
    tp = (T & E).sum()
    fp = (~T & E).sum()
    fn = (T & ~E).sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def evaluate(W_true, A_true, W_est, A_est, fixed=(0.1,), max_thresholds=400):
    vals = np.abs(np.concatenate([W_est.ravel(), *[a.ravel() for a in A_est]]))
    vals = np.unique(vals[vals > 0])
    if len(vals) > max_thresholds:  # approximate oracle threshold on a quantile grid
        vals = np.quantile(vals, np.linspace(0, 1, max_thresholds))
    if len(vals) == 0:
        vals = np.array([0.0])
    best_t, best_s = vals.max(), np.inf
    for t in vals:
        s = shd(W_true, A_true, np.where(np.abs(W_est) >= t, W_est, 0),
                [np.where(np.abs(a) >= t, a, 0) for a in A_est])
        if s < best_s:
            best_t, best_s = t, s
    out = {"threshold_best": float(best_t)}
    for name, t in [("best", best_t)] + [(f"t{t}", t) for t in fixed]:
        We = np.where(np.abs(W_est) >= t, W_est, 0)
        Ae = [np.where(np.abs(a) >= t, a, 0) for a in A_est]
        out[f"shd_{name}"] = shd(W_true, A_true, We, Ae)
        out[f"f1_{name}"] = f1(W_true, A_true, We, Ae)
    out["n_true_edges"] = int((W_true != 0).sum() + sum((a != 0).sum() for a in A_true))
    return out
