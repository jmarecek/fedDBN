"""fedDBN: federated learning of linear dynamic Bayesian networks.

The model is the structural vector autoregression used by DYNOTEARS and ExDBN,

    X_t = X_t W + sum_{k=1}^p X_{t-k} A_k + Z_t,

where W (intra-slice, d x d) must be acyclic and A_1..A_p (inter-slice) are
unconstrained.  Stacking Z = [X, Y_1, ..., Y_p] and Theta = [W; A_1; ...; A_p]
gives the least-squares loss f(Theta) = 1/(2n) ||X - Z Theta||_F^2.

fedDBN follows the psiDAG scheme (Ziu et al., 2024): stochastic first-order
steps on the loss, alternated with a projection of W onto the set of DAGs
that are consistent with a topological order.  The projection
(`projection_order1` from psiDAG's utils.py) only needs the current
parameter and no data, so in the federated version it runs on the server.
Clients hold disjoint parts of the time series and only exchange Theta.
"""
from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]


def _load_psidag_utils():
    # psiDAG's helpers live in the top-level ``utils.py``.  ExDBN (the baseline)
    # also ships a top-level ``utils`` module, so we load psiDAG's under a
    # private name to let both coexist in one process.
    spec = importlib.util.spec_from_file_location("psidag_utils", _ROOT / "utils.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_psidag = _load_psidag_utils()
projection_order = _psidag.projection_order1


def order_mask(order: torch.Tensor, d: int) -> torch.Tensor:
    """Boolean d x d mask with mask[i, j] = True iff i precedes j in `order`.

    Vectorised equivalent of psiDAG's ``mask_from_order`` (that one is an O(d^2)
    Python loop, too slow for d in the thousands).
    """
    rank = torch.empty(d, dtype=torch.long)
    rank[order.long()] = torch.arange(d)
    return rank[:, None] < rank[None, :]


@dataclass
class FedDBNConfig:
    # Defaults were selected on held-out instances (seeds 100-104), see experiments/README.md.
    epochs: int = 30                 # psiDAG outer iterations (one projection each)
    rounds_unconstrained: int = 3    # communication rounds before each projection
    rounds_constrained: int = 10     # communication rounds after each projection
    local_steps: int = 100           # local steps per round (FedAvg)
    batch_size: Optional[int] = None # None = exact local gradient, else minibatch SGD
    lr: float = 1.0                  # step = lr / L_k (L_k: local preconditioned smoothness)
    l1: float = 0.0                  # optional l1 penalty (proximal step)
    precondition: bool = True        # Jacobi preconditioning by local second moments
    momentum: float = 0.95           # Nesterov momentum for local steps (reset every round)
    max_staleness: int = 0           # bounded-delay asynchrony: client k starts from the model
                                     # broadcast s_k ~ U{0..max_staleness} rounds ago
    seed: int = 0


class Client:
    """A data holder.  Raw data never leaves this object."""

    def __init__(self, X: np.ndarray, Y: Sequence[np.ndarray], precondition: bool = True):
        self.X = torch.as_tensor(np.asarray(X), dtype=torch.float64)
        lags = [torch.as_tensor(np.asarray(y), dtype=torch.float64) for y in Y]
        self.Z = torch.cat([self.X, *lags], dim=1)
        self.n = self.X.shape[0]
        m = self.Z.shape[1]
        h = (self.Z ** 2).mean(0).clamp_min(1e-12)
        self.P = (1.0 / h) if precondition else torch.ones(m, dtype=torch.float64)
        self.L = self._smoothness()
        self._ZtZ = None

    def _smoothness(self, iters: int = 100) -> float:
        """lambda_max(P^1/2 Z^T Z P^1/2 / n) by power iteration (O(n m) per step)."""
        s = self.P.sqrt()
        g = torch.Generator().manual_seed(12345)
        v = torch.randn(self.Z.shape[1], generator=g, dtype=torch.float64)
        v /= v.norm()
        lam = 0.0
        for _ in range(iters):
            w = s * (self.Z.T @ (self.Z @ (s * v))) / self.n
            lam = float(w.norm())
            v = w / lam
        return 1.05 * lam

    def loss(self, Theta: torch.Tensor) -> float:
        R = self.X - self.Z @ Theta
        return 0.5 * float((R ** 2).sum()) / self.n

    def local_update(self, Theta, mask, steps, batch_size, lr, l1, gen, momentum=0.0):
        Th = Theta.clone()
        Th_prev = Th
        step = (lr / self.L) * self.P[:, None]
        full = batch_size is None or batch_size >= self.n
        if full and self._ZtZ is None:
            self._ZtZ = self.Z.T @ self.Z / self.n
            self._ZtX = self.Z.T @ self.X / self.n
        for _ in range(steps):
            V = Th + momentum * (Th - Th_prev) if momentum > 0 else Th
            if full:
                G = self._ZtZ @ V - self._ZtX
            else:
                idx = torch.randint(self.n, (batch_size,), generator=gen)
                Zb = self.Z[idx]
                G = Zb.T @ (Zb @ V - self.X[idx]) / batch_size
            Th_prev = Th
            Th = V - step * G
            if l1 > 0:
                Th = torch.sign(Th) * torch.clamp(Th.abs() - step * l1, min=0.0)
            Th = Th * mask
        return Th


@dataclass
class FedDBNResult:
    W: np.ndarray
    A: List[np.ndarray]
    order: np.ndarray
    loss_history: List[float]
    rounds: int
    floats_communicated: int
    time_sequential: float   # all clients simulated one after another
    time_parallel: float     # sum over rounds of (slowest client + server)
    time_setup: float        # per-client preprocessing (max over clients)
    history: List[dict] = field(default_factory=list)


def fit_feddbn(clients: Sequence[Client], d: int, p: int, cfg: FedDBNConfig) -> FedDBNResult:
    torch.manual_seed(cfg.seed)
    gens = [torch.Generator().manual_seed(cfg.seed * 1000 + k) for k in range(len(clients))]
    m = d + p * d
    N = sum(c.n for c in clients)
    w = [c.n / N for c in clients]

    M0 = torch.ones(m, d, dtype=torch.float64)
    M0[:d, :d] -= torch.eye(d, dtype=torch.float64)
    Theta = torch.zeros(m, d, dtype=torch.float64)

    t_seq = t_par = 0.0
    rounds = 0
    rng = np.random.default_rng(cfg.seed)
    broadcasts = []  # last (Theta, mask) pairs sent by the server, for the asynchronous variant

    def fed_round(Theta, mask):
        nonlocal t_seq, t_par, rounds
        broadcasts.append((Theta, mask))
        del broadcasts[:-(cfg.max_staleness + 1)]
        new = torch.zeros_like(Theta)
        slowest = 0.0
        for k, c in enumerate(clients):
            s = int(rng.integers(0, len(broadcasts))) if cfg.max_staleness > 0 else 0
            base, base_mask = broadcasts[-1 - s]
            t0 = time.perf_counter()
            Th_k = c.local_update(base, base_mask, cfg.local_steps, cfg.batch_size, cfg.lr, cfg.l1,
                                  gens[k], cfg.momentum)
            dt = time.perf_counter() - t0
            t_seq += dt
            slowest = max(slowest, dt)
            new += w[k] * Th_k
        t_par += slowest
        rounds += 1
        return new * mask  # stale updates may violate the current order; re-project

    best = (np.inf, None, None)
    loss_history, history = [], []
    for epoch in range(cfg.epochs):
        for _ in range(cfg.rounds_unconstrained):
            Theta = fed_round(Theta, M0)
        t0 = time.perf_counter()
        order = projection_order(Theta[:d].detach().clone())
        M = M0.clone()
        M[:d] = order_mask(order, d).to(torch.float64)
        Theta = Theta * M
        dt = time.perf_counter() - t0
        t_seq += dt
        t_par += dt
        for _ in range(cfg.rounds_constrained):
            Theta = fed_round(Theta, M)
        # Each client reports one scalar (its local loss); the server keeps the best epoch.
        loss = sum(wk * c.loss(Theta) for wk, c in zip(w, clients))
        loss_history.append(loss)
        history.append(dict(epoch=epoch, loss=loss, rounds=rounds, time_parallel=t_par,
                            order=order.numpy().copy()))
        if loss < best[0]:
            best = (loss, Theta.clone(), order.clone())

    _, Theta, order = best
    Th = Theta.numpy()
    W = Th[:d].copy()
    A = [Th[d + k * d: d + (k + 1) * d].copy() for k in range(p)]
    floats = 2 * len(clients) * rounds * m * d
    return FedDBNResult(W=W, A=A, order=order.numpy(), loss_history=loss_history, rounds=rounds,
                        floats_communicated=floats, time_sequential=t_seq, time_parallel=t_par,
                        time_setup=0.0, history=history)


def split_rows(X: np.ndarray, Y: Sequence[np.ndarray], K: int):
    """Split a (lagged) time series into K contiguous, disjoint time windows."""
    n = X.shape[0]
    bounds = np.linspace(0, n, K + 1).astype(int)
    return [(X[a:b], [y[a:b] for y in Y]) for a, b in zip(bounds[:-1], bounds[1:])]


def feddbn(X: np.ndarray, Y: Sequence[np.ndarray], K: int = 1, cfg: Optional[FedDBNConfig] = None,
           parts=None) -> FedDBNResult:
    """Convenience wrapper: split (X, Y) over K clients and run fedDBN.  K=1 is centralised psiDBN."""
    cfg = cfg or FedDBNConfig()
    parts = parts if parts is not None else split_rows(X, Y, K)
    t0 = time.perf_counter()
    setup = []
    clients = []
    for Xk, Yk in parts:
        s = time.perf_counter()
        clients.append(Client(Xk, Yk, precondition=cfg.precondition))
        setup.append(time.perf_counter() - s)
    res = fit_feddbn(clients, X.shape[1], len(Y), cfg)
    res.time_setup = max(setup)
    res.time_sequential += sum(setup)
    res.time_parallel += max(setup)
    return res


def ols_on_order(X: np.ndarray, Y: Sequence[np.ndarray], order) -> np.ndarray:
    """Exact (centralised) least squares restricted to a topological order.

    Analysis tool only: it removes the optimisation error of fedDBN, so that
    SHD(ols_on_order(order_found)) - SHD(ols_on_order(true_order)) isolates the
    error caused by the projection's order.
    """
    d = X.shape[1]
    p = len(Y)
    Z = np.concatenate([X, *Y], axis=1)
    Th = np.zeros((d + p * d, d))
    pos = np.empty(d, dtype=int)
    pos[np.asarray(order, dtype=int)] = np.arange(d)
    lags = list(range(d, d + p * d))
    for j in range(d):
        cols = [i for i in range(d) if pos[i] < pos[j]] + lags
        Th[cols, j] = np.linalg.lstsq(Z[:, cols], X[:, j], rcond=None)[0]
    return Th
