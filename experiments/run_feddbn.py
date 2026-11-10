"""fedDBN on the cached ExDBN instances, including the decomposition of the price of federation.

For every instance we record
  oracle      exact least squares on the true topological order (no order error, no optimisation error)
  order-K     exact least squares on the order found by fedDBN with K clients (order error only)
  fedDBN-K    the fedDBN iterate itself (order + optimisation + federation); K=1 is centralised psiDBN
and, on a subset, the effect of the communication budget and of bounded-delay asynchrony.

    EXDBN_PATH=... python experiments/run_feddbn.py --part main --jobs 2
"""
import argparse
import itertools
import os
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
DATA = HERE / "data"
OUT = HERE / "results"

import torch  # noqa: E402

torch.set_num_threads(1)

from feddbn import FedDBNConfig, feddbn, ols_on_order  # noqa: E402

KS = [1, 2, 4, 8]


def meta(f):
    cfg, d, n, s = f.stem.split("_")
    return dict(config=cfg, d=int(d[1:]), n=int(n[1:]), seed=int(s[1:]))


def split(Th, d, p):
    return Th[:d], [Th[d + k * d: d + (k + 1) * d] for k in range(p)]


def main_job(f):
    import networkx as nx
    import exdbn_bridge as eb
    D = np.load(f)
    X, Y, Wt, At, p = D["X"], list(D["Y"]), D["W_true"], D["A_true"], int(D["p"])
    d = X.shape[1]
    rows = []
    base = meta(f)
    true_order = list(nx.topological_sort(nx.DiGraph(Wt)))
    Th = ols_on_order(X, Y, true_order)
    rows.append(dict(base, method="oracle", K=0, **eb.evaluate(Wt, At, *split(Th, d, p))))
    cfg = FedDBNConfig()
    for K in KS:
        t0 = time.perf_counter()
        r = feddbn(X, Y, K=K, cfg=cfg)
        wall = time.perf_counter() - t0
        info = dict(time_parallel=r.time_parallel, time_sequential=r.time_sequential, wall=wall,
                    rounds=r.rounds, floats=r.floats_communicated, loss=min(r.loss_history),
                    n_orders=len({tuple(h["order"]) for h in r.history}))
        rows.append(dict(base, method="fedDBN", K=K, **info, **eb.evaluate(Wt, At, r.W, r.A)))
        Th = ols_on_order(X, Y, r.order)
        rows.append(dict(base, method="order", K=K, **eb.evaluate(Wt, At, *split(Th, d, p))))
    return rows


def delay_job(args):
    f, K, S, epochs = args
    import exdbn_bridge as eb
    D = np.load(f)
    X, Y, Wt, At = D["X"], list(D["Y"]), D["W_true"], D["A_true"]
    cfg = replace(FedDBNConfig(), max_staleness=S, epochs=epochs)
    r = feddbn(X, Y, K=K, cfg=cfg)
    return [dict(meta(f), method="fedDBN", K=K, staleness=S, epochs=epochs, rounds=r.rounds,
                 time_parallel=r.time_parallel, loss=min(r.loss_history),
                 **eb.evaluate(Wt, At, r.W, r.A))]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["main", "delay"], default="main")
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    files = sorted(f for f in DATA.glob("*.npz"))
    if a.part == "main":
        tasks, fn, out = files, main_job, OUT / "feddbn_main.csv"
    else:
        sub = [f for f in files if meta(f)["n"] == 1000 and meta(f)["d"] in (10, 20)
               and meta(f)["config"] in ("ER3-p1", "SF3-p1")]
        tasks = [(f, 4, S, 30) for f in sub for S in (0, 1, 2, 4, 8)]
        tasks += [(f, 4, 0, e) for f in sub for e in (1, 2, 5, 10)]
        fn, out = delay_job, OUT / "feddbn_delay.csv"
    rows = []
    with Pool(a.jobs) as pool:
        for i, rs in enumerate(pool.imap_unordered(fn, tasks)):
            rows.extend(rs)
            if i % 10 == 0:
                print(f"{i + 1}/{len(tasks)}", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)
