"""Scalability: fedDBN vs ExDBN (original formulation) vs ExDBN in sufficient-statistics form.

    python experiments/run_scaling.py --part data
    python experiments/run_scaling.py --part feddbn --jobs 3
    EXDBN_PATH=... python experiments/run_scaling.py --part exdbn --jobs 2 --threads 4
"""
import argparse
import itertools
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
DATA = HERE / "data_scaling"
OUT = HERE / "results"

import torch  # noqa: E402

torch.set_num_threads(1)

from feddbn import FedDBNConfig, feddbn, metrics  # noqa: E402
from feddbn.data import simulate_dbn  # noqa: E402

DS = [10, 25, 50, 100, 250, 500, 1000]
SEEDS = [0, 1, 2]
CFG = FedDBNConfig(epochs=10, local_steps=20)           # 130 communication rounds
EXDBN_GRAM_DS = [10, 25, 50, 100, 250]


def n_of(d):
    return max(1000, 4 * d)   # n > number of regressors (d + p d) for every d


def path(d, s):
    return DATA / f"ER2-p1_d{d}_n{n_of(d)}_s{s}.npz"


def load(d, s):
    D = np.load(path(d, s))
    return D["X"], list(D["Y"]), D["W_true"], list(D["A_true"])


def feddbn_job(args):
    d, s, K = args
    X, Y, Wt, At = load(d, s)
    t0 = time.perf_counter()
    r = feddbn(X, Y, K=K, cfg=CFG)
    wall = time.perf_counter() - t0
    ev = metrics.evaluate(Wt, At, r.W, r.A)
    return dict(method=f"fedDBN", K=K, d=d, n=X.shape[0], seed=s, time_parallel=r.time_parallel,
                time_sequential=r.time_sequential, wall=wall, rounds=r.rounds,
                floats=r.floats_communicated, **ev)


def exdbn_job(args):
    variant, d, s, tl, threads = args
    import exdbn_bridge as eb
    X, Y, Wt, At = load(d, s)
    if variant == "orig":
        r = eb.solve_exdbn(X, Y, tl, threads)
        r = r or dict(failed=True)
        r.setdefault("failed", False)
    else:
        r = eb.solve_exdbn_gram(X, Y, tl, threads)
    row = dict(method=f"ExDBN-{variant}", K=1, d=d, n=X.shape[0], seed=s, time_limit=tl,
               threads=threads, failed=r["failed"], time=r.get("time"), build=r.get("build"),
               gap=r.get("gap"), n_bin=r.get("n_bin"))
    if not r["failed"]:
        row.update(metrics.evaluate(Wt, At, r["W"], list(r["A"])))
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["data", "feddbn", "exdbn"], required=True)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--time-limit", type=float, default=120)
    # the original ExDBN formulation needs O(n d m^2) memory to build: run d=50 separately, one job
    ap.add_argument("--orig-ds", default="10,25")
    ap.add_argument("--gram-ds", default=",".join(map(str, EXDBN_GRAM_DS)))
    ap.add_argument("--seeds", type=lambda x: [int(v) for v in x.split(",")], default=SEEDS)
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.part == "data":
        DATA.mkdir(exist_ok=True)
        for d, s in itertools.product(DS, SEEDS):
            if not path(d, s).exists():
                D = simulate_dbn(d, n_of(d), p=1, intra_degree=4, seed=s)
                np.savez(path(d, s), **D)
                print(d, s, "rho=%.2f" % D["rho"], flush=True)
        sys.exit()
    if a.part == "feddbn":
        tasks = [(d, s, K) for d in DS for s in SEEDS for K in (1, 8)
                 if not (K == 8 and d == 1000 and s > 0)]
        fn, out = feddbn_job, OUT / "scaling_feddbn.csv"
    else:
        orig_ds = [int(x) for x in a.orig_ds.split(",") if x]
        gram_ds = [int(x) for x in a.gram_ds.split(",") if x]
        tasks = [("orig", d, s, a.time_limit, a.threads) for d in orig_ds for s in a.seeds]
        tasks += [("gram", d, s, a.time_limit, a.threads) for d in gram_ds for s in a.seeds]
        fn, out = exdbn_job, OUT / f"scaling_exdbn{a.suffix}.csv"
    tasks.sort(key=lambda t: -t[1] if a.part == "exdbn" else -t[0])
    rows = []
    with Pool(a.jobs, maxtasksperchild=1) as pool:
        for row in pool.imap_unordered(fn, tasks):
            rows.append(row)
            print({k: row.get(k) for k in ("method", "K", "d", "seed", "shd_best", "f1_best",
                                           "time_parallel", "time", "gap")}, flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)
