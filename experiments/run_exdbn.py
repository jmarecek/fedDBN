"""Run ExDBN (Gurobi MIQP, branch-and-cut with lazy cycle constraints) on cached instances.

    EXDBN_PATH=... python experiments/run_exdbn.py --time-limit 120 --threads 2 --jobs 4
"""
import argparse
import contextlib
import io
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "results" / "exdbn"


class _Stamp(io.StringIO):
    """stdout sink remembering when ExDBN first prints (right after building the model)."""

    def __init__(self):
        super().__init__()
        self.first = None

    def write(self, s):
        if self.first is None and s.strip():
            self.first = time.perf_counter()
        return super().write(s)


def job(args):
    f, time_limit, threads = args
    out = OUT / (f.stem + ".npz")
    if out.exists():
        return f.stem, "cached"
    import gurobipy as gp
    import exdbn_bridge  # noqa: F401  (puts ExDBN on sys.path)
    import solve_milp
    D = np.load(f)
    gp.setParam("OutputFlag", 0)
    gp.setParam("Threads", threads)
    sink = _Stamp()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(sink):
        W, A, gap, lazy, _ = solve_milp.solve(D["X"], 0.1, 0.1, 0, Y=list(D["Y"]), B_ref=None,
                                              tabu_edges=None, time_limit=time_limit)
    total = time.perf_counter() - t0
    build = (sink.first - t0) if sink.first else np.nan
    if W is None:
        np.savez(out, failed=True, time=total, build=build)
        return f.stem, "no incumbent"
    np.savez(out, W=W, A=np.stack(A), gap=gap, lazy=lazy, time=total, build=build, failed=False)
    return f.stem, f"gap={gap:.3f} t={total:.0f}s build={build:.1f}s"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=120)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--pattern", default="*.npz")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA.glob(a.pattern), key=lambda p: -p.stat().st_size)
    (OUT / "settings.json").write_text(json.dumps(vars(a)))
    with Pool(a.jobs, maxtasksperchild=1) as pool:
        for stem, status in pool.imap_unordered(job, [(f, a.time_limit, a.threads) for f in files]):
            print(stem, status, flush=True)
