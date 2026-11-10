"""Generate (and cache) the ExDBN benchmark instances with ExDBN's own generator.

    EXDBN_PATH=/path/to/ExDBN python experiments/gen_data.py --jobs 6
"""
import argparse
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

GRID = dict(
    configs=["ER3-p1", "SF3-p1", "ER2-p2"],
    d=[5, 10, 15, 20, 25],
    n=[250, 1000],
    seeds=list(range(5)),
)


def fname(config, d, n, seed):
    return DATA / f"{config}_d{d}_n{n}_s{seed}.npz"


def job(args):
    config, d, n, seed = args
    f = fname(config, d, n, seed)
    if f.exists():
        return (args, "cached", 0.0)
    import exdbn_bridge as eb
    t0 = time.time()
    try:
        D = eb.generate(config, d, n, seed)
    except eb.ExDagDataException as e:  # ExDBN skips these too (e.g. ER3 with d=5)
        return (args, f"skip: {str(e)[:80]}", time.time() - t0)
    np.savez(f, **D)
    return (args, "ok", time.time() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    tasks = list(itertools.product(GRID["configs"], GRID["d"], GRID["n"], GRID["seeds"]))
    # biggest first for load balancing
    tasks.sort(key=lambda t: -t[1] * t[2])
    log = {}
    with Pool(a.jobs) as pool:
        for args, status, dt in pool.imap_unordered(job, tasks):
            print(args, status, f"{dt:.1f}s", flush=True)
            log["_".join(map(str, args))] = status
    (DATA / "generation_log.json").write_text(json.dumps(log, indent=1))
