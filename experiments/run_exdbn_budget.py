"""Robustness check: does ExDBN at d = 20, 25 improve with more threads and time?

Gram form (no model-building overhead) on four ExDBN benchmark instances.
    EXDBN_PATH=... python experiments/run_exdbn_budget.py 300 6
    EXDBN_PATH=... python experiments/run_exdbn_budget.py 120 2
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
import exdbn_bridge as eb  # noqa: E402

INSTANCES = ["ER3-p1_d20_n1000_s1", "ER3-p1_d25_n1000_s1", "SF3-p1_d20_n1000_s1", "SF3-p1_d25_n1000_s1"]
TL, TH = (float(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (300.0, 6)
OUTF = HERE / "results" / f"exdbn_budget_{int(TL)}s_{TH}t.csv"
rows = []
for name in INSTANCES:
    D = np.load(HERE / "data" / f"{name}.npz")
    for tl, th in [(TL, TH)]:
        r = eb.solve_exdbn_gram(D["X"], list(D["Y"]), tl, th)
        row = dict(instance=name, time_limit=tl, threads=th, gap=r.get("gap"), time=r["time"],
                   build=r["build"])
        if not r["failed"]:
            row.update(eb.evaluate(D["W_true"], D["A_true"], r["W"], list(r["A"])))
        rows.append(row)
        print({k: row.get(k) for k in ("instance", "gap", "shd_best", "f1_best", "time")}, flush=True)
        pd.DataFrame(rows).to_csv(OUTF, index=False)
