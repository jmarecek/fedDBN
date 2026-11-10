"""Evaluate ExDBN solutions (results/exdbn/*.npz) with ExDBN's metric code -> results/exdbn_main.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import exdbn_bridge as eb  # noqa: E402
from run_feddbn import meta  # noqa: E402

rows = []
for f in sorted((HERE / "results" / "exdbn").glob("*.npz")):
    R = np.load(f)
    D = np.load(HERE / "data" / f.name)
    row = dict(meta(f), method="ExDBN", K=1, time=float(R["time"]), build=float(R["build"]),
               failed=bool(R["failed"]))
    if not row["failed"]:
        row["gap"] = float(R["gap"])
        row.update(eb.evaluate(D["W_true"], D["A_true"], R["W"], list(R["A"])))
    rows.append(row)
pd.DataFrame(rows).to_csv(HERE / "results" / "exdbn_main.csv", index=False)
print(len(rows), "rows")
