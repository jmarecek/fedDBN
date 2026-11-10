# fedDBN experiments

All experiments compare **fedDBN** (this repository, `feddbn/`) with
**ExDBN** ([pavelrt/ExDBN](https://github.com/pavelrt/ExDBN)), using ExDBN's own data
generator, MIP solver and evaluation code through `exdbn_bridge.py`.

## Setup

```bash
git clone https://github.com/pavelrt/ExDBN.git /path/to/ExDBN
pip install -r requirements.txt gurobipy omegaconf ipython scikit-learn pandas igraph
export EXDBN_PATH=/path/to/ExDBN      # needed by every script that touches ExDBN
```

A Gurobi licence that is not size-limited is required for ExDBN (the pip licence is
restricted to 2000 variables).

## Pipeline

| step | command | output |
|---|---|---|
| ExDBN benchmark instances (ExDBN generator, grid ER3-p1 / SF3-p1 / ER2-p2, d = 5..25, n = 250, 1000, seeds 0-4) | `python experiments/gen_data.py --jobs 8` | `experiments/data/*.npz` |
| ExDBN (MIQP, Gurobi, 120 s, 2 threads) | `python experiments/run_exdbn.py --jobs 4 --threads 2` | `experiments/results/exdbn/*.npz` |
| ExDBN metrics | `python experiments/eval_exdbn.py` | `results/exdbn_main.csv` |
| fedDBN, K = 1, 2, 4, 8, plus the decomposition (oracle order, exact refit on the found order) | `python experiments/run_feddbn.py --part main` | `results/feddbn_main.csv` |
| communication budget and bounded staleness | `python experiments/run_feddbn.py --part delay` | `results/feddbn_delay.csv` |
| scaling instances (d up to 1000) | `python experiments/run_scaling.py --part data` | `experiments/data_scaling/` |
| scaling: fedDBN | `python experiments/run_scaling.py --part feddbn --jobs 3` | `results/scaling_feddbn.csv` |
| scaling: ExDBN, original and Gram form | `python experiments/run_scaling.py --part exdbn --jobs 2 --threads 4` | `results/scaling_exdbn.csv` |
| tables and figures for the paper | `python experiments/make_report.py` | `paper/tables/*.tex`, `paper/figures/*.pdf` |

## Notes

* fedDBN hyperparameters were chosen on tuning instances with seeds 100-104, which are
  disjoint from the evaluation seeds 0-4, and then frozen (`FedDBNConfig` defaults).
* `exdbn_bridge.solve_exdbn_gram` is ExDBN's MIQP written via the pooled Gram matrix
  (sufficient statistics). It uses the same variables, constraints and lazy cycle cuts. On a
  small instance it reaches the same optimum as `solve_milp.solve` within the 1e-4 MIP gap.
* `feddbn/metrics.py` re-implements ExDBN's SHD/F1 conventions in NumPy and matches ExDBN's
  code exactly on SHD. It is used where ExDBN's pure-Python metric is too slow (d > 25).
* ExDBN's generator gives up on 40 of the 150 grid points (series magnitude > 1e3 after 1000
  attempts, or ER3 with d = 5). These are skipped, as in ExDBN's own `run_experiments.py`.
