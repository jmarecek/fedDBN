# ψDAG: Projected Stochastic Approximation Iteration for DAG Structure Learning

This repository contains the implementation of [**ψDAG**](https://arxiv.org/abs/2410.23862), a novel framework for learning Directed Acyclic Graphs (DAGs) using a Stochastic Approximation approach integrated with Stochastic Gradient Descent (SGD)-based optimization techniques. ψDAG introduces new projection methods tailored to efficiently enforce DAG constraints, ensuring convergence to a feasible local minimum with improved computational efficiency.

---

## Overview

Learning the structure of DAGs is a fundamental challenge in machine learning due to the vast combinatorial search space, which scales exponentially with the number of nodes. Traditional methods often face optimization difficulties due to the highly non-convex nature of DAG constraints and high per-iteration computational complexity. 

ψDAG addresses these challenges by:
- Employing a Stochastic Approximation approach combined with SGD-based optimization.
- Introducing projection methods that efficiently enforce DAG constraints.
- Achieving low iteration complexity, making it suitable for large-scale problems.

Comprehensive experimental evaluations demonstrate the effectiveness and scalability of ψDAG across various settings.

---

## Installation

To use ψDAG, clone this repository and install the required dependencies:

```bash
git clone https://github.com/kz29/-psiDAG.git
cd psidag
pip install -r requirements.txt
``` 
---

## fedDBN: federated dynamic Bayesian networks (extension)

`feddbn/` extends ψDAG to **dynamic Bayesian networks** (structural VARs
`X_t = X_t W + Σ_k X_{t-k} A_k + Z_t`, with `W` acyclic and the lag matrices `A_k`
unconstrained). It also adds a **federated** setting: clients hold disjoint parts of the time
series and run local preconditioned Nesterov steps (FedAvg-style), and the server averages the
models and applies ψDAG's order projection (`utils.projection_order1`), which needs no data.
With `K=1` it is a centralised "ψDBN".

```python
from feddbn import feddbn, FedDBNConfig
res = feddbn(X, [X_lag1], K=8, cfg=FedDBNConfig())   # X: (n, d) array, lags: list of (n, d) arrays
W, A = res.W, res.A                                    # threshold |W|, |A| afterwards
```

`experiments/` compares fedDBN with the exact MIP method
[ExDBN](https://github.com/pavelrt/ExDBN) on ExDBN's own benchmark. It estimates the *price of
federation* and breaks it down into the projection's order, incomplete optimisation, federation
and communication. See `experiments/README.md`. A TMLR-style draft is in `paper/`
(`cd paper && latexmk -pdf feddbn.tex`).

**Licence.** MIT (see `LICENSE`). The ψDAG code (`main.py`, `opt.py`, `utils.py`,
`data_generator.py` and the notebooks) is © 2024 Klea Ziu. The fedDBN additions
(`feddbn/`, `experiments/`, `paper/`) are © 2026 Jakub Marecek.

---
## BibTeX
If you find this work useful, please cite our paper:
```
@article{ziu2024psi,
  title={$$\backslash$psi $ DAG: Projected Stochastic Approximation Iteration for DAG Structure Learning},
  author={Ziu, Klea and Hanzely, Slavom{\'\i}r and Li, Loka and Zhang, Kun and Tak{\'a}{\v{c}}, Martin and Kamzolov, Dmitry},
  journal={arXiv preprint arXiv:2410.23862},
  year={2024}
}
```
