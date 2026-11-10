"""Tables (LaTeX) and figures (PDF) for the paper from experiments/results/*.csv."""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
PAPER = HERE.parent / "paper"
TAB = PAPER / "tables"
FIG = PAPER / "figures"
TAB.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

# Reference categorical palette (light mode), fixed slot order: blue, orange, aqua, yellow.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5, "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "lines.markersize": 5,
    "legend.frameon": False, "figure.dpi": 150,
})
CONFIG_ORDER = ["ER3-p1", "SF3-p1", "ER2-p2"]


def fmt(x, nd=1):
    return "--" if pd.isna(x) else f"{x:.{nd}f}"


def tfmt(x):
    return "--" if pd.isna(x) else (f"{x:.2f}" if x < 10 else f"{x:.0f}")


def load_main():
    ex = pd.read_csv(RES / "exdbn_main.csv")
    fd = pd.read_csv(RES / "feddbn_main.csv")
    return ex, fd


def per_instance(ex, fd):
    key = ["config", "d", "n", "seed"]
    E = ex[~ex.failed][key + ["shd_best", "f1_best", "gap", "time", "n_true_edges"]].rename(
        columns={"shd_best": "ex", "f1_best": "ex_f1", "time": "ex_time"})
    o = fd[fd.method == "oracle"][key + ["shd_best", "f1_best"]].rename(
        columns={"shd_best": "oracle", "f1_best": "oracle_f1"})
    df = E.merge(o, on=key)
    for K in (1, 2, 4, 8):
        f = fd[(fd.method == "fedDBN") & (fd.K == K)][key + ["shd_best", "f1_best", "time_parallel",
                                                             "rounds", "floats"]]
        f = f.rename(columns={"shd_best": f"fed{K}", "f1_best": f"fed{K}_f1",
                              "time_parallel": f"t{K}", "rounds": f"R{K}", "floats": f"fl{K}"})
        r = fd[(fd.method == "order") & (fd.K == K)][key + ["shd_best"]].rename(
            columns={"shd_best": f"ord{K}"})
        df = df.merge(f, on=key).merge(r, on=key)
    return df


def table_main(df):
    g = df.groupby(["config", "d", "n"])
    lines = [r"\begin{tabular}{llr rrr rrrrr r}", r"\toprule",
             r" & & & \multicolumn{3}{c}{\exdbn{} (pooled, 120\,s)} & \multicolumn{5}{c}{SHD of \feddbn{} with $K$ clients} & \\",
             r"\cmidrule(lr){4-6}\cmidrule(lr){7-11}",
             r"config & $d$ & $n$ & SHD & F1 & gap & $K{=}1$ & 2 & 4 & 8 & F1$_{K=8}$ & edges \\",
             r"\midrule"]
    last = None
    for (cfg, d, n), x in sorted(g, key=lambda t: (CONFIG_ORDER.index(t[0][0]), t[0][1], t[0][2])):
        if last is not None and cfg != last:
            lines.append(r"\midrule")
        lab = cfg if cfg != last else ""
        last = cfg
        lines.append(f"{lab} & {d} & {n} & {fmt(x.ex.mean())} & {fmt(x.ex_f1.mean(), 2)} & "
                     f"{fmt(x.gap.mean(), 2)} & {fmt(x.fed1.mean())} & {fmt(x.fed2.mean())} & "
                     f"{fmt(x.fed4.mean())} & {fmt(x.fed8.mean())} & {fmt(x.fed8_f1.mean(), 2)} & "
                     f"{fmt(x.n_true_edges.mean(), 0)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "main.tex").write_text("\n".join(lines) + "\n")


def table_decomp(df, K=8):
    df = df.copy()
    df["d_est"] = df.oracle - df.ex
    df["d_order"] = df.ord1 - df.oracle
    df["d_opt"] = df.fed1 - df.ord1
    df["d_fed"] = df[f"fed{K}"] - df.fed1
    df["d_fed_order"] = df[f"ord{K}"] - df.ord1
    df["pof"] = df[f"fed{K}"] - df.ex
    df["pof_rel"] = df.pof / df.n_true_edges
    lines = [r"\begin{tabular}{llr r rrrr r r}", r"\toprule",
             r"config & $d$ & inst. & SHD$_{\mathrm{MIP}}$ & $\Delta_{\mathrm{est}}$ & $\Delta_{\mathrm{order}}$ & $\Delta_{\mathrm{opt}}$ & $\Delta_{\mathrm{fed}}(8)$ & PoF$(8)$ & PoF/edges \\",
             r"\midrule"]
    last = None
    g = df.groupby(["config", "d"])
    for (cfg, d), x in sorted(g, key=lambda t: (CONFIG_ORDER.index(t[0][0]), t[0][1])):
        if last is not None and cfg != last:
            lines.append(r"\midrule")
        lab = cfg if cfg != last else ""
        last = cfg
        lines.append(f"{lab} & {d} & {len(x)} & {fmt(x.ex.mean())} & {fmt(x.d_est.mean())} & "
                     f"{fmt(x.d_order.mean())} & {fmt(x.d_opt.mean())} & {fmt(x.d_fed.mean())} & "
                     f"{fmt(x.pof.mean())} & {fmt(100 * x.pof_rel.mean(), 0)}\\% \\\\")
    lines.append(r"\midrule")
    lines.append(f"all & & {len(df)} & {fmt(df.ex.mean())} & {fmt(df.d_est.mean())} & "
                 f"{fmt(df.d_order.mean())} & {fmt(df.d_opt.mean())} & {fmt(df.d_fed.mean())} & "
                 f"{fmt(df.pof.mean())} & {fmt(100 * df.pof_rel.mean(), 0)}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "decomp.tex").write_text("\n".join(lines) + "\n")
    summ = df[["d_est", "d_order", "d_opt", "d_fed", "d_fed_order", "pof", "pof_rel"]].describe()
    summ.to_csv(RES / "decomp_summary.csv")
    by_d = df.groupby("d")[["ex", "oracle", "ord1", "fed1", f"fed{K}", "d_est", "d_order", "d_opt",
                            "d_fed", "pof", "pof_rel", "gap"]].mean()
    by_d.to_csv(RES / "decomp_by_d.csv")
    byK = {k: (df[f"fed{k}"] - df.fed1).mean() for k in (2, 4, 8)}
    byK_order = {k: (df[f"ord{k}"] - df.ord1).mean() for k in (2, 4, 8)}
    pd.DataFrame(dict(d_fed=byK, d_fed_order=byK_order)).to_csv(RES / "dfed_by_K.csv")
    return df


def table_comm(df, scal):
    """Wall-clock model (eq. wallclock) for representative instances and networks."""
    nets = [("datacentre", 1e-4, 10e9 / 8), ("LAN", 1e-3, 1e9 / 8), ("WAN", 50e-3, 100e6 / 8),
            ("cross-continent", 150e-3, 20e6 / 8)]
    rows = []
    x = df[(df.d == 25)]
    if len(x):
        rows.append(("ExDBN bench., $d{=}25$, $K{=}8$", 25, 1, x.R8.mean(), x.t8.mean()))
    for d in (100, 1000):
        s = scal[(scal.method == "fedDBN") & (scal.d == d) & (scal.K == 8)]
        if len(s):
            rows.append((f"scaling, $d{{=}}{d}$, $K{{=}}8$", d, 1, s.rounds.mean(), s.time_parallel.mean()))
    lines = [r"\begin{tabular}{l r r " + "r" * len(nets) + "}", r"\toprule",
             r"instance & rounds & compute [s] & " + " & ".join(n for n, _, _ in nets) + r" \\",
             r"\midrule"]
    for name, d, p, R, tc in rows:
        m = d * (p + 1)
        tot = [tc + R * (2 * lat + 2 * 8 * m * d / bw) for _, lat, bw in nets]
        lines.append(f"{name} & {R:.0f} & {tc:.1f} & " + " & ".join(f"{t:.1f}" for t in tot) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "comm.tex").write_text("\n".join(lines) + "\n")


def fig_delay():
    f = RES / "feddbn_delay.csv"
    if not f.exists():
        return
    dl = pd.read_csv(f)
    dl["rel"] = dl.shd_best / dl.n_true_edges
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.3), sharey=True)
    for ax, (col, sub, xl) in zip(axes, [("staleness", dl[dl.epochs == 30], "max. staleness $S$ [rounds]"),
                                         ("rounds", dl[dl.staleness == 0], "communication rounds $R$")]):
        for i, (d, s) in enumerate(sub.groupby("d")):
            m = s.groupby(col).rel.agg(["mean", "sem"]).reset_index()
            ax.plot(m[col], m["mean"], marker="o", color=C[i], label=f"$d={d}$")
            ax.fill_between(m[col], m["mean"] - m["sem"], m["mean"] + m["sem"], color=C[i], alpha=0.15,
                            linewidth=0)
            ax.annotate(f"$d={d}$", (m[col].iloc[-1], m["mean"].iloc[-1]), xytext=(4, 0),
                        textcoords="offset points", color=INK2, va="center", fontsize=8)
        ax.set_xlabel(xl)
        ax.set_title("(a) bounded-delay asynchrony, $R=390$" if col == "staleness"
                     else "(b) communication budget, synchronous", fontsize=8.5, color=INK, loc="left")
        if col == "rounds":
            ax.set_xscale("log")
    axes[0].set_ylabel("SHD / true edges")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "delay.pdf")
    plt.close(fig)


def fig_scaling(scal):
    s = scal.copy()
    s["rel"] = s.shd_best / s.n_true_edges
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.75))
    series = [("fedDBN, $K=1$", (s.method == "fedDBN") & (s.K == 1), "time_parallel", C[0]),
              ("fedDBN, $K=8$ (parallel)", (s.method == "fedDBN") & (s.K == 8), "time_parallel", C[1]),
              ("ExDBN (Gram form)", (s.method == "ExDBN-gram") & (~s.failed.eq(True)), "time", C[2]),
              ("ExDBN (original)", (s.method == "ExDBN-orig") & (~s.failed.eq(True)), "time", C[3])]
    for name, sel, tcol, col in series:
        x = s[sel]
        if not len(x):
            continue
        m = x.groupby("d").agg(t=(tcol, "mean"), r=("rel", "mean"), f1=("f1_best", "mean")).reset_index()
        for ax, ycol in ((axes[0], "t"), (axes[1], "r")):
            ax.plot(m.d, m[ycol], color=col, label=name)
            ok, empty = m[m.f1 >= 0.05], m[m.f1 < 0.05]
            ax.plot(ok.d, ok[ycol], "o", color=col, markersize=4.5)
            ax.plot(empty.d, empty[ycol], "o", color=col, markersize=4.5, markerfacecolor="white")
        if (m.f1 < 0.05).any():
            e = m[m.f1 < 0.05]
            axes[1].annotate("MIP returns\nthe empty graph", (e.d.iloc[0], e.r.iloc[0]), xytext=(-6, -4),
                             textcoords="offset points", ha="right", va="top", color=INK2, fontsize=7)
    tl = scal.time_limit.dropna()
    if len(tl):
        axes[0].axhline(tl.iloc[0], color=MUTED, linewidth=1, linestyle="--")
        axes[0].annotate("MIP time limit", (350, tl.iloc[0]), xytext=(0, -4), textcoords="offset points",
                         ha="center", va="top", color=INK2, fontsize=7.5)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("number of variables $d$")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("wall-clock time [s]")
    axes[1].set_ylabel("SHD / true edges")
    axes[1].set_ylim(0, None)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / "scaling.pdf")
    plt.close(fig)


def table_scaling(scal):
    s = scal.copy()
    lines = [r"\begin{tabular}{rr rrr rrr rrrr rr}", r"\toprule",
             r" & & \multicolumn{3}{c}{\feddbn, $K{=}1$} & \multicolumn{3}{c}{\feddbn, $K{=}8$} & \multicolumn{4}{c}{\exdbn{}, Gram form} & \multicolumn{2}{c}{\exdbn{}, orig.} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-12}\cmidrule(lr){13-14}",
             r"$d$ & edges & SHD & F1 & time [s] & SHD & F1 & time [s] & SHD & F1 & gap & build [s] & SHD & time [s] \\", r"\midrule"]
    for d in sorted(s.d.unique()):
        a = s[(s.method == "fedDBN") & (s.K == 1) & (s.d == d)]
        b = s[(s.method == "fedDBN") & (s.K == 8) & (s.d == d)]
        g = s[(s.method == "ExDBN-gram") & (s.d == d)]
        gok = g[~g.failed.eq(True)]
        o = s[(s.method == "ExDBN-orig") & (s.d == d)]
        if not len(o):
            orig_s = "-- & --"
        elif o.failed.eq(True).any():
            orig_s = r"\multicolumn{2}{c}{failed}"
        else:   # dagger: fewer seeds than the other columns
            dag = r"$^\dagger$" if len(o) < len(a) else ""
            orig_s = f"{fmt(o.shd_best.mean())}{dag} & {fmt(o.time.mean(), 0)}{dag}"
        gap = gok.gap.mean()
        gap_s = r"$\gg 1$" if gap > 1 else fmt(gap, 2)   # only the empty-graph incumbent was found
        gcols = (f"{fmt(gok.shd_best.mean())} & {fmt(gok.f1_best.mean(), 2)} & {gap_s} & "
                 f"{tfmt(g.build.mean())}") if len(g) else r"\multicolumn{4}{c}{not attempted}"
        if len(g) and not len(gok):
            gcols = r"\multicolumn{3}{c}{no incumbent} & " + fmt(g.build.mean())
        lines.append(f"{d} & {fmt(a.n_true_edges.mean(), 0)} & {fmt(a.shd_best.mean())} & {fmt(a.f1_best.mean(), 2)} & {tfmt(a.time_parallel.mean())} & "
                     f"{fmt(b.shd_best.mean())} & {fmt(b.f1_best.mean(), 2)} & {tfmt(b.time_parallel.mean())} & "
                     f"{gcols} & {orig_s} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "scaling.tex").write_text("\n".join(lines) + "\n")


def table_budget(df):
    f120, f300 = RES / "exdbn_budget_120s_2t.csv", RES / "exdbn_budget_300s_6t.csv"
    if not (f120.exists() and f300.exists()):
        return
    b120, b300 = pd.read_csv(f120).set_index("instance"), pd.read_csv(f300).set_index("instance")
    lines = [r"\begin{tabular}{l rr rr rr r}", r"\toprule",
             r" & \multicolumn{2}{c}{original, 120\,s, 2 thr.} & \multicolumn{2}{c}{Gram, 120\,s, 2 thr.} & "
             r"\multicolumn{2}{c}{Gram, 300\,s, 6 thr.} & \feddbn{} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-8}",
             r"instance & SHD & gap & SHD & gap & SHD & gap & SHD ($K{=}8$) \\", r"\midrule"]
    for inst in b300.index:
        cfg, d, n, sd = inst.split("_")
        row = df[(df.config == cfg) & (df.d == int(d[1:])) & (df.n == int(n[1:])) & (df.seed == int(sd[1:]))]
        o = row.iloc[0]
        a = b120.loc[inst] if inst in b120.index else None
        lines.append(f"{cfg}, $d={d[1:]}$ & {fmt(o.ex)} & {fmt(o.gap, 2)} & "
                     f"{fmt(a.shd_best) if a is not None else '--'} & {fmt(a.gap, 2) if a is not None else '--'} & "
                     f"{fmt(b300.loc[inst].shd_best)} & {fmt(b300.loc[inst].gap, 2)} & {fmt(o.fed8)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "budget.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ex, fd = load_main()
    df = per_instance(ex, fd)
    df.to_csv(RES / "per_instance.csv", index=False)
    table_main(df)
    table_decomp(df)
    table_budget(df)
    scal = None
    if (RES / "scaling_feddbn.csv").exists():
        parts = [pd.read_csv(RES / "scaling_feddbn.csv")]
        parts += [pd.read_csv(f) for f in sorted(RES.glob("scaling_exdbn*.csv"))]
        scal = pd.concat(parts, ignore_index=True)
        for c in ("failed", "time_limit", "time", "build", "gap"):
            if c not in scal:
                scal[c] = np.nan
        fig_scaling(scal)
        table_scaling(scal)
    table_comm(df, scal if scal is not None else pd.DataFrame(columns=["method", "d", "K"]))
    fig_delay()
    print(open(RES / "decomp_by_d.csv").read())
    print(open(RES / "dfed_by_K.csv").read())
