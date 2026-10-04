"""Figures 1-4 of the main text. Output: PNG files (300 dpi, white background, for print) in paper-E/figures/, or in the
directory given by the environment variable PAPERE_FIG_DIR.

Figure 1  Distribution of the per-unit values at the specification time (30 min): single peak (slow release) and two
          clusters (spread opening), each with the density of the normal distribution with the same mean and SD.
Figure 2  Probability of failing against the fraction p of units below R: three ageing types x true (solid) and the
          normal calculation (dashed), one panel each for Interpretation 1 and Interpretation 2.
Figure 3  Probability of passing for lots of equal quality (the framework of Novick et al.): Interpretation 1, against
          the fraction of units below Q, for five shapes (two normal lots and the three ageing types).
Figure 4  Published data: the values read from Nagase et al. (1976) for 4 tablets on opening and after 80 days, with
          the unit model fitted to each tablet.

Colours are taken in order from a standard categorical palette (blue, orange, aqua, yellow, magenta). Types are
distinguished by line style and direct labels as well as by colour. All values are assumed, except Figure 4, which
shows values read from a published figure.
"""
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

from regime_diagnostic import qmod, verdicts, S_MEAS, R2, Q1, T_SPEC
import equal_quality as NF
import nagase1976 as NG

OUT = Path(os.environ.get("PAPERE_FIG_DIR", Path(__file__).resolve().parent.parent / "figures"))
OUT.mkdir(exist_ok=True)
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
TYPES = [("S slow release", "Slow release", C["blue"]), ("O spread opening", "Spread opening", C["orange"]), ("M mixed", "Mixed", C["aqua"])]
rng = np.random.default_rng(20261001)
N_LOT = 100_000

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2.0,
    "legend.frameon": False, "savefig.dpi": 300, "figure.facecolor": "white",
})


def units(th, n):
    mL, sL, mk, sk = th
    L = np.exp(mL + sL * rng.standard_normal(n)); k = np.exp(mk + sk * rng.standard_normal(n))
    return qmod(T_SPEC, L, k) + S_MEAS * rng.standard_normal(n)


def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    for ax, (kind, label, col) in zip(axes, TYPES[:2]):
        th = NF.solve(kind, 0.10)
        y = units(th, 400_000)
        bins = np.arange(-5, 106, 2.5)
        ax.hist(y, bins=bins, density=True, color=col, alpha=0.85, edgecolor="white", linewidth=0.6)
        xs = np.linspace(-5, 110, 400)
        ax.plot(xs, norm.pdf(xs, y.mean(), y.std()), color=INK, lw=1.4, ls="--")
        for x, t, ha in ((R2, "R", "center"), (Q1 - 15, "Q−15", "left"), (Q1 - 25, "Q−25", "right")):
            ax.axvline(x, color=INK2, lw=0.8, ls=":")
            ax.text(x + (1 if ha == "left" else -1 if ha == "right" else 0), 1.02, t, transform=ax.get_xaxis_transform(),
                    ha=ha, va="bottom", fontsize=7.5, color=INK2)
        ax.set_yscale("log"); ax.set_ylim(1e-4, 0.3); ax.set_xlim(-5, 107)
        ax.set_xlabel(f"Amount dissolved at {T_SPEC:.0f} min (% of label claim)")
        ax.set_title(f"{label}  (p = 0.10)", loc="left", fontsize=9, color=INK, pad=14)
        ax.text(0.03, 0.93, f"mean {y.mean():.1f}, SD {y.std():.1f}", transform=ax.transAxes, fontsize=7.5, color=INK2)
    axes[0].set_ylabel("Density (log scale)")
    axes[1].annotate("normal with\nthe same\nmean and SD", xy=(35, 9e-4), xytext=(7, 0.02), fontsize=7.5, color=INK,
                     arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    fig.tight_layout()
    fig.savefig(OUT / "fig1_unit_distributions.png"); plt.close(fig)


def fig2():
    ps = np.array([0.02, 0.035, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30])
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
    res = {}
    for kind, label, col in TYPES:
        tr, nr = [], []
        for p in ps:
            th = NF.solve(kind, p)
            mL, sL, mk, sk = th
            L = np.exp(mL + sL * rng.standard_normal((N_LOT, 24))); k = np.exp(mk + sk * rng.standard_normal((N_LOT, 24)))
            q = qmod(T_SPEC, L, k) + S_MEAS * rng.standard_normal((N_LOT, 24))
            tr.append(verdicts(q)); nr.append(verdicts(q.mean() + q.std() * rng.standard_normal((N_LOT, 24))))
        res[kind] = (np.array(tr), np.array(nr))
    for j, (ax, title) in enumerate(zip(axes, ("Interpretation 1 (Q value, S1–S3)", "Interpretation 2 (individual limit, 6 → 12)"))):
        for kind, label, col in TYPES:
            tr, nr = res[kind]
            ax.plot(ps, tr[:, j], color=col, lw=2.0)
            ax.plot(ps, nr[:, j], color=col, lw=1.4, ls="--")
            if j == 0:
                ax.text(ps[-1] + 0.006, tr[-1, j], label, color=INK2, fontsize=7.2, va="center")
            else:
                ax.text(ps[-1] + 0.006, nr[-1, j] + {"S slow release": -0.03, "O spread opening": 0.02, "M mixed": -0.02}[kind],
                        label + " (normal)", color=INK2, fontsize=7.2, va="center")
        if j == 1:
            ax.text(0.215, 0.17, "true: identical for\nall three types", color=INK2, fontsize=7.2)
        ax.set_xlim(0, 0.36); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(f"Fraction of units below R at {T_SPEC:.0f} min (p)")
        ax.set_title(title, loc="left", fontsize=9, color=INK)
    axes[0].set_ylabel("Probability of failing")
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=INK2, lw=2), Line2D([], [], color=INK2, lw=1.4, ls="--")],
               ["true (unit model)", "normal with the same mean and SD"], loc="upper center", ncol=2, fontsize=7.5,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "fig2_true_vs_normal.png"); plt.close(fig)
    return ps, res


def fig3():
    ps = np.array([0.01, 0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30])
    curves = {"Normal, SD 3": [], "Normal, SD 6": []}
    for kind, label, col in TYPES:
        curves[label] = []
    for p in ps:
        curves["Normal, SD 3"].append(NF.oc_normal(p, 3.0, Q1)[0][0])
        curves["Normal, SD 6"].append(NF.oc_normal(p, 6.0, Q1)[0][0])
        for kind, label, col in TYPES:
            curves[label].append(NF.oc_lag(NF.solve(kind, p, Q1))[0])
    style = {"Normal, SD 3": (MUTED, "-"), "Normal, SD 6": (MUTED, ":"), "Slow release": (C["blue"], "-"),
             "Spread opening": (C["orange"], "-"), "Mixed": (C["aqua"], "-")}
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    for name, ys in curves.items():
        col, ls = style[name]
        ax.plot(ps, ys, color=col, ls=ls, lw=2.0 if col != MUTED else 1.6)
    ax.text(0.305, curves["Spread opening"][-1], "Spread opening", color=INK2, fontsize=7.2, va="center")
    ax.text(0.305, curves["Mixed"][-1], "Mixed", color=INK2, fontsize=7.2, va="center")
    ax.text(0.305, 0.94, "Slow release,\nnormal SD 3 and 6", color=INK2, fontsize=7.2, va="center")
    ax.set_xlim(0, 0.39); ax.set_ylim(0, 1.03)
    ax.set_xlabel(f"Fraction of units below Q at {T_SPEC:.0f} min")
    ax.set_ylabel("Probability of passing\n(Interpretation 1)")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_equal_quality.png"); plt.close(fig)


def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    ts = np.array([20, 30, 40, 50, 60])
    for ax, (cond, title) in zip(axes, (("A", "On opening"), ("B", "After 80 days at 25 °C / 81 % RH"))):
        for i, (o, t50, pct) in enumerate(NG.DATA[cond]):
            L, k, _ = NG.fit_unit(o, t50, pct)
            tt = np.linspace(0, 70, 300)
            col = (C["blue"], C["orange"], C["aqua"], C["yellow"])[i]
            ax.plot(tt, NG.model(tt, L, k), color=col, lw=1.6)
            ax.plot(np.r_[o, t50, ts], np.r_[0, 50, pct], "o", color=col, ms=4, mec="white", mew=0.6)
        ax.set_xlim(0, 70); ax.set_ylim(-3, 105)
        ax.set_xlabel("Time (min)")
        ax.set_title(title, loc="left", fontsize=9, color=INK)
    axes[0].set_ylabel("Amount dissolved (%)")
    axes[1].text(36, 12, "points: read from the published figure\nlines: unit model fitted to each tablet",
                 fontsize=7, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_nagase1976.png"); plt.close(fig)


if __name__ == "__main__":
    fig1(); print("fig1")
    fig2(); print("fig2")
    fig3(); print("fig3")
    fig4(); print("fig4")
