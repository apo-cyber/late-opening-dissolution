"""Graphical abstract (required by the journal). As the journal asks, it conveys the main point at a glance and shows
no specific numerical results. Three panels, from left to right:
 A Units whose opening is delayed (dissolution curves; units not yet opened at the specification time are near 0 %).
 B The values at the specification time form two clusters -> the normal distribution with the same mean and SD puts too
   many units below the shallow threshold R and almost none below the deep threshold Q - 25.
 C Record each vessel at 10, 20 and 30 min -> the unit model identifies the regime and forecasts the probability of
   failing.
All curves and points are schematic (not values from the simulations of the paper). Colours as in figures.py.
Output: graphical_abstract.{eps,pdf,png} in paper-E/figures/, or in the directory given by PAPERE_FIG_DIR (EPS is the
journal's recommended format; fonts are embedded).
"""
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.stats import norm

C = {"blue": "#2a78d6", "orange": "#eb6834"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.6, "figure.facecolor": "white", "ps.fonttype": 42, "pdf.fonttype": 42,
})
OUT = Path(os.environ.get("PAPERE_FIG_DIR", Path(__file__).resolve().parent.parent / "figures"))


def D(t, L, k):
    """For the schematic only: release starts gradually (a sugar coat thins progressively rather than opening all at
    once like a shell), rising from the start of opening with a Weibull shape (shape parameter 2). This is closer to
    reality than the model of the paper (shape parameter 1)."""
    u = np.clip(t - L, 0, None)
    return 100 * (1 - np.exp(-(k * u) ** 2))


fig = plt.figure(figsize=(9.0, 4.4))
axA = fig.add_axes([0.075, 0.30, 0.235, 0.50])
axB = fig.add_axes([0.405, 0.30, 0.235, 0.50])
axC = fig.add_axes([0.735, 0.30, 0.235, 0.50])
TS, R, QM25 = 30.0, 75.0, 45.0

# ---- A: units that open late (L is the start of opening) ----
t = np.linspace(0, 45, 400)
Ls = [2, 3, 4, 5, 6, 7, 8.5, 10, 12, 15, 31, 36]
for L in Ls:
    col = C["orange"] if L > TS else C["blue"]
    axA.plot(t, D(t, L, 0.16), color=col, alpha=0.85)
    axA.plot([TS], [D(np.array([TS]), L, 0.16)[0]], "o", ms=4.5, color=col, mec="white", mew=0.6, zorder=3)
axA.axvline(TS, color=MUTED, ls="--", lw=0.9)
axA.text(TS - 0.8, 60, "specification\ntime", fontsize=7.5, color=INK2, va="center", ha="right")
axA.set_xlim(0, 45); axA.set_ylim(-4, 108)
axA.set_xlabel("Time (min)"); axA.set_ylabel("Dissolved (%)")
axA.set_xticks([]); axA.set_yticks([0, 100])
axA.text(1, 92, "opened", color=C["blue"], fontsize=8, fontweight="bold")
axA.text(23.5, 14, "not yet\nopened", color=C["orange"], fontsize=8, fontweight="bold", ha="center")
fig.text(0.075, 0.115, "Opening times lengthen and spread\n(sugar-coated tablets, capsules)", fontsize=7.4, color=INK2, va="center", linespacing=1.35)
fig.text(0.075, 0.91, "On storage, some units\nopen late", fontsize=10, fontweight="bold", color=INK, va="center")

# ---- B: two clusters and the normal distribution ----
hi = np.linspace(91, 100, 21)
lo = np.array([0.5, 1.5, 2.5])
vals = np.r_[hi, lo]
jit = np.tile([0.15, 0.45, 0.30, 0.60], 6)[: len(vals)]
axB.scatter(hi, jit[: len(hi)] * 0.010, s=16, color=C["blue"], edgecolor="white", linewidth=0.5, zorder=3)
axB.scatter(lo, jit[len(hi):] * 0.010, s=16, color=C["orange"], edgecolor="white", linewidth=0.5, zorder=3)
x = np.linspace(-10, 125, 500)
pdf = norm.pdf(x, vals.mean(), vals.std())
axB.plot(x, pdf, color=MUTED, ls="--", lw=1.4)
axB.fill_between(x, 0, pdf, where=(x > QM25) & (x < R), color=GRID, lw=0)
for xv, lab in ((R, "R"), (QM25, "Q \u2212 25")):
    axB.axvline(xv, color=INK2, lw=0.9, ls=":")
    axB.text(xv, 0.0215, lab, ha="center", fontsize=8, color=INK2)
axB.set_xlim(-8, 110); axB.set_ylim(0, 0.023)
axB.set_yticks([]); axB.spines["left"].set_visible(False)
axB.set_xticks([0, 100]); axB.set_xlabel("Dissolved at specification time (%)")
axB.text(14, 0.0150, "normal with the\nsame mean and SD", fontsize=7.2, color=INK2, ha="center")
axB.annotate("", xy=(37, 0.0084), xytext=(20, 0.0140), arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
fig.text(0.405, 0.91, "Unit values form two clusters;\nthe normal calculation errs", fontsize=10, fontweight="bold", color=INK, va="center")
fig.text(0.405, 0.115, "Just below R: too many units\n\u2192 JP individual-value test: failures overstated\n"
                       "Below Q \u2212 25: almost none\n\u2192 harmonized test: failures missed",
         fontsize=7.4, color=INK2, va="center", linespacing=1.35)

# ---- C: records at three times and the unit model ----
months = np.array([0, 3, 6, 9, 12])
fut = np.linspace(0, 18, 200)
curve = 1 / (1 + np.exp(-(fut - 14.5) / 1.6))
axC.axvspan(12, 18, color=GRID, lw=0)
axC.plot(fut[fut <= 12], curve[fut <= 12], color=C["blue"])
axC.plot(fut[fut >= 12], curve[fut >= 12], color=C["blue"], ls="--")
axC.plot(months, 1 / (1 + np.exp(-(months - 14.5) / 1.6)) + np.array([0.01, -0.005, 0.012, -0.01, 0.02]),
         "o", ms=5, mfc="white", mec=INK2, mew=1.1, zorder=3)
axC.text(15, 0.08, "forecast", ha="center", fontsize=8, color=INK2)
axC.set_xlim(0, 18); axC.set_ylim(-0.03, 1.05)
axC.set_xticks([]); axC.set_yticks([])
axC.set_xlabel("Storage time"); axC.set_ylabel("Probability of failing")
fig.text(0.735, 0.91, "Record each vessel at two\nearlier times; fit a unit model", fontsize=10, fontweight="bold", color=INK, va="center")
fig.text(0.735, 0.115, "Opening time and release rate\nof each unit estimated\n\u2192 regime identified, failure forecast",
         fontsize=7.4, color=INK2, va="center", linespacing=1.35)

for x0, x1 in ((0.33, 0.385), (0.66, 0.715)):
    fig.add_artist(FancyArrowPatch((x0, 0.55), (x1, 0.55), transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=14, color=MUTED, lw=1.4))

OUT.mkdir(exist_ok=True)
for ext in ("eps", "pdf", "png"):
    fig.savefig(OUT / f"graphical_abstract.{ext}", dpi=600 if ext == "png" else None)
print("Written:", [str(OUT / f"graphical_abstract.{e}") for e in ("eps", "pdf", "png")])
