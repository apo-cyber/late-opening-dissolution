"""Graphical abstract for Drug Development and Industrial Pharmacy (maximum width 525 pixels). Same three panels and the
same schematic curves as graphical_abstract.py, laid out directly at 525 pixels wide (100 dpi) with fewer words and
larger type, because the full-size version shrunk to 525 pixels leaves its smallest text about 5 pixels high.
All curves and points are schematic (not values from the simulations of the paper).
Output: ddip/GraphicalAbstract1.png in paper-E/figures/, or in the directory given by PAPERE_FIG_DIR.
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
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.3, "figure.facecolor": "white",
})
OUT = Path(os.environ.get("PAPERE_FIG_DIR", Path(__file__).resolve().parent.parent / "figures")) / "ddip"
W_PX, DPI = 525, 100
TITLE, NOTE = 9, 8                      # pt; at 100 dpi, 8 pt is about 11 pixels


def D(t, L, k):
    """Schematic release with a gradual start (Weibull shape 2), as in graphical_abstract.py."""
    u = np.clip(t - L, 0, None)
    return 100 * (1 - np.exp(-(k * u) ** 2))


fig = plt.figure(figsize=(W_PX / DPI, 3.4))
COL = (0.015, 0.350, 0.685)           # left edge of each column (title, panel, note)
axA, axB, axC = (fig.add_axes([c + 0.075, 0.30, 0.215, 0.44]) for c in COL)
TS, R, QM25 = 30.0, 75.0, 45.0
TOP, BOTTOM = 0.90, 0.09

# ---- A: units that open late ----
t = np.linspace(0, 45, 400)
for L in [2, 3, 4, 5, 6, 7, 8.5, 10, 12, 15, 31, 36]:
    col = C["orange"] if L > TS else C["blue"]
    axA.plot(t, D(t, L, 0.16), color=col, alpha=0.85)
axA.axvline(TS, color=MUTED, ls="--", lw=0.8)
axA.set_xlim(0, 45); axA.set_ylim(-4, 108)
axA.set_xlabel("Time"); axA.set_ylabel("Dissolved (%)")
axA.set_xticks([]); axA.set_yticks([0, 100])
axA.annotate("late", xy=(33.3, 14), xytext=(23, 14), color=C["orange"], fontsize=NOTE, fontweight="bold",
             ha="center", va="center", arrowprops=dict(arrowstyle="-", color=C["orange"], lw=0.8, shrinkA=1, shrinkB=1))
fig.text(COL[0], TOP, "On storage, some\nunits open late", fontsize=TITLE, fontweight="bold", color=INK, va="center")
fig.text(COL[0], BOTTOM, "Opening times\nlengthen and spread", fontsize=NOTE, color=INK2, va="center")

# ---- B: two clusters and the normal distribution ----
hi = np.linspace(91, 100, 21)
lo = np.array([0.5, 1.5, 2.5])
vals = np.r_[hi, lo]
jit = np.tile([0.15, 0.45, 0.30, 0.60], 6)[: len(vals)]
axB.scatter(hi, jit[: len(hi)] * 0.010, s=9, color=C["blue"], edgecolor="white", linewidth=0.3, zorder=3)
axB.scatter(lo, jit[len(hi):] * 0.010, s=9, color=C["orange"], edgecolor="white", linewidth=0.3, zorder=3)
x = np.linspace(-10, 125, 500)
pdf = norm.pdf(x, vals.mean(), vals.std())
axB.plot(x, pdf, color=MUTED, ls="--", lw=1.2)
axB.fill_between(x, 0, pdf, where=(x > QM25) & (x < R), color=GRID, lw=0)
axB.text(vals.mean(), pdf.max() * 1.12, "normal", fontsize=NOTE, color=INK2, ha="center")
axB.set_xlim(-8, 110); axB.set_ylim(0, 0.023)
axB.set_yticks([]); axB.spines["left"].set_visible(False)
axB.set_xticks([0, 100]); axB.set_xlabel("Dissolved (%)")
fig.text(COL[1], TOP, "Two clusters: normal\ncalculation errs", fontsize=TITLE, fontweight="bold", color=INK, va="center")
fig.text(COL[1], BOTTOM, "Failures overstated (JP)\nor missed (harmonized)", fontsize=NOTE, color=INK2, va="center")

# ---- C: records at several times and the unit model ----
months = np.array([0, 3, 6, 9, 12])
fut = np.linspace(0, 18, 200)
curve = 1 / (1 + np.exp(-(fut - 14.5) / 1.6))
axC.axvspan(12, 18, color=GRID, lw=0)
axC.plot(fut[fut <= 12], curve[fut <= 12], color=C["blue"])
axC.plot(fut[fut >= 12], curve[fut >= 12], color=C["blue"], ls="--")
axC.plot(months, 1 / (1 + np.exp(-(months - 14.5) / 1.6)) + np.array([0.01, -0.005, 0.012, -0.01, 0.02]),
         "o", ms=3.5, mfc="white", mec=INK2, mew=0.9, zorder=3)
axC.set_xlim(0, 18); axC.set_ylim(-0.03, 1.05)
axC.set_xticks([]); axC.set_yticks([])
axC.set_xlabel("Storage time"); axC.set_ylabel("P(fail)")
fig.text(COL[2], TOP, "Record each vessel;\nfit a unit model", fontsize=TITLE, fontweight="bold", color=INK, va="center")
fig.text(COL[2], BOTTOM, "Regime identified,\nfailure forecast", fontsize=NOTE, color=INK2, va="center")

for x0, x1 in ((0.315, 0.405), (0.65, 0.74)):
    fig.add_artist(FancyArrowPatch((x0, 0.52), (x1, 0.52), transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=10, color=MUTED, lw=1.2))

OUT.mkdir(parents=True, exist_ok=True)
fig.canvas.draw()
from PIL import Image                 # RGB without alpha (an alpha channel caused trouble at submission for paper-B)
Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB").save(
    OUT / "GraphicalAbstract1.png", dpi=(DPI, DPI))
print("Written:", OUT / "GraphicalAbstract1.png")
