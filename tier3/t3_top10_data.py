"""Tier 3 top 10: score bars + the raw data behind each score. usage: python t3_top10_data.py <ranking csv> <out png>"""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
BLUE, INK, INK2, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
SRC = {"city (Eurostat)": "", "region (OECD)": "†", "country (UN)": "*"}
r = pd.read_csv(sys.argv[1]).nsmallest(10, "rank").iloc[::-1]
fig = plt.figure(figsize=(8.6, 4.0)); ax = fig.add_axes([0.30, 0.19, 0.47, 0.72])
bars = ax.barh([f"{c} ({k})" for c, k in zip(r.city, r.country)], r.score, height=0.62, color=BLUE)
for b, v in zip(bars, r.score):
    ax.text(v + 0.004, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=8, color=INK2)
ax.set_xlim(0.6, 1.05); ax.set_ylim(-0.6, len(r) - 0.4)
ax.set_xlabel("Tier 3 score (1 = best on every criterion)")
ax.spines[["top", "right"]].set_visible(False); ax.spines[["left", "bottom"]].set_color(GRID)
ax.tick_params(length=0); ax.grid(axis="x", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
from matplotlib.transforms import blended_transform_factory
tr = blended_transform_factory(fig.transFigure, ax.transData)
for x, h in [(0.85, "Born abroad"), (0.95, "Venues")]:
    ax.text(x, len(r) - 0.3, h, transform=tr, ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=INK2)
for i, (_, row) in enumerate(r.iterrows()):
    ax.text(0.85, i, f"{row.diversity:.1f}%{SRC.get(row.diversity_source, '')}", transform=tr, ha="center", va="center", fontsize=8.5)
    ax.text(0.95, i, f"{int(row.venues)}", transform=tr, ha="center", va="center", fontsize=8.5)
fig.text(0.30, 0.015, "† region's share (no city figure)   * national share (no city or region figure)", ha="left", fontsize=7, color=INK2)
plt.savefig(sys.argv[2], dpi=220); plt.close(); print(sys.argv[2])
