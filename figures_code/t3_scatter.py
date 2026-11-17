"""Tier 3: diversity vs entertainment for every city, top 10 highlighted. usage: python t3_scatter.py <ranking csv> <out png>"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
BLUE, LIGHT, INK2, GRID = "#2a78d6", "#b9c3cf", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})
r = pd.read_csv(sys.argv[1])
top = r[r["rank"] <= 10]; rest = r[r["rank"] > 10]
fig, ax = plt.subplots(figsize=(7.0, 4.8))
ax.scatter(rest.diversity, rest.venues, s=30, color=LIGHT, edgecolor="white", lw=0.8, zorder=2, label="other cities")
ax.scatter(top.diversity, top.venues, s=46, color=BLUE, edgecolor="white", lw=0.8, zorder=3, label="top 10")
OFF = {"Göteborg": (6, -2), "Edinburgh": (6, 2), "Glasgow": (6, 0), "Aberdeen": (6, 0), "Lübeck": (6, 0), "Bergen": (-6, 0),
       "Wellington": (-6, -6), "Canberra": (6, -8), "Saarbrücken": (-6, 6), "Flensburg": (6, -2)}
for _, row in top.iterrows():
    dx, dy = OFF.get(row.city, (6, 0))
    ax.annotate(f"{int(row['rank'])}. {row.city}", (row.diversity, row.venues), xytext=(dx, dy), textcoords="offset points",
                ha="left" if dx > 0 else "right", va="center", fontsize=8, color="#0b0b0b")
ax.set_yscale("log"); ax.set_yticks([2, 5, 10, 20, 50, 100, 150]); ax.set_yticklabels(["2", "5", "10", "20", "50", "100", "150"])
ax.set_xlabel("Diversity: residents born abroad (%)"); ax.set_ylabel("Entertainment: venues within 5 km (log scale)")
ax.spines[["top", "right"]].set_visible(False); ax.tick_params(length=0); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_xlim(0, 40)
ax.legend(loc="lower right", frameon=False, fontsize=8)
ax.text(0.01, 0.02, "r = 0.23 between the two criteria (44 cities)", transform=ax.transAxes, va="bottom", fontsize=8, color=INK2)
plt.tight_layout(); plt.savefig(sys.argv[2], dpi=220); plt.close(); print(sys.argv[2])
