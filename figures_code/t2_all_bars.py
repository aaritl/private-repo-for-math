"""All regions passed to Tier 3, same style as the Tier 2 top-15 chart. usage: python t2_all_bars.py <tier2_regions_to_tier3.csv> <out png>"""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, INK, INK2, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
SHORT = {"Australian Capital Territory": "Australian Capital Terr.", "Capital Region of Denmark": "Capital Region",
         "Northern and Eastern Finland": "Northern & Eastern Finland", "Newfoundland and Labrador": "Newfoundland & Labrador"}
w = pd.read_csv(sys.argv[1]).sort_values("score")
if len(sys.argv) > 3: w = w.tail(int(sys.argv[3]))
labels = [f"{SHORT.get(r, r)} ({c})" + ("  *" if r == "Mayotte" else "") for r, c in zip(w.region, w.country)]

fig, ax = plt.subplots(figsize=(6.8, 0.9 + 0.24 * len(w)) if len(w) <= 30 else (6.8, 0.6 + 0.205 * len(w)))
bars = ax.barh(labels, w.score, height=0.62, color=BLUE)
for b, v in zip(bars, w.score):
    ax.text(v + 0.004, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=7.5, color=INK2)
lo = 0.6 if w.score.min() >= 0.6 else (w.score.min() // 0.1) * 0.1
ax.set_xlim(lo, 0.95); ax.set_ylim(-0.6, len(w) - 0.4)
ax.set_xlabel("Tier 2 score (1 = best on every criterion)")
ax.tick_params(axis="y", labelsize=8)
ax.spines[["top", "right"]].set_visible(False); ax.spines[["left", "bottom"]].set_color(GRID)
ax.tick_params(length=0); ax.grid(axis="x", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
if len(w) > 30: ax.xaxis.set_ticks_position("both"); ax.tick_params(axis="x", labeltop=True)
if (w.region == "Mayotte").any():
    fig.text(0.99, 0.002, "* Mayotte: overseas territory, left out of Tier 3 (reached France's top 5 only through missing data)",
             ha="right", va="bottom", fontsize=7, color=INK2)
plt.tight_layout(rect=(0, 0.006, 1, 1)); plt.savefig(sys.argv[2], dpi=200); plt.close()
print(sys.argv[2], len(w))
