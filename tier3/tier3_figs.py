"""Tier 3 charts (same style as Tier 1/2). usage: python tier3_figs.py <tier3 results dir> <fig dir>"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

IN, FIG = Path(sys.argv[1]), Path(sys.argv[2]); FIG.mkdir(parents=True, exist_ok=True)
BLUE, INK, INK2, MUTED, GRID, LIGHT = "#2a78d6", "#0b0b0b", "#52514e", "#9a9994", "#e6e5e1", "#cde2fb"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("b", RAMP)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False); ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(length=0); ax.grid(axis="x", color=GRID, linewidth=0.6); ax.set_axisbelow(True); ax.grid(axis="y", visible=False)

r = pd.read_csv(IN / "tier3_city_ranking.csv")
g = pd.read_csv(IN / "tier3_grades.csv").set_index("fua_code")
w = pd.read_csv(IN / "tier3_weights.csv")
fq = pd.read_csv(IN / "tier3_top10_frequency.csv")
NAME = {"Greater Sydney": "Sydney", "Greater Melbourne": "Melbourne", "Greater Perth": "Perth", "'s-Gravenhage": "The Hague",
        "Praha": "Prague", "Zürich": "Zurich", "Antwerpen": "Antwerp", "München": "Munich", "Wien": "Vienna",
        "København": "Copenhagen", "Bruxelles/Brussel/Leuven": "Brussels", "Lisboa": "Lisbon", "Warszawa": "Warsaw", "Gent": "Ghent"}
lab = lambda c, k: f"{NAME.get(c, c)} ({k})"
COLS = list(w.criterion); LABEL = dict(zip(w.criterion, w.label))

# 1. top 15 cities
top = r.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(6.6, 4.4))
bars = ax.barh([lab(c, k) for c, k in zip(top.city, top.country)], top.score, height=0.62, color=BLUE)
for b, v in zip(bars, top.score):
    ax.text(v + 0.003, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=8, color=INK2)
ax.set_xlim(0.6, 0.8); ax.set_xlabel("Tier 3 score (1 = best on every criterion)"); tidy(ax)
plt.tight_layout(); plt.savefig(FIG / "top15.png", dpi=220); plt.close()

# 2. weights
ws = w.sort_values("weight")
fig, ax = plt.subplots(figsize=(6.6, 2.4))
bars = ax.barh(ws.label, ws.weight * 100, height=0.6, color=BLUE)
for b, v in zip(bars, ws.weight * 100):
    ax.text(v + 0.3, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=8, color=INK2)
ax.set_xlim(0, 30); ax.set_xlabel("Weight (%)"); tidy(ax)
plt.tight_layout(); plt.savefig(FIG / "weights.png", dpi=220); plt.close()

# 3. heatmap: top 10 x 6 criteria (colour = 0-1 grade, text = raw value)
hm = r.head(10)
S = g.loc[hm.fua_code, COLS].values.astype(float)
FMT = {"pm25": "{:.1f}", "green_area": "{:.0f} m²", "transit": "{:.0f}%", "services": "{:.0f}%", "unemployment": "{:.1f}%"}
fig, ax = plt.subplots(figsize=(7.4, 4.3))
im = ax.imshow(S, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(COLS))); ax.set_xticklabels([LABEL[c].replace(" ", "\n", 1) for c in COLS], fontsize=8.5)
ax.set_yticks(range(len(hm))); ax.set_yticklabels([f"{int(k)}. {lab(c, n)}" for k, c, n in zip(hm["rank"], hm.city, hm.country)], fontsize=8.5)
ax.xaxis.tick_top(); ax.tick_params(length=0)
for sp in ax.spines.values(): sp.set_visible(False)
for i, (_, row) in enumerate(hm.iterrows()):
    for j, c in enumerate(COLS):
        if np.isnan(S[i, j]):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#f5f5f3", edgecolor="white", lw=2))
            ax.text(j, i, "no data", ha="center", va="center", fontsize=7, color=MUTED); continue
        txt = f"{int(row.venues)} venues" if c == "entertainment" else FMT[c].format(row[c])
        ax.text(j, i, txt, ha="center", va="center", fontsize=7.8, color="white" if S[i, j] > 0.55 else INK)
ax.set_xticks(np.arange(-0.5, len(COLS), 1), minor=True); ax.set_yticks(np.arange(-0.5, len(hm), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2); ax.tick_params(which="minor", length=0)
ax.set_xlim(-0.5, len(COLS) - 0.45); ax.set_ylim(len(hm) - 0.45, -0.55)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels(["worst", "", "best"])
cb.outline.set_visible(False); cb.ax.tick_params(length=0, labelsize=8)
plt.tight_layout(); plt.savefig(FIG / "heatmap.png", dpi=220); plt.close()

# 4. how often each city lands in the top 10 across the stability runs
f = fq.head(15).iloc[::-1]; n = int(fq.out_of.iloc[0])
fig, ax = plt.subplots(figsize=(6.6, 4.4))
cols = [BLUE if (k <= 10) else LIGHT for k in f.main_rank]
bars = ax.barh([lab(c, k) for c, k in zip(f.city, f.country)], f.times_in_top10, height=0.62, color=cols)
for b, v, k in zip(bars, f.times_in_top10, f.main_rank):
    ax.text(v + 0.1, b.get_y() + b.get_height() / 2, f"{v}/{n}   (main rank #{int(k)})", va="center", fontsize=7.8, color=INK2)
ax.set_xlim(0, n + 3.2); ax.set_xticks(range(0, n + 1, 2)); ax.set_xlabel(f"Times in the top 10 across {n} model versions"); tidy(ax)
ax.scatter([], [], color=BLUE, marker="s", label="in the main top 10"); ax.scatter([], [], color=LIGHT, marker="s", label="outside the main top 10")
ax.legend(loc="lower right", fontsize=7.5, frameon=False)
plt.tight_layout(); plt.savefig(FIG / "top10_frequency.png", dpi=220); plt.close()
print("saved", FIG)
