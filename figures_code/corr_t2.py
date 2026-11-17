"""Tier 2 criteria correlation (0-1 grades of ranked regions). usage: python corr_t2.py <grades csv> <out png>"""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
S = pd.read_csv(sys.argv[1], index_col=0)
ORDER = ["homicide_rate", "vehicle_theft_rate", "pm25_exposure", "green_space", "transit_access"]
NAMES = ["Homicide", "Vehicle theft", "Air quality", "Green space", "Public transport"]
C = S[ORDER].corr().values
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 10})
cmap = LinearSegmentedColormap.from_list("d", ["#c8693a", "#f3d9c9", "#ffffff", "#c9dcf2", "#2f6db5"])
n = len(ORDER)
fig, ax = plt.subplots(figsize=(5.4, 4.6))
for i in range(n):
    for j in range(i + 1):
        val = C[i, j]
        face = "#f0f0f0" if i == j else cmap((val + 1) / 2)
        ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=face, edgecolor="white", lw=2))
        ax.text(j + 0.5, i + 0.5, "1" if i == j else f"{val:.2f}", ha="center", va="center", fontsize=10.5)
ax.set_xlim(0, n); ax.set_ylim(n, 0); ax.set_aspect("equal"); ax.axis("off")
for i, name in enumerate(NAMES):
    ax.text(-0.1, i + 0.5, name, ha="right", va="center")
    ax.text(i + 0.5, n + 0.12, name, ha="right", va="top", rotation=40, rotation_mode="anchor")
plt.savefig(sys.argv[2], dpi=250, bbox_inches="tight", pad_inches=0.1, facecolor="white")
print(sys.argv[2])
