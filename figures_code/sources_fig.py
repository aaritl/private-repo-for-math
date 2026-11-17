"""Plain data-sources table for Tier 2 and Tier 3. usage: python sources_fig.py <out png>"""
import sys, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 10})
ROWS = [
 ("Tier 2", "Homicide rate", "Intentional homicides per 100,000", "OECD Regional Statistics: Safety – Regions", "2020–2025"),
 ("", "Vehicle theft rate", "Motor vehicle thefts per 100,000", "OECD Regional Statistics: Safety – Regions", "2020–2025"),
 ("", "Air quality", "Average exposure to PM2.5 (µg/m³)", "OECD Regional Statistics: Exposure to air pollution – Regions", "2023"),
 ("", "Green space", "Green area per person (m²), average of the region's cities", "OECD Cities (Functional Urban Areas): Green areas", "2021"),
 ("", "Public transport", "Share of people within a 10-minute walk of a stop (%), average of the region's cities", "OECD Cities (Functional Urban Areas): Access to public transport", "2023"),
 ("Tier 3", "Diversity", "Share of residents born in another country (%)", "Eurostat Urban Audit (cities); OECD Foreign-born population – Regions (if no city value); UN International Migrant Stock via World Bank (New Zealand)", "2013–2025"),
 ("", "Entertainment", "Notable museums, theatres, cinemas, concert halls, galleries, music venues, stadiums, zoos and theme parks within 5 km of the centre (log scale)", "Wikidata Query Service", "2026"),
 ("Both", "City locations", "Placing each city inside its region (not scored)", "GeoNames; Eurostat GISCO NUTS boundaries", "2024"),
]
HEAD = ("Tier", "Criterion", "What it measures", "Source", "Years")
X = [0.0, 0.07, 0.21, 0.55, 0.92]; WRAP = [8, 18, 44, 46, 10]
lines = [[textwrap.wrap(c, w) or [""] for c, w in zip(r, WRAP)] for r in ROWS]
heights = [max(len(c) for c in r) for r in lines]
LH = 0.26; total = (sum(heights) + len(ROWS) * 0.6 + 1.6) * LH
fig = plt.figure(figsize=(10, total)); ax = fig.add_axes([0.02, 0.02, 0.96, 0.96]); ax.axis("off")
H = sum(heights) + len(ROWS) * 0.6 + 1.2; ax.set_ylim(H, 0); ax.set_xlim(0, 1)
y = 0.3
for x, h in zip(X, HEAD): ax.text(x, y, h, fontweight="bold", va="top")
y += 1.0; ax.plot([0, 1], [y - 0.15] * 2, color="black", lw=1)
ax.plot([0, 1], [0.05] * 2, color="black", lw=1.2)
for i, (r, ln, h) in enumerate(zip(ROWS, lines, heights)):
    if i > 0 and r[0]: ax.plot([0, 1], [y - 0.15] * 2, color="#888888", lw=0.6)
    for x, cell in zip(X, ln):
        for k, t in enumerate(cell): ax.text(x, y + k, t, va="top", fontweight="bold" if x == 0 else "normal")
    y += h + 0.6
ax.plot([0, 1], [y - 0.3] * 2, color="black", lw=1.2)
plt.savefig(sys.argv[1], dpi=250, bbox_inches="tight", pad_inches=0.1, facecolor="white"); print(sys.argv[1])
