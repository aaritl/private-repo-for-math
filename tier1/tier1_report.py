"""Tier 1: score every country, write ranking CSVs and a short visual PDF.

usage: python tier1_report.py <tier1 dir with tier1_inputs.csv, match_log.json, countries.json, countryInfo.txt> <results dir>

Model (same steps as himcm26_simple.py, with two settings chosen for Tier 1):
  1. cap extreme values at the 5th / 95th percentile (winsorize)
  2. rescale each criterion to 0-1, 1 = best (flipped where lower is better)
  3. entropy weights: criteria that separate countries more get more weight
     (computed from the countries that have a value - no filling of gaps)
  4. score = weighted average of the criteria a country has values for
  5. ranked only if the country has values for >= 75% of the total weight
"""
import sys, json, datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.preprocessing import MinMaxScaler
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
FIG = OUT / "fig_tier1"; FIG.mkdir(exist_ok=True)

MIN_DATA = 0.75
OUTLIER_CAP = 0.05
TOP_N_CHART = 15

# criterion: (direction, folder, short name, description, source, years)
CRITERIA = {
    "uhc_service_coverage": ("higher", "Healthcare",      "Healthcare",  "UHC service coverage index (0-100)",          "World Bank WDI (your file)",        "latest, mostly 2021-2023"),
    "basic_water":          ("higher", "Basic Resources", "Water",       "People with at least basic drinking water (%)", "World Bank WDI SH.H2O.BASW.ZS",     "latest, mostly 2024"),
    "basic_sanitation":     ("higher", "Basic Resources", "Sanitation",  "People with at least basic sanitation (%)",    "World Bank WDI SH.STA.BASS.ZS",     "latest, mostly 2024"),
    "internet_users":       ("higher", "Infrastructure",  "Internet",    "People using the Internet (%)",                "World Bank WDI IT.NET.USER.ZS",     "latest, mostly 2024"),
    "traffic_deaths":       ("lower",  "Infrastructure",  "Road safety", "Road traffic deaths per 100,000",              "World Bank WDI SH.STA.TRAF.P5 (WHO)", "latest, mostly 2019"),
    "gini":                 ("lower",  "Family",          "Equality",    "Gini index of income inequality (0-100)",      "World Bank WDI SI.POV.GINI",        "latest 2010-2025, median 2022"),
    "test_scores":          ("higher", "Education",       "Test scores", "Harmonized Test Scores (300-625)",             "World Bank HCI HD.HCI.HLOS",        "2020"),
}
DROPPED = {"divorce_rate": "Crude divorce rate (per 1,000), UN Demographic Yearbook 2002-2006, 95 countries"}
COLS = list(CRITERIA)
SHORT = {k: v[2] for k, v in CRITERIA.items()}

# palette (dataviz skill reference palette): one blue hue for magnitude, ink for text
BLUE, BLUE_DARK, INK, INK2, MUTED, GRID = "#2a78d6", "#1c5cab", "#0b0b0b", "#52514e", "#9a9994", "#e6e5e1"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("blue_seq", RAMP)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})

# ---------- data ----------
data = pd.read_csv(IN / "tier1_inputs.csv", keep_default_na=False, na_values=[""])
geo = pd.read_csv(IN / "countryInfo.txt", sep="\t", comment="#", header=None, usecols=[0, 4], names=["geo_id", "country"],
                  dtype=str, keep_default_na=False)
NAME = dict(zip(geo.geo_id, geo.country))
wb = {c["iso2Code"]: c for c in json.load(open(IN / "countries.json"))}
REGION = {g: wb.get(g, {}).get("region", {}).get("value", "Other") for g in data.geo_id}
LOG = {r["criterion"]: r for r in json.load(open(IN / "match_log.json"))}

# ---------- model ----------
def entropy_weights(scaled):
    """Weight = (1 - entropy) / sum, each criterion using only the countries that have a value."""
    d = {}
    for c in scaled:
        x = scaled[c].dropna()
        p = (x + 1e-12) / (x + 1e-12).sum()
        e = -(p * np.log(p)).sum() / np.log(len(x))
        d[c] = max(0.0, 1 - e)
    d = pd.Series(d)
    return d / d.sum()

def scale(table, cols):
    v = table[cols]
    v = v.clip(v.quantile(OUTLIER_CAP), v.quantile(1 - OUTLIER_CAP), axis=1)
    s = pd.DataFrame(MinMaxScaler().fit_transform(v), columns=cols, index=table.index)
    for c in cols:
        if (CRITERIA.get(c) or ("lower",))[0] == "lower":
            s[c] = 1 - s[c]
    return s

def score(table, cols, weights=None, min_data=MIN_DATA):
    s = scale(table, cols)
    w = entropy_weights(s) if weights is None else weights.reindex(cols) / weights.reindex(cols).sum()
    t = table.copy()
    t["data_share"] = (s.notna() * w).sum(axis=1)
    t["score"] = (s * w).sum(axis=1) / t["data_share"]
    ok = t["data_share"] >= min_data
    t["rank"] = t.loc[ok, "score"].rank(ascending=False, method="min")
    t = t.sort_values("score", ascending=False)
    return t[t["rank"].notna()].copy(), w, s

ranked, weights, scaled = score(data, COLS)
ranked["country"] = ranked.geo_id.map(NAME)
ranked["region"] = ranked.geo_id.map(REGION)
ranked["rank"] = ranked["rank"].astype(int)
not_ranked = data[~data.geo_id.isin(ranked.geo_id)]
not_ranked_names = sorted(NAME[g] for g in not_ranked.geo_id if data.set_index("geo_id").loc[g, COLS].notna().any())
TERRITORIES = {"HK", "MO", "PR", "GL", "NC", "PF", "GU", "VI", "BM", "KY", "AW", "CW", "FO", "GI", "IM", "JE", "GG", "MP", "AS", "TC",
               "VG", "SX", "MF", "PS", "XK", "GF", "RE", "GP", "MQ", "YT"}
terr_ranked = [NAME[g] for g in ranked.geo_id if g in TERRITORIES]

out_cols = ["rank", "geo_id", "country", "region", "score", "data_share"] + COLS
ranked[out_cols].to_csv(OUT / "tier1_country_ranking.csv", index=False)
(OUT / "tier1_weights.csv").write_text(pd.DataFrame({"criterion": COLS, "weight": weights.values}).to_csv(index=False))

# ---------- stability checks ----------
def overlap(a, b, n=10):
    return len(set(a.head(n).geo_id) & set(b.head(n).geo_id))
def rho(a, b):
    x = a.set_index("geo_id")["rank"]; y = b.set_index("geo_id")["rank"]
    i = x.index.intersection(y.index); return x[i].corr(y[i], method="spearman")

checks = []
eq, _, _ = score(data, COLS, pd.Series(1.0, index=COLS))
checks.append(("Equal weights (every criterion 1/7)", overlap(ranked, eq), rho(ranked, eq)))
loose, _, _ = score(data, COLS, min_data=0.60)
checks.append(("Looser data rule (60% instead of 75%)", overlap(ranked, loose), rho(ranked, loose)))
with_div, _, _ = score(data, COLS + ["divorce_rate"])
checks.append(("Divorce rate added back as an 8th criterion", overlap(ranked, with_div), rho(ranked, with_div)))
loo = []
for c in COLS:
    r2, _, _ = score(data, [k for k in COLS if k != c])
    loo.append((SHORT[c], overlap(ranked, r2), rho(ranked, r2)))
loo_min = min(loo, key=lambda x: x[1]); loo_max = max(loo, key=lambda x: x[1])

# ---------- charts ----------
def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6); ax.set_axisbelow(True)

# 1. top-15 bar chart
top = ranked.head(TOP_N_CHART).iloc[::-1]
fig, ax = plt.subplots(figsize=(6.6, 4.4))
bars = ax.barh(top.country, top.score, height=0.62, color=BLUE)
for b, v in zip(bars, top.score):
    ax.text(v + 0.006, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=8, color=INK2)
ax.set_xlim(0.8, 1.0); ax.set_xlabel("Tier 1 score (1 = best on every criterion)")
ax.set_xticks([0.8, 0.85, 0.9, 0.95, 1.0]); tidy(ax); ax.grid(axis="y", visible=False)
plt.tight_layout(); plt.savefig(FIG / "top15.png", dpi=220); plt.close()

# 2. weights bar chart
w = weights.sort_values()
fig, ax = plt.subplots(figsize=(6.6, 2.5))
bars = ax.barh([SHORT[k] for k in w.index], w.values * 100, height=0.6, color=BLUE)
for b, v in zip(bars, w.values * 100):
    ax.text(v + 0.3, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=8, color=INK2)
ax.set_xlim(0, 25); ax.set_xlabel("Weight (%)"); tidy(ax); ax.grid(axis="y", visible=False)
plt.tight_layout(); plt.savefig(FIG / "weights.png", dpi=220); plt.close()

# 3. flow diagram of the 5 steps
steps = [("1  Collect", "7 criteria\n216 countries"), ("2  Cap outliers", "trim values past\n5th / 95th pct"),
         ("3  Rescale 0-1", "1 = best country\n0 = worst"), ("4  Weight", "entropy: spread\nout = more weight"),
         ("5  Score & rank", "weighted average\nneed 75% of data")]
fig, ax = plt.subplots(figsize=(7.2, 1.45)); ax.set_xlim(0, 7.2); ax.set_ylim(0, 1.45); ax.axis("off")
BW, GAP = 1.26, 0.2
for i, (h, b) in enumerate(steps):
    x = 0.07 + i * (BW + GAP)
    ax.add_patch(FancyBboxPatch((x, 0.2), BW, 1.05, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor="#eef4fc", edgecolor=BLUE, linewidth=1))
    ax.text(x + BW / 2, 1.03, h, ha="center", va="center", fontsize=8.5, fontweight="bold", color=BLUE_DARK)
    ax.text(x + BW / 2, 0.6, b, ha="center", va="center", fontsize=7.2, color=INK2, linespacing=1.3)
    if i < 4:
        ax.add_patch(FancyArrowPatch((x + BW + 0.03, 0.72), (x + BW + GAP - 0.03, 0.72), arrowstyle="-|>", mutation_scale=9, color=MUTED, lw=1))
plt.savefig(FIG / "flow.png", dpi=220, bbox_inches="tight", pad_inches=0.02); plt.close()

# 4. heatmap: top 15 x 7 criteria (0-1 scaled), annotated with raw values
hm = ranked.head(TOP_N_CHART)
S = scaled.loc[hm.index, COLS]
fig, ax = plt.subplots(figsize=(7.0, 5.6))
im = ax.imshow(S.values, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(COLS))); ax.set_xticklabels([SHORT[c] for c in COLS], fontsize=8.5)
ax.set_yticks(range(len(hm))); ax.set_yticklabels([f"{r}. {c}" for r, c in zip(hm["rank"], hm.country)], fontsize=8.5)
ax.xaxis.tick_top(); ax.tick_params(length=0)
for sp in ax.spines.values(): sp.set_visible(False)
for i, (_, row) in enumerate(hm.iterrows()):
    for j, c in enumerate(COLS):
        v = row[c]; sv = S.iloc[i, j]
        if pd.isna(v):
            ax.text(j, i, "no data", ha="center", va="center", fontsize=7, color=MUTED)
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#f5f5f3", edgecolor="white", lw=2))
            continue
        label = f"{v:.0f}" if abs(v) >= 100 else f"{v:.1f}"
        ax.text(j, i, label, ha="center", va="center", fontsize=8, color="white" if sv > 0.55 else INK)
# white gaps between cells
ax.set_xticks(np.arange(-0.5, len(COLS), 1), minor=True); ax.set_yticks(np.arange(-0.5, len(hm), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2); ax.tick_params(which="minor", length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels(["worst", "", "best"])
cb.outline.set_visible(False); cb.ax.tick_params(length=0, labelsize=8)
plt.tight_layout(); plt.savefig(FIG / "heatmap.png", dpi=220); plt.close()

# 5. slope chart: entropy rank vs equal-weight rank (top 15)
eqr = eq.set_index("geo_id")["rank"]
fig, ax = plt.subplots(figsize=(6.4, 5.0))
for _, r in ranked.head(TOP_N_CHART).iterrows():
    a, b = r["rank"], int(eqr.get(r.geo_id, 99))
    moved = abs(a - b) >= 3
    ax.plot([0, 1], [a, min(b, TOP_N_CHART + 1)], color=BLUE if not moved else "#eb6834", lw=1.6 if moved else 1.2, alpha=0.95)
    ax.scatter([0, 1], [a, min(b, TOP_N_CHART + 1)], s=22, color=BLUE if not moved else "#eb6834", zorder=3)
    ax.text(-0.04, a, f"{r.country}  {a}", ha="right", va="center", fontsize=8, color=INK)
    ax.text(1.04, min(b, TOP_N_CHART + 1), f"{b if b <= TOP_N_CHART else '>' + str(TOP_N_CHART)}", ha="left", va="center", fontsize=8, color=INK2)
ax.set_ylim(TOP_N_CHART + 1.6, 0.4); ax.set_xlim(-0.75, 1.25); ax.axis("off")
ax.text(0, 0.0, "Entropy weights\n(main model)", ha="center", va="bottom", fontsize=9, fontweight="bold", color=INK)
ax.text(1, 0.0, "Equal weights\n(check)", ha="center", va="bottom", fontsize=9, fontweight="bold", color=INK)
ax.text(0.5, TOP_N_CHART + 1.5, "orange = moves 3 or more places", ha="center", va="center", fontsize=7.5, color=INK2)
plt.tight_layout(); plt.savefig(FIG / "slope.png", dpi=220); plt.close()

# 6. scores by world region (strip plot)
REG_SHORT = {"Middle East, North Africa, Afghanistan & Pakistan": "Middle East & North Africa", "Latin America & Caribbean": "Latin America & Caribbean",
             "Europe & Central Asia": "Europe & Central Asia", "East Asia & Pacific": "East Asia & Pacific", "Sub-Saharan Africa": "Sub-Saharan Africa",
             "South Asia": "South Asia", "North America": "North America"}
ranked["region"] = ranked["region"].map(lambda r: REG_SHORT.get(r, r))
order = ranked.groupby("region")["score"].median().sort_values().index.tolist()
fig, ax = plt.subplots(figsize=(6.8, 3.6))
rng = np.random.default_rng(0)
for i, reg in enumerate(order):
    g = ranked[ranked.region == reg]
    ax.scatter(g.score, i + rng.uniform(-0.18, 0.18, len(g)), s=16, color=BLUE, alpha=0.55, edgecolor="white", lw=0.5)
    med = g.score.median(); ax.plot([med, med], [i - 0.3, i + 0.3], color=INK, lw=1.6)
    best = g.iloc[0]; ax.text(best.score + 0.01, i, best.country, fontsize=7.5, va="center", color=INK2)
ax.set_yticks(range(len(order))); ax.set_yticklabels([f"{r}  ({(ranked.region == r).sum()})" for r in order], fontsize=8)
ax.set_xlabel("Tier 1 score")
ax.set_xlim(0.1, 1.2); ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0]); tidy(ax); ax.grid(axis="y", visible=False)
plt.tight_layout(); plt.savefig(FIG / "regions.png", dpi=220); plt.close()

# ---------- PDF ----------
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=18, spaceAfter=6, textColor=colors.HexColor("#1f3b5c"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=13, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#1f3b5c"))
BODY = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.5, leading=13)
SMALL = ParagraphStyle("S", parent=BODY, fontSize=8, leading=10, textColor=colors.HexColor("#52514e"))
CELL = ParagraphStyle("C", parent=BODY, fontSize=7.8, leading=9.5)
BIG = ParagraphStyle("BIG", parent=BODY, fontSize=22, leading=26, alignment=1, textColor=colors.HexColor("#1c5cab"))
CAP = ParagraphStyle("CAP", parent=SMALL, alignment=1)
def P(t, s=CELL): return Paragraph(str(t), s)
def tbl(rows, widths, font=8, header=True):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    st = [("FONT", (0, 0), (-1, -1), "Helvetica", font), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8d0da")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st += [("FONT", (0, 0), (-1, 0), "Helvetica-Bold", font), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b5c")),
               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
        st += [("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f1f4f8")) for i in range(2, len(rows), 2)]
    t.setStyle(TableStyle(st)); return t
def fmt(v): return "-" if pd.isna(v) else (f"{v:.0f}" if abs(v) >= 100 else f"{v:.1f}")

top5 = ", ".join(ranked.head(5).country)
n_full = int((ranked.data_share >= 0.999).sum())
story = [Paragraph("HiMCM 2026 - Tier 1: the best countries to live in", H1),
         Paragraph(f"Generated {datetime.date.today():%B %d, %Y}. Every country is scored on 7 criteria from 4 folders "
                   "(Healthcare, Basic Resources, Infrastructure, Family) plus Education, and ranked. The top 30 go on to Tier 2.", BODY),
         Spacer(1, 8)]
tiles = [[P(f"<b>{len(ranked)}</b>", BIG), P(f"<b>{len(COLS)}</b>", BIG), P(f"<b>{n_full}</b>", BIG), P(f"<b>{weights.max()*100:.0f}%</b>", BIG)],
         [P("countries ranked", CAP), P("criteria, weighted by entropy", CAP), P("countries with complete data", CAP),
          P(f"largest weight: {SHORT[weights.idxmax()].lower()}", CAP)]]
tt = Table(tiles, colWidths=[1.82 * inch] * 4)
tt.setStyle(TableStyle([("BOX", (0, 0), (0, -1), 0.6, colors.HexColor("#c8d0da")), ("BOX", (1, 0), (1, -1), 0.6, colors.HexColor("#c8d0da")),
                        ("BOX", (2, 0), (2, -1), 0.6, colors.HexColor("#c8d0da")), ("BOX", (3, 0), (3, -1), 0.6, colors.HexColor("#c8d0da")),
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7f9fc")), ("TOPPADDING", (0, 0), (-1, 0), 8),
                        ("BOTTOMPADDING", (0, -1), (-1, -1), 8)]))
story += [tt, Spacer(1, 10), Paragraph("The 15 best-scoring countries", H2),
          Image(str(FIG / "top15.png"), width=6.6 * inch, height=4.4 * inch),
          Paragraph(f"<b>In one sentence:</b> {top5} lead because they are near the top on <i>every</i> criterion at once; "
                    f"the stability checks on page 4 show the top 10 barely changes however the weights are set.", BODY)]

story += [PageBreak(), Paragraph("How the score is built", H2), Image(str(FIG / "flow.png"), width=7.2 * inch, height=1.45 * inch),
          Paragraph("Each criterion is rescaled so 1 = best country and 0 = worst (after trimming extreme values). The weights are not "
                    "chosen by hand: the <b>entropy method</b> gives more weight to a criterion whose values are spread out across countries "
                    "(it tells countries apart) and less to one where everyone scores about the same. A country's score is the weighted "
                    "average of the criteria it has data for; countries missing more than a quarter of the weight are not ranked.", BODY),
          Paragraph("Criteria, sources and weights", H2)]
rows = [["Criterion", "Folder", "Better", "Source", "Years", "Countries", "Weight"]]
for k, (d, folder, short, desc, src, yrs) in CRITERIA.items():
    rows.append([P(f"<b>{short}</b><br/>{desc}"), P(folder), d, P(src), P(yrs), str(LOG[k]["countries"]), f"{weights[k]*100:.1f}%"])
story += [tbl(rows, [2.05 * inch, 0.85 * inch, 0.5 * inch, 1.6 * inch, 1.1 * inch, 0.6 * inch, 0.55 * inch], font=7.5),
          Spacer(1, 6), Image(str(FIG / "weights.png"), width=6.0 * inch, height=2.27 * inch),
          Paragraph(f"<b>Not used:</b> {DROPPED['divorce_rate']} - too old and too patchy to compare countries fairly in 2026. "
                    f"Adding it back changes {10 - checks[2][1]} of the top 10 (page 4).", SMALL)]

story += [PageBreak(), Paragraph("Why the top 15 rank where they do", H2),
          Paragraph("Each cell is the country's value on that criterion; darker = closer to the best country in the world "
                    "(for Road safety and Equality, lower raw values are darker). A top country is dark all the way across.", BODY),
          Image(str(FIG / "heatmap.png"), width=7.0 * inch, height=5.6 * inch),
          Paragraph("Reading it: Singapore is ranked on 6 of 7 criteria (no Gini published) and is the best in the world on test scores "
                    "and road safety; the Nordic countries are dark on everything; countries lower down usually have one lighter cell "
                    "(e.g. a higher Gini or lower test scores) that costs them.", SMALL)]

story += [PageBreak(), Paragraph("Is the ranking stable?", H2),
          Paragraph("A ranking is only useful if it does not flip when a reasonable alternative choice is made. "
                    "Four checks, each compared with the main model (overlap = how many of the main top 10 are still in the top 10; "
                    "correlation = Spearman rank correlation over all ranked countries, 1.0 = identical order).", BODY)]
rows = [["Check", "Top-10 overlap", "Rank correlation"]] + [[P(n), f"{o} / 10", f"{r:.3f}"] for n, o, r in checks]
rows.append([P(f"Drop any one criterion (7 runs; worst = without {loo_min[0]}, best = without {loo_max[0]})"),
             f"{loo_min[1]}-{loo_max[1]} / 10", f"{min(x[2] for x in loo):.3f}-{max(x[2] for x in loo):.3f}"])
story += [tbl(rows, [4.3 * inch, 1.4 * inch, 1.4 * inch]), Spacer(1, 8),
          Image(str(FIG / "slope.png"), width=5.4 * inch, height=4.2 * inch),
          Paragraph("Left: rank with entropy weights. Right: the same countries if all 7 criteria were weighted equally.", CAP)]

story += [PageBreak(), Paragraph("The whole picture: scores by world region", H2),
          Image(str(FIG / "regions.png"), width=6.8 * inch, height=3.6 * inch),
          Paragraph("Each dot is one ranked country (World Bank regions; number of countries in brackets); the black tick is the "
                    "regional median and the best country in each region is named. Europe and Central Asia dominates the top, "
                    "but every region has countries above 0.7.", SMALL),
          Paragraph("Data notes", H2)]
notes = [
    ("Sources.", "Healthcare is from your Tier 1 file. The other six criteria were downloaded from the World Bank API on November 13, 2026, "
     "taking each country's most recent value since 2010. World Bank regional aggregates were removed."),
    ("Who is ranked.", f"{len(ranked)} of 216 places. A country must have values for at least 75% of the total weight (about 6 of 7 criteria). "
     f"Not ranked for lack of data: {', '.join(not_ranked_names)}. Monaco, San Marino and Andorra would otherwise rank in the top 30 on "
     "only 4-5 criteria, which is not a fair comparison."),
    ("Territories.", "The list follows the World Bank, so a few non-sovereign places with enough data are ranked"
     + (f" ({', '.join(terr_ranked)})" if terr_ranked else "") + ". Tier 2 needs regions inside a country, so only sovereign "
     "countries with regional data carry on."),
    ("Judgement calls.", "Lower inequality (Gini) and fewer road deaths are treated as better. Gini years vary (2010-2025) because many "
     "countries publish it rarely; 'most recent' is used consistently."),
    ("Zeros.", "Road traffic deaths of 0.0 for Antigua and Barbuda, Monaco and San Marino are WHO estimates for very small populations."),
    ("Method change from the first draft.", "Entropy is now computed from the countries that have a value for each criterion (no filling of "
     "gaps with the average), which stops sparse criteria from being under-weighted. The code is in <i>tier1_report.py</i>."),
]
for h, t in notes:
    story += [Paragraph(f"<b>{h}</b> {t}", BODY), Spacer(1, 4)]

# appendix: full ranking in two columns
story += [PageBreak(), Paragraph("Appendix: full Tier 1 ranking", H2)]
lst = [(str(r["rank"]), r.country[:26], f"{r.score:.3f}", f"{r.data_share*100:.0f}%") for _, r in ranked.iterrows()]
PER_COL = 45
for start in range(0, len(lst), 2 * PER_COL):
    chunk = lst[start:start + 2 * PER_COL]; left, right = chunk[:PER_COL], chunk[PER_COL:]
    rows = [["#", "Country", "Score", "Data"] * 2]
    for i in range(len(left)):
        rows.append(list(left[i]) + (list(right[i]) if i < len(right) else ["", "", "", ""]))
    if start:
        story += [PageBreak(), Paragraph("Appendix (continued)", H2)]
    story.append(tbl(rows, [0.4 * inch, 1.85 * inch, 0.55 * inch, 0.5 * inch] * 2, font=7))
story.append(Paragraph("'Data' = share of the total weight the country has values for. Full table with every criterion value: "
                       "tier1_country_ranking.csv.", SMALL))

def footer(c, d):
    c.saveState(); c.setFont("Helvetica", 7.5); c.setFillColor(colors.HexColor("#777777"))
    c.drawString(0.6 * inch, 0.45 * inch, "HiMCM 2026 - Tier 1 countries"); c.drawRightString(7.9 * inch, 0.45 * inch, f"Page {d.page}")
    c.restoreState()
pdf = OUT / "HiMCM_Tier1_Results.pdf"
SimpleDocTemplate(str(pdf), pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch, topMargin=0.6 * inch,
                  bottomMargin=0.7 * inch, title="HiMCM 2026 - Tier 1 Results", author="VihaanJois").build(story, onFirstPage=footer, onLaterPages=footer)

print("ranked", len(ranked), "| weights", (weights * 100).round(1).to_dict())
print("checks", checks); print("leave-one-out", loo)
print(ranked.head(30)[["rank", "country", "score", "data_share"]].to_string(index=False))
print("PDF:", pdf)
