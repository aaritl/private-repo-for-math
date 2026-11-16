"""PDF for the Tier 2 regional funnel. usage: python tier2_report.py <results dir>"""
import sys, datetime, subprocess
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

OUT = Path(sys.argv[1])
here = Path(__file__).parent
log = subprocess.run([sys.executable, str(here / "tier2_regions.py"), str(here / "oecd_raw"),
                      str(here / "run/data/places/countryInfo.txt"), str(OUT / "tier1_country_ranking.csv"), str(OUT)],
                     capture_output=True, text=True, check=True).stdout
info = {l.split(":")[0]: l.split(":", 1)[1].strip() for l in log.splitlines() if ":" in l and not l.startswith(" ")}
weights = eval(info["weights %"])
years = eval(info["years"])

import importlib.util
spec = importlib.util.spec_from_file_location("t2", here / "tier2_regions.py")
src = (here / "tier2_regions.py").read_text()
ns = {}; exec(src[src.index("CRITERIA = {"):src.index("def latest")], ns)
CRITERIA = ns["CRITERIA"]
SHORT = {"homicide_rate": "Homicide", "vehicle_theft_rate": "Vehicle theft", "pm25_exposure": "PM2.5",
         "tertiary_education": "Tertiary edu.", "transport_deaths": "Transport deaths"}

ranked = pd.read_csv(OUT / "tier2_region_ranking.csv")
winners = pd.read_csv(OUT / "tier2_regions_to_tier3.csv")
t1 = pd.read_csv(OUT / "tier1_country_ranking.csv", keep_default_na=False, na_values=[""]).head(30)
t1_rank = dict(zip(t1.country, t1["rank"]))

FIG = OUT / "fig_tier2"; FIG.mkdir(exist_ok=True)
w = pd.Series(weights).sort_values()
fig, ax = plt.subplots(figsize=(6.4, 2.6))
ax.barh([SHORT[k] for k in w.index], w.values, color="#3b6ea5")
for i, v in enumerate(w.values):
    ax.text(v + 0.4, i, f"{v:.1f}%", va="center", fontsize=8)
ax.set_xlabel("Entropy weight (%)"); ax.set_xlim(0, w.max() * 1.2); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(FIG / "weights.png", dpi=200); plt.close()
top = ranked.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.barh(top.region + " (" + top.country + ")", top.score, color="#5a9e6f")
ax.set_xlabel("Score (0 = worst, 1 = best)"); ax.set_xlim(0, 1); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(FIG / "top15.png", dpi=200); plt.close()

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=17, spaceAfter=8, textColor=colors.HexColor("#1f3b5c"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=5, textColor=colors.HexColor("#1f3b5c"))
BODY = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.5, leading=13)
SMALL = ParagraphStyle("S", parent=BODY, fontSize=8, leading=10)
CELL = ParagraphStyle("C", parent=BODY, fontSize=7.5, leading=9)
def P(t, s=CELL): return Paragraph(str(t), s)
def tbl(rows, widths, font=8):
    t = Table(rows, colWidths=widths, repeatRows=1)
    st = [("FONT", (0, 0), (-1, -1), "Helvetica", font), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", font),
          ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b5c")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
          ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8d0da")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    st += [("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f1f4f8")) for i in range(2, len(rows), 2)]
    t.setStyle(TableStyle(st)); return t
def fmt(v):
    return "-" if pd.isna(v) else (f"{v:.0f}" if abs(v) >= 100 else f"{v:.1f}" if abs(v) >= 1 else f"{v:.2f}")

story = [Paragraph("HiMCM 2026 - Tier 2: best regions inside the Tier 1 countries", H1),
         Paragraph(f"Generated {datetime.date.today():%B %d, %Y}. Tier 2 is run on the results of Tier 1: only regions "
                   "inside the top 30 Tier 1 countries are scored, and the top 5 regions in each country go on to Tier 3.", BODY),
         Spacer(1, 8)]
summary = [["Summary", ""],
           [P("<b>Countries carried from Tier 1</b>"), P(f"Top 30 of the Tier 1 ranking; {info['countries carried from Tier 1'].split('|')[0].strip()} countries")],
           [P("<b>Countries with regional data</b>"), P(info["countries carried from Tier 1"].split(":")[-1].strip())],
           [P("<b>No regions / no regional data</b>"), P(info["no regional data"] + ", Iceland (missing education and transport data at region level)")],
           [P("<b>Regions ranked (>= 60% data)</b>"), P(info["regions with any data"].split("|")[1].split(":")[1].strip())],
           [P("<b>Regions passed to Tier 3</b>"), P(f"{len(winners)} (top 5 per country; fewer where a country has fewer regions)")],
           [P("<b>Top 5 regions overall</b>"), P(", ".join(ranked.head(5).region + " (" + ranked.head(5).country + ")"))],
           [P("<b>Largest weight</b>"), P(f"{SHORT[max(weights, key=weights.get)]} ({max(weights.values()):.0f}%)")],
           [P("<b>Stability (equal weights)</b>"), P(info["equal weights"])]]
story += [tbl(summary, [2.0 * inch, 5.3 * inch])]
story += [Paragraph("How Tier 2 was run", H2), Paragraph(
    "<b>Regions:</b> OECD large regions (TL2), downloaded from the OECD Regional Statistics API (sdmx.oecd.org). "
    "TL2 regions are states, provinces or groups of counties (e.g. US states, Canadian provinces, German Lander, "
    "Japanese regions such as Southern-Kanto). <b>Model:</b> identical to <i>himcm26_simple.py</i>: values capped at the "
    "5th/95th percentile, rescaled 0-1 (flipped where lower is better), entropy weights, weighted average over the criteria "
    "a region has, ranked if it has at least 60% of the weight. All regions are scored together (so weights are shared), "
    "then the top 5 inside each country are kept.", BODY)]
rows = [["Criterion", "Folder", "Better", "Years", "Regions", "Weight"]]
for k, (d, folder, desc) in CRITERIA.items():
    rows.append([P(desc), P(folder), d, years[k], str(int(ranked[k].notna().sum())), f"{weights[k]:.1f}%"])
story += [Paragraph("Criteria and weights", H2), tbl(rows, [2.6 * inch, 1.0 * inch, 0.55 * inch, 0.9 * inch, 0.6 * inch, 0.6 * inch]),
          Spacer(1, 4), Image(str(FIG / "weights.png"), width=5.0 * inch, height=2.03 * inch)]

story += [PageBreak(), Paragraph("Top 25 regions overall", H2)]
rows = [["#", "Region", "Country", "Score", "Data"] + [SHORT[k].replace("Transport deaths", "Transport") for k in CRITERIA]]
for _, x in ranked.head(25).iterrows():
    rows.append([int(x.rank_overall), P(x.region), P(x.country), f"{x.score:.3f}", f"{x.data_share*100:.0f}%"] + [fmt(x[k]) for k in CRITERIA])
story += [tbl(rows, [0.3 * inch, 1.55 * inch, 1.0 * inch, 0.45 * inch, 0.4 * inch] + [0.72 * inch] * 5, font=7),
          Paragraph("'-' = no data. Regions with less than 100% data are scored on the criteria they have.", SMALL),
          Spacer(1, 6), Image(str(FIG / "top15.png"), width=4.6 * inch, height=3.16 * inch)]

story += [PageBreak(), Paragraph("Regions passed to Tier 3 (top 5 per country, by Tier 1 rank)", H2)]
rows = [["T1 #", "Country", "#", "Region", "Score", "Data"]]
order = sorted(winners.country.unique(), key=lambda c: t1_rank.get(c, 99))
for c in order:
    for _, x in winners[winners.country == c].sort_values("rank_in_country").iterrows():
        rows.append([int(t1_rank[c]), P(c), int(x.rank_in_country), P(x.region), f"{x.score:.3f}", f"{x.data_share*100:.0f}%"])
story.append(tbl(rows, [0.45 * inch, 1.5 * inch, 0.3 * inch, 2.6 * inch, 0.6 * inch, 0.5 * inch], font=7.5))

story += [PageBreak(), Paragraph("Data notes", H1)]
for head, text in [
    ("Why these criteria.", "They match the Tier 2 folders (Crime, Environment, Education, Transportation) but at region level. "
     "Your Tier 2 files (UNODC crime, PM2.5 by country) are national, so they cannot tell regions inside a country apart."),
    ("Crime.", "Homicide is the most comparable crime statistic across countries (bodies are almost always recorded). Vehicle theft "
     "is also fairly consistently reported because of insurance claims."),
    ("Transportation.", "The OECD has no region-level public-transport access, so deaths from transport accidents are used "
     "(transport safety). SDG 11.2.1 (your file) is city-level and is better used in Tier 3."),
    ("Education.", "Share of adults 25-64 with tertiary (college) education. It gets the largest weight (35%) because it differs "
     "most between regions; capital regions score high on it."),
    ("City-states and microstates.", "Monaco, Singapore, Macao, San Marino, Andorra and the Bahamas have no OECD regions. They are "
     "effectively one city each and can go straight to Tier 3 if you want to keep them."),
    ("Iceland.", "Its 2 regions have only homicide, vehicle theft and PM2.5 (51% of the weight), below the 60% minimum."),
    ("Mayotte.", "France's 5th region is Mayotte, an overseas department scored on 65% of the data (no education value). "
     "Consider excluding overseas regions."),
    ("Single-region countries.", "Estonia and Malta are one TL2 region each, so the region is the whole country."),
]:
    story += [Paragraph(f"<b>{head}</b> {text}", BODY), Spacer(1, 5)]

def footer(c, d):
    c.saveState(); c.setFont("Helvetica", 7.5); c.setFillColor(colors.HexColor("#777777"))
    c.drawString(0.6 * inch, 0.45 * inch, "HiMCM 2026 - Tier 2 regions"); c.drawRightString(7.9 * inch, 0.45 * inch, f"Page {d.page}")
    c.restoreState()
pdf = OUT / "HiMCM_Tier2_Regions.pdf"
SimpleDocTemplate(str(pdf), pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch, topMargin=0.6 * inch,
                  bottomMargin=0.7 * inch, title="HiMCM 2026 - Tier 2 Regions", author="VihaanJois").build(story, onFirstPage=footer, onLaterPages=footer)
print("PDF:", pdf)
