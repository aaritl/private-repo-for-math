"""Place every OECD city (FUA) inside a Tier 2 region.

1. find the city's main town in GeoNames (same country, matching name) -> latitude/longitude
2. Europe: point-in-polygon on official NUTS boundaries (codes match the OECD TL2 codes)
   AU/CA/NZ/JP/KR: GeoNames state/prefecture -> OECD TL2 region (lookup table below)
usage: python tier3_locate.py <oecd_city dir> <geo dir> <countryInfo.txt> <out csv>
"""
import sys, json, re, unicodedata
from pathlib import Path
import pandas as pd
from shapely.geometry import shape, Point
from shapely.strtree import STRtree

CITY, GEO, GEONAMES_INFO, OUT = map(Path, sys.argv[1:5])

# ---------- every FUA with a name ----------
frames = []
for f in CITY.glob("DSD_FUA_*.csv"):
    d = pd.read_csv(f, usecols=lambda c: c in ("REF_AREA", "Reference area"), low_memory=False)
    if "REF_AREA" in d: frames.append(d)
fua = pd.concat(frames).drop_duplicates("REF_AREA")
fua = fua[fua.REF_AREA.str.endswith("F")].rename(columns={"REF_AREA": "fua_code", "Reference area": "fua_name"})

info = pd.read_csv(GEONAMES_INFO, sep="\t", comment="#", header=None, usecols=[0, 1], names=["iso2", "iso3"], dtype=str, keep_default_na=False)
ISO3_2 = dict(zip(info.iso3, info.iso2))
fua["iso2"] = [ISO3_2.get(c[:3]) if c[:3] in ISO3_2 and not c[2].isdigit() else c[:2] for c in fua.fua_code]
fua["iso2"] = fua.iso2.replace({"UK": "GB", "EL": "GR"})

# ---------- GeoNames towns ----------
cols = ["gid", "name", "ascii", "alt", "lat", "lon", "fclass", "fcode", "cc", "cc2", "a1", "a2", "a3", "a4", "pop", "elev", "dem", "tz", "mod"]
gn = pd.read_csv(GEO / "cities15000.txt", sep="\t", header=None, names=cols, keep_default_na=False, low_memory=False,
                 dtype={"a1": str, "cc": str})
def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()
gn = gn[gn.fcode != "PPLX"]   # drop neighbourhoods/sections of a city (e.g. "Salamanca" district of Madrid)
index = {}
for r in gn.itertuples():
    for n in {r.name, r.ascii, *str(r.alt).split(",")}:
        k = (r.cc, norm(n))
        if k[1] and (k not in index or r.pop > index[k][2]):
            index[k] = (r.lat, r.lon, r.pop, r.a1, r.name)

def candidates(name):
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"^(greater|grand|gross|groß|metropolitan|agglomeration)\s+", "", name, flags=re.I).strip()
    out = [name]
    out += re.split(r"\s*[-/–,]\s*|\s+and\s+", name)        # "Gosford - Springfield" -> both parts
    out += [name.split()[0]] if " " in name else []
    return [norm(x) for x in out if norm(x)]

# FUAs named after a district/region rather than a town: main town's coordinates (lat, lon, town)
MANUAL = {"DE006F": (51.456, 7.012, "Essen"), "UK002F": (52.481, -1.900, "Birmingham"), "SI001F": (46.051, 14.506, "Ljubljana"),
          "UK038F": (52.478, 1.751, "Lowestoft"), "UK043F": (52.806, -1.637, "Burton upon Trent"), "UK062F": (53.362, -2.734, "Widnes"),
          "UK513F": (51.380, 0.522, "Chatham"), "UK543F": (53.567, -0.080, "Grimsby"), "UK556F": (51.753, -0.448, "Hemel Hempstead"),
          "UK561F": (50.462, -3.525, "Torquay"), "UK580F": (51.293, -0.755, "Farnborough"), "FR077F": (47.590, 7.560, "Saint-Louis"),
          "FR062F": (-12.780, 45.228, "Mamoudzou"), "FR046F": (14.616, -61.059, "Fort-de-France"), "FR072F": (4.933, -52.330, "Cayenne")}
hits = []
for r in fua.itertuples():
    if r.fua_code in MANUAL:
        la, lo, town = MANUAL[r.fua_code]; hits.append((la, lo, None, None, town)); continue
    found = None
    for c in candidates(r.fua_name):
        found = index.get((r.iso2, c))
        if found: break
    hits.append(found or (None, None, None, None, None))
fua[["lat", "lon", "town_pop", "admin1", "matched_town"]] = pd.DataFrame(hits, index=fua.index)

# ---------- region assignment ----------
def load(path):
    fs = json.load(open(path))["features"]
    return [(f["properties"]["NUTS_ID"], shape(f["geometry"])) for f in fs]
polys = load(GEO / "nuts_2024_L1.geojson") + load(GEO / "nuts_2024_L2.geojson") + \
        [p for p in load(GEO / "nuts_2016_L1.geojson") if p[0].startswith("UK")] + \
        [p for p in load(GEO / "nuts_2016_L2.geojson") if p[0].startswith("UK")]
tree = STRtree([g for _, g in polys])

def nuts_codes(lat, lon):
    p = Point(lon, lat)
    idx = tree.query(p)
    inside = [polys[i][0] for i in idx if polys[i][1].contains(p)]
    if not inside:   # towns on the coast can fall just outside the simplified outline: nearest polygon
        i = tree.nearest(p); inside = [polys[i][0]]
    return inside

ADMIN = {  # GeoNames state/prefecture name -> OECD TL2 code
    "AU": {"Australian Capital Territory": "AU8", "New South Wales": "AU1", "Victoria": "AU2", "Queensland": "AU3",
           "South Australia": "AU4", "Western Australia": "AU5", "Tasmania": "AU6", "Northern Territory": "AU7"},
    "CA": {"Newfoundland and Labrador": "CA10", "Prince Edward Island": "CA11", "Nova Scotia": "CA12", "New Brunswick": "CA13",
           "Quebec": "CA24", "Ontario": "CA35", "Manitoba": "CA46", "Saskatchewan": "CA47", "Alberta": "CA48", "British Columbia": "CA59"},
    "NZ": {"Auckland": "NZ12", "Waikato Region": "NZ13", "Bay of Plenty": "NZ14", "Wellington Region": "NZ19", "Canterbury": "NZ23"},
    "JP": {**{p: "JPA" for p in ["Hokkaido"]},
           **{p: "JPD" for p in ["Saitama", "Chiba", "Tokyo", "Kanagawa"]},
           **{p: "JPG" for p in ["Shiga", "Kyoto", "Osaka", "Hyogo", "Nara", "Wakayama"]},
           **{p: "JPH" for p in ["Tottori", "Shimane", "Okayama", "Hiroshima", "Yamaguchi"]},
           **{p: "JPI" for p in ["Tokushima", "Kagawa", "Ehime", "Kochi"]}},
    "KR": {**{p: "KR01" for p in ["Seoul", "Incheon", "Gyeonggi-do"]},
           **{p: "KR02" for p in ["Busan", "Ulsan", "Gyeongsangnam-do"]},
           **{p: "KR03" for p in ["Daegu", "Gyeongsangbuk-do"]},
           **{p: "KR05" for p in ["Daejeon", "Sejong-si", "North Chungcheong", "Chungcheongnam-do"]},
           **{p: "KR07" for p in ["Jeju-do"]}},
}
a1 = pd.read_csv(GEO / "admin1CodesASCII.txt", sep="\t", header=None, names=["code", "name", "ascii", "gid"], keep_default_na=False)
A1NAME = dict(zip(a1.code, a1.ascii))

regions = []
for r in fua.itertuples():
    if pd.isna(r.lat): regions.append(""); continue
    if r.iso2 in ADMIN:
        regions.append(ADMIN[r.iso2].get(A1NAME.get(f"{r.iso2}.{r.admin1}", ""), ""))
    else:
        regions.append("|".join(nuts_codes(r.lat, r.lon)))
fua["nuts_or_tl2"] = regions
fua.to_csv(OUT, index=False)
print("FUAs:", len(fua), "| located:", fua.lat.notna().sum())
