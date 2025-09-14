# streamlit_app.py
from __future__ import annotations
import os, io, math, json, textwrap, shutil
from typing import Dict, List, Tuple, Optional

import requests
import streamlit as st
from PIL import Image, ImageDraw
# PDF
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics

# -------------------------
# App config
# -------------------------
APP_PASSWORD = "Flogas2025"
APP_TITLE = "LPG Customer Tank — Pre-Check"

# -------------------------
# CoP thresholds & vehicle
# -------------------------
CoP = {
    "to_building_m": 3.0, "to_boundary_m": 3.0, "to_ignition_m": 3.0, "to_drain_m": 3.0,
    "overhead_info_m": 10.0, "overhead_block_m": 5.0, "rail_attention_m": 30.0,
    "poi_radius_m": 400.0, "wind_stagnant_mps": 1.0, "slope_attention_pct": 3.0,
    "approach_grade_warn_pct": 18.0, "route_vs_crowfly_ratio_warn": 1.7,
}

VEHICLES = {
    "Mini-bulker (default)": {"max_height_m": 3.6, "max_width_m": 2.55, "gross_weight_t": 18.0},
    "Rigid HGV (medium)":    {"max_height_m": 4.0, "max_width_m": 2.55, "gross_weight_t": 26.0},
}
DEFAULT_VEHICLE = "Mini-bulker (default)"

# -------------------------
# Secrets
# -------------------------
W3W_API_KEY    = st.secrets.get("W3W_API_KEY", "")
MAPBOX_TOKEN   = st.secrets.get("MAPBOX_TOKEN", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")  # optional

# -------------------------
# Utility functions
# -------------------------
def meters_per_degree(lat_deg: float) -> Tuple[float, float]:
    lat = math.radians(lat_deg)
    return (
        111132.92 - 559.82 * math.cos(2 * lat) + 1.175 * math.cos(4 * lat),
        111412.84 * math.cos(lat) - 93.5 * math.cos(3 * lat),
    )

def ll_to_xy(lat0, lon0, lat, lon):
    mlat, mlon = meters_per_degree(lat0)
    return (lon - lon0) * mlon, (lat - lat0) * mlat

def dist_line(lat0, lon0, line: List[Tuple[float,float]]) -> Optional[float]:
    if not line or len(line) < 2:
        return None
    px, py = 0.0, 0.0
    verts = [ll_to_xy(lat0, lon0, la, lo) for la, lo in line]
    best = None
    for (ax, ay), (bx, by) in zip(verts, verts[1:]):
        apx, apy = px - ax, py - ay
        abx, aby = bx - ax, by - ay
        ab2 = abx * abx + aby * aby
        t = 0.0 if ab2 == 0 else max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
        cx, cy = ax + t * abx, ay + t * aby
        d = math.hypot(px - cx, py - cy)
        best = d if best is None else min(best, d)
    return best

def dist_poly(lat0, lon0, poly: List[Tuple[float,float]]) -> Optional[float]:
    if not poly or len(poly) < 2:
        return None
    return dist_line(lat0, lon0, poly + poly[:1])

def _dist_m(lat0, lon0, lat1, lon1):
    mlat, mlon = meters_per_degree(lat0)
    return math.hypot((lon1 - lon0) * mlon, (lat1 - lat0) * mlat)

# -------------------------
# External APIs
# -------------------------
def w3w_to_latlon(words: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        r = requests.get(
            "https://api.what3words.com/v3/convert-to-coordinates",
            params={"words": words, "key": W3W_API_KEY}, timeout=15
        )
        if r.status_code == 200:
            c = r.json().get("coordinates", {})
            return c.get("lat"), c.get("lng")
    except Exception:
        pass
    return None, None

def reverse_geocode(lat, lon) -> Dict:
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "jsonv2"},
            headers={"User-Agent": "LPG-Precheck"}, timeout=15
        )
        if r.status_code == 200:
            j = r.json(); a = j.get("address") or {}
            return {
                "display_name": j.get("display_name"),
                "road": a.get("road"),
                "postcode": a.get("postcode"),
                "city": a.get("town") or a.get("city") or a.get("village"),
                "county": a.get("county"),
                "state_district": a.get("state_district"),
                "local_authority": a.get("municipality") or a.get("county") or a.get("state_district"),
            }
    except Exception:
        pass
    return {}

OVERPASS = "https://overpass-api.de/api/interpreter"
UA = {"User-Agent": "LPG-Precheck-Pro/2.0"}

def overpass(lat, lon, r) -> Dict:
    q = f"""
[out:json][timeout:60];
(
  way(around:{r},{lat},{lon})["building"];
  relation(around:{r},{lat},{lon})["building"];

  way(around:{r},{lat},{lon})["highway"];
  node(around:{r},{lat},{lon})["man_made"="manhole"];
  node(around:{r},{lat},{lon})["manhole"];
  way(around:{r},{lat},{lon})["waterway"="drain"];
  way(around:{r},{lat},{lon})["tunnel"="culvert"];

  way(around:{r},{lat},{lon})["power"="line"];
  node(around:{r},{lat},{lon})["power"~"tower|pole"];

  way(around:{r},{lat},{lon})["railway"]["railway"!="abandoned"]["railway"!="disused"];
  way(around:{r},{lat},{lon})["waterway"~"river|stream|ditch"];
  way(around:{r},{lat},{lon})["natural"="water"];

  way(around:{r},{lat},{lon})["landuse"];

  /* Road restrictions we care about for access */
  way(around:{r},{lat},{lon})["maxheight"];
  way(around:{r},{lat},{lon})["maxwidth"];
  way(around:{r},{lat},{lon})["maxweight"];
  way(around:{r},{lat},{lon})["hgv"];
  way(around:{r},{lat},{lon})["access"];
  way(around:{r},{lat},{lon})["oneway"];
  way(around:{r},{lat},{lon})["surface"];
  way(around:{r},{lat},{lon})["smoothness"];
);
out tags geom;
"""
    try:
        r = requests.post(OVERPASS, data={"data": q}, headers=UA, timeout=90)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.info(f"Overpass note: {e}")
        return {"elements": []}

def parse_num(s):
    if s is None: return None
    s = str(s).strip().lower()
    for u in ("m","meter","metre","meters","metres","t","ton","tons","tonne","tonnes"):
        if s.endswith(u):
            s = s[:-len(u)].strip()
            break
    s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return None

def restriction_notes(ways, vehicle) -> List[str]:
    out = []
    for w in ways:
        t = w.get("tags", {})
        h = parse_num(t.get("maxheight"))
        wdt = parse_num(t.get("maxwidth"))
        wt = parse_num(t.get("maxweight"))
        if (h is not None) and (h < vehicle["max_height_m"]): out.append(f"Max height {h} m")
        if (wdt is not None) and (wdt < vehicle["max_width_m"]): out.append(f"Max width {wdt} m")
        if (wt is not None) and (wt < vehicle["gross_weight_t"]): out.append(f"Max weight {wt} t")
        if (t.get("hgv") or "").lower() in ("no","destination"): out.append(f"HGV {t.get('hgv').lower()}")
        if (t.get("access") or "").lower() in ("no","private"): out.append(f"Access {t.get('access').lower()}")
        if (t.get("oneway") or "").lower() == "yes": out.append("One-way")
    # unique
    seen, out2 = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            out2.append(s)
    return out2

def surface_info(ways) -> Dict:
    risky = 0; samples = []
    for w in ways:
        t = w.get("tags", {})
        surf = (t.get("surface") or "").lower()
        smooth = (t.get("smoothness") or "").lower()
        if any(k in surf for k in ("gravel","ground","dirt","grass","unpaved","compacted","sand")): risky += 1
        if any(k in smooth for k in ("bad","very_bad","horrible","impassable")): risky += 1
        if surf or smooth: samples.append(f"{surf or 'n/a'}/{smooth or 'n/a'}")
    return {"risky_count": risky, "samples": samples[:8]}

def parse_osm(lat0, lon0, data) -> Dict:
    bpolys, roads, drains, manholes, plines, pnodes, rails, wlines, wpolys, land_polys = [],[],[],[],[],[],[],[],[],[]
    rest_ways, surf_ways = [],[]

    for el in data.get("elements", []):
        t = el.get("type"); tags = el.get("tags", {}) or {}
        geom = el.get("geometry"); coords = [(g["lat"], g["lon"]) for g in (geom or [])]

        if tags.get("building") and t in ("way","relation"): bpolys.append(coords)
        elif tags.get("highway") and t == "way":
            roads.append(coords)
            if any(k in tags for k in ("maxheight","maxwidth","maxweight","hgv","access","oneway")):
                rest_ways.append({"tags":tags, "coords":coords})
            if ("surface" in tags) or ("smoothness" in tags):
                surf_ways.append({"tags":tags, "coords":coords})
        elif t == "way" and (tags.get("waterway") == "drain" or tags.get("tunnel") == "culvert"): drains.append(coords)
        elif t == "node" and (tags.get("man_made") == "manhole" or "manhole" in tags): manholes.append((el.get("lat"), el.get("lon")))
        elif t == "way" and tags.get("power") == "line": plines.append(coords)
        elif t == "node" and tags.get("power") in ("tower","pole"): pnodes.append((el.get("lat"), el.get("lon")))
        elif t == "way" and tags.get("railway") and tags.get("railway") not in ("abandoned","disused"): rails.append(coords)
        elif t == "way" and tags.get("waterway") in ("river","stream","ditch"): wlines.append(coords)
        elif t == "way" and tags.get("natural") == "water": wpolys.append(coords)
        elif t in ("way","relation") and tags.get("landuse"): land_polys.append({"tag":tags.get("landuse"), "coords":coords})

    # Defensive filtering (prevents TypeErrors)
    valid_bpolys = [p for p in bpolys if p and len(p) >= 2]
    valid_roads  = [l for l in roads  if l and len(l) >= 2]
    valid_drains = [l for l in drains if l and len(l) >= 2]
    valid_plines = [l for l in plines if l and len(l) >= 2]
    valid_rails  = [l for l in rails  if l and len(l) >= 2]
    valid_wlines = [l for l in wlines if l and len(l) >= 2]
    valid_wpolys = [p for p in wpolys if p and len(p) >= 2]

    d_build = min([dist_poly(lat0, lon0, p) for p in valid_bpolys] or [None])
    d_road  = min([dist_line(lat0, lon0, l) for l in valid_roads] or [None])
    d_drain = min(([dist_line(lat0, lon0, l) for l in valid_drains] +
                   ([dist_line(lat0, lon0, [(la,lo),(la,lo)]) for la,lo in manholes] if manholes else [])) or [None])
    d_over  = min(([dist_line(lat0, lon0, l) for l in valid_plines] +
                   ([dist_line(lat0, lon0, [(la,lo),(la,lo)]) for la,lo in pnodes] if pnodes else [])) or [None])
    d_rail  = min([dist_line(lat0, lon0, l) for l in valid_rails] or [None])
    d_water = min(([dist_line(lat0, lon0, l) for l in valid_wlines] +
                   [dist_poly(lat0, lon0, p) for p in valid_wpolys]) or [None])

    land_counts = {}
    for lp in land_polys:
        tag = lp["tag"]; land_counts[tag] = land_counts.get(tag, 0) + 1
    if land_counts:
        top = max(land_counts, key=lambda k: land_counts[k])
        if top in ("residential","commercial","retail"): land_class = "Domestic/Urban"
        elif top in ("industrial","industrial;retail"):   land_class = "Industrial"
        else: land_class = "Rural/Agricultural"
    else:
        land_class = "Domestic/Urban" if len(valid_bpolys) > 80 else ("Rural/Agricultural" if len(valid_bpolys) < 20 else "Mixed")

    # First road line (for approach grade sampling; optional)
    nearest_road_line = valid_roads[0] if valid_roads else None

    return {
        "d_building_m": None if d_build is None else round(d_build, 1),
        "d_road_m":     None if d_road  is None else round(d_road, 1),
        "d_drain_m":    None if d_drain is None else round(d_drain, 1),
        "d_overhead_m": None if d_over  is None else round(d_over, 1),
        "d_rail_m":     None if d_rail  is None else round(d_rail, 1),
        "d_water_m":    None if d_water is None else round(d_water, 1),
        "land_class": land_class,
        "restrictions": rest_ways,
        "surfaces":     surf_ways,
        "nearest_road_line": nearest_road_line,
    }

# -------------------------
# Map box (static) with rings
# -------------------------
def fetch_map(lat, lon, zoom=17, size=(1000, 750)):
    if not MAPBOX_TOKEN:
        return None
    try:
        w, h = size
        marker = f"pin-l+f30({lon},{lat})"; style = "light-v11"
        url = (f"https://api.mapbox.com/styles/v1/mapbox/{style}/static/"
               f"{marker}/{lon},{lat},{zoom},0/{w}x{h}?access_token={MAPBOX_TOKEN}")
        r = requests.get(url, timeout=15); r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGBA")
        return img
    except Exception:
        return None

def overlay_rings(img: Image.Image, lat, zoom=17):
    if img is None: return None
    def mpp(lat,zoom): return 156543.03392*math.cos(math.radians(lat))/(2**zoom)
    scale = mpp(lat,zoom)
    cx, cy = img.width//2, img.height//2
    d = ImageDraw.Draw(img, "RGBA")
    for r, col in ((3,(220,0,0,180)), (6,(255,140,0,160))):
        px = max(1, int(r/scale))
        d.ellipse((cx-px,cy-px,cx+px,cy+px), outline=col, width=4)
    return img

def make_map_card(words, lat, lon):
    img = fetch_map(lat,lon)
    if img is None:
        # placeholder
        img = Image.new("RGBA",(1000,750),(245,247,250,255))
        d = ImageDraw.Draw(img)
        d.text((20,20),"Map unavailable", fill=(80,80,80))
    img = overlay_rings(img, lat, 17)
    out = f"map_{words.replace('.','_')}.png"
    img.save(out)
    return out

# -------------------------
# Streamlit UI helpers
# -------------------------
st.set_page_config(page_title="LPG Pre-Check", page_icon="icon.png", layout="wide")
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

def header_with_icon(title: str):
    c1, c2 = st.columns([0.08, 0.92])
    with c1:
        try:
            img = Image.open("icon.png")
            st.image(img, use_container_width=True)
        except Exception:
            st.write("")
    with c2:
        st.title(title)

def auth_gate():
    header_with_icon("LPG Pre-Check — Access")
    st.caption("Step 1: Enter the access password to continue.")
    pwd = st.text_input("Password", type="password", placeholder="Enter password…")
    if st.button("Unlock", type="primary"):
        if pwd == APP_PASSWORD:
            st.session_state.auth_ok = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

if not st.session_state.auth_ok:
    auth_gate()

header_with_icon(APP_TITLE)

# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.subheader("Location")
    words = st.text_input("what3words (word.word.word)", value=st.session_state.get("w3w",""))
    vehicle_name = st.selectbox("Vehicle", list(VEHICLES.keys()), index=list(VEHICLES.keys()).index(DEFAULT_VEHICLE))
    run = st.button("Run Pre-Check", type="primary", use_container_width=True)

# -------------------------
# Fetch auto data
# -------------------------
auto = {}
lat = lon = None
addr = {}
if run and words.strip():
    with st.status("Fetching site data…", expanded=True) as status:
        la, lo = w3w_to_latlon(words.strip().lstrip("/"))
        if la is None:
            st.error("what3words lookup failed."); st.stop()
        lat, lon = la, lo
        status.update(label="Reverse geocoding…")
        addr = reverse_geocode(lat, lon)
        status.update(label="Reading OSM features…")
        osm = overpass(lat, lon, int(CoP["poi_radius_m"]))
        feats = parse_osm(lat, lon, osm)
        # simple environment placeholders (wire real services if you wish)
        auto = {
            "building_m": feats["d_building_m"],
            "boundary_m": None,  # user-provided
            "road_m": feats["d_road_m"],
            "drain_m": feats["d_drain_m"],
            "overhead_m": feats["d_overhead_m"],
            "rail_m": feats["d_rail_m"],
            "water_m": feats["d_water_m"],
            "land_class": feats["land_class"],

            "wind_mps": 6.8, "wind_deg": 191, "slope_pct": 3.5,
            "approach_avg": 0.9, "approach_max": 3.5,
            "rr": None,

            "restrictions": feats["restrictions"],
            "surfaces": feats["surfaces"],
        }
        st.session_state["w3w"] = words.strip()
        st.session_state["auto"] = auto
        st.session_state["addr"] = addr
        st.session_state["latlon"] = (lat, lon)
        st.session_state["vehicle"] = vehicle_name
        status.update(label="Auto data ready.", state="complete")

# Use cached
auto = st.session_state.get("auto", {})
addr = st.session_state.get("addr", {})
latlon = st.session_state.get("latlon", (None,None))
lat, lon = latlon
vehicle_name = st.session_state.get("vehicle", DEFAULT_VEHICLE)
vehicle = VEHICLES[vehicle_name]

# -------------------------
# Form widgets
# -------------------------
def distance_field(label: str, key_prefix: str, auto_val: Optional[float], max_val=2000.0) -> Optional[float]:
    col_val, col_unknown = st.columns([0.9, 0.1])
    with col_unknown:
        unknown = st.checkbox("Not mapped", key=f"{key_prefix}_unknown", value=(auto_val is None))
    with col_val:
        disabled = unknown
        v0 = 0.0 if auto_val is None else float(auto_val)
        value = st.number_input(label, min_value=0.0, max_value=float(max_val), step=0.1, value=v0, disabled=disabled, key=f"{key_prefix}_val")
    return None if unknown else float(value)

st.markdown("### Edit & confirm")
with st.form("inputs"):
    st.subheader("Environment & approach")
    e1, e2, e3 = st.columns(3)
    with e1:
        wind_mps = st.number_input("Wind (m/s)", min_value=0.0, max_value=60.0, step=0.1, value=float(auto.get("wind_mps", 0.0)))
    with e2:
        wind_deg = st.number_input("Wind dir (°)", min_value=0, max_value=359, step=1, value=int(auto.get("wind_deg", 0)))
    with e3:
        slope_pct = st.number_input("Slope (%)", min_value=0.0, max_value=100.0, step=0.1, value=float(auto.get("slope_pct", 0.0)))

    a1, a2, a3 = st.columns(3)
    with a1:
        approach_avg = st.number_input("Approach avg (%)", min_value=0.0, max_value=100.0, step=0.1, value=float(auto.get("approach_avg", 0.0)))
    with a2:
        approach_max = st.number_input("Approach max (%)", min_value=0.0, max_value=100.0, step=0.1, value=float(auto.get("approach_max", 0.0)))
    with a3:
        rr_default = "" if auto.get("rr") is None else f"{auto['rr']:.2f}"
        rr_str = st.text_input("Route indirectness (× crow-fly) — optional", value=rr_default, placeholder="leave blank")
        try:
            route_ratio = float(rr_str) if rr_str.strip() else None
        except:
            route_ratio = None

    st.markdown("---")
    st.subheader("Separations (~400 m)")

    s1, s2 = st.columns(2)
    with s1:
        building_m = distance_field("Building (m)", "d_building_m", auto.get("building_m"))
        road_m     = distance_field("Road/footpath (m)", "d_road_m", auto.get("road_m"))
        overhead_m = distance_field("Overhead power lines (m)", "d_overhead_m", auto.get("overhead_m"))
        water_m    = distance_field("Watercourse (m)", "d_water_m", auto.get("water_m"))
    with s2:
        boundary_m = distance_field("Boundary (m)", "d_boundary_m", auto.get("boundary_m"))
        drain_m    = distance_field("Drain/manhole (m)", "d_drain_m", auto.get("drain_m"))
        rail_m     = distance_field("Railway (m)", "d_rail_m", auto.get("rail_m"))
        land_use   = st.selectbox("Land use", ["Domestic/Urban","Industrial","Mixed","Rural/Agricultural"], index=["Domestic/Urban","Industrial","Mixed","Rural/Agricultural"].index(auto.get("land_class","Domestic/Urban")))

    st.markdown("---")
    st.subheader("Site options")
    v1, v2 = st.columns([0.4,0.6])
    with v1:
        veg_3m = st.slider("Vegetation within 3 m of tank (0 none → 3 heavy)", 0, 3, 1)
        los_issue = st.checkbox("Restricted line-of-sight at stand", value=False)
    with v2:
        notes_txt = st.text_input("Notes (vegetation / sightlines / special instructions)", value="")

    submitted = st.form_submit_button("Confirm & assess", type="primary")

# -------------------------
# Risk scoring
# -------------------------
def risk_score(values: Dict, access: Dict, veg_3m: int, los_issue: bool) -> Dict:
    score = 0.0; why = []

    def add(x, msg): 
        nonlocal score; score += x; why.append(msg)

    def penal(dist, lim, msg, base=18, per=6, cap=40):
        if dist is None or dist >= lim: return
        pts = min(cap, base + per*(lim - dist)); add(pts, f"{msg} below {lim} m (≈ {dist:.1f} m)")

    penal(values["building_m"], CoP["to_building_m"], "Below 3.0 m")
    penal(values["road_m"],     CoP["to_ignition_m"], "Ignition proxy (road/footpath)")
    penal(values["drain_m"],    CoP["to_drain_m"],    "Drain/manhole <3 m")

    d_ov = values["overhead_m"]
    if d_ov is not None and d_ov < CoP["overhead_block_m"]: add(28,"Overhead in no-go band")
    elif d_ov is not None and d_ov < CoP["overhead_info_m"]: add(10,"Overhead within 10 m")

    d_rail = values["rail_m"]
    if d_rail is not None and d_rail < CoP["rail_attention_m"]: add(10,"Railway within 30.0 m")
    if values["water_m"] is not None and values["water_m"] < 50: add(8,"Watercourse within 50 m")

    if values["wind_mps"] is not None and values["wind_mps"] < CoP["wind_stagnant_mps"]: add(6,f"Low wind {values['wind_mps']:.1f} m/s")
    if values["slope_pct"] is not None and values["slope_pct"] >= CoP["slope_attention_pct"]: add(8, f"Local slope {values['slope_pct']:.1f}%")
    if values["approach_max"] is not None and values["approach_max"] >= CoP["approach_grade_warn_pct"]: add(12,f"Steep approach (max {values['approach_max']:.1f}%)")
    if values["route_ratio"] is not None and values["route_ratio"] > CoP["route_vs_crowfly_ratio_warn"]: add(10, f"Route length ≫ crow-fly ({values['route_ratio']:.2f}×)")

    if veg_3m >= 2: add(6,"Vegetation near tank (≤3 m)")
    if los_issue: add(8,"Restricted line-of-sight at stand")

    # Access flags
    if access["notes"]: add(min(12, 4*len(access["notes"])), "Access restrictions: "+", ".join(access["notes"]))
    if access["surface"]["risky_count"]>0: add(min(10,2*access["surface"]['risky_count']), f"Surface flags={access['surface']['risky_count']}")

    score = round(min(100.0, score), 1)
    status = "PASS" if score < 20 else ("ATTENTION" if score < 50 else "BLOCKER")
    return {"score":score,"status":status,"explain":why[:7]}

def make_controls(rs_status: str) -> List[str]:
    out = [
        "Use a trained banksman during manoeuvres and reversing.",
        "Add temporary cones/signage; consider a convex mirror or visibility aids.",
        "Plan approach/egress to avoid reversing where practicable."
    ]
    if rs_status != "PASS":
        out.append("Confirm separations to CoP1; protect drains/manholes within 3 m; manage vegetation.")
    return out

# -------------------------
# AI commentary (optional online; offline fallback)
# -------------------------
def ai_sections(context: Dict) -> Dict[str,str]:
    offline = {
        "Safety Risk Profile": f"Local slope {context['slope_pct']:.1f}% (aspect not derived). Key separations (m): "
                               f"bldg {context['building_m'] or 'n/a'}, boundary {context['boundary_m'] or 'n/a'}, "
                               f"road {context['road_m'] or 'n/a'}, drain {context['drain_m'] or 'n/a'}, "
                               f"overhead {context['overhead_m'] or 'n/a'}, rail {context['rail_m'] or 'n/a'}. "
                               f"Wind {context['wind_mps']:.1f} m/s from {context['wind_deg']}°. Heuristic "
                               f"{context['risk']['score']}/100 → {context['risk']['status']}.",
        "Environmental Considerations": f"Flood Low/unknown (watercourse ~{context['water_m'] or 'n/a'} m). "
                               "Protect drains during transfers; control vegetation.",
        "Access & Logistics": f"Approach avg/max {context['approach_avg']:.1f}/{context['approach_max']:.1f}%. "
                               "Validate signage/restrictions; ensure sound hardstanding.",
        "Overall Site Suitability": "Site appears suitable with routine controls where PASS; when ATTENTION/BLOCKER, "
                                    "address separations, overheads, drains, and sightlines before delivery.",
    }
    if not OPENAI_API_KEY:
        return offline

    try:
        prompt = f"""
Act as an LPG siting assessor. Write four concise sections:
[1] Safety Risk Profile
[2] Environmental Considerations
[3] Access & Logistics
[4] Overall Site Suitability
Use the numeric context below and keep practical/operational tone (≈120–180 words per section).

Context JSON:
{json.dumps(context, ensure_ascii=False)}
""".strip()
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type":"application/json"},
            json={"model":"gpt-4o-mini","temperature":0.3,"max_tokens":900,
                  "messages":[{"role":"system","content":"You are an LPG safety and logistics assessor."},
                              {"role":"user","content":prompt}]},
            timeout=45
        )
        if r.status_code != 200:
            return offline
        text = r.json()["choices"][0]["message"]["content"].strip()
        sections = {"Safety Risk Profile":"","Environmental Considerations":"","Access & Logistics":"","Overall Site Suitability":""}
        current = None
        mapping = {"[1]":"Safety Risk Profile","[2]":"Environmental Considerations","[3]":"Access & Logistics","[4]":"Overall Site Suitability"}
        for line in text.splitlines():
            t = line.strip()
            for k,name in mapping.items():
                if t.startswith(k):
                    current = name
                    t = t[len(k):].lstrip(":- \t")
                    if t: sections[current] += t + "\n"
                    break
            else:
                if current: sections[current] += t + "\n"
        # fallback if any empty
        for k in sections:
            if not sections[k].strip(): sections[k] = offline[k]
        return sections
    except Exception:
        return offline

# -------------------------
# PDF report
# -------------------------
def build_pdf(words, addr, lat, lon, values, rs, breakdown, controls, ai, map_png) -> str:
    W, H = A4; M = 38; LEAD = 12; y = H - 46; PAGE_BOTTOM = 40
    blue = colors.HexColor("#1f4e79"); grey = colors.HexColor("#555555")
    out = f"precheck_{words.replace('.','_')}.pdf"
    c = rl_canvas.Canvas(out, pagesize=A4)

    def new_page():
        nonlocal y; c.showPage(); y = H - 46
    def ensure(h):
        nonlocal y; 
        if y - h < PAGE_BOTTOM: new_page()
    def text_line(txt, col=colors.black, font="Helvetica", size=10):
        nonlocal y; ensure(size+3); c.setFillColor(col); c.setFont(font,size); c.drawString(M,y,txt); y -= (size+3); c.setFillColor(colors.black)
    def header(txt, size=16, col=blue):
        nonlocal y; ensure(size+6); c.setFillColor(col); c.setFont("Helvetica-Bold", size); c.drawString(M,y,txt); y -= (size+6); c.setFillColor(colors.black)
    def section(txt, size=12, col=blue):
        nonlocal y; ensure(size+8); y -= 4; c.setFillColor(col); c.setFont("Helvetica-Bold", size); c.drawString(M,y,txt); y -= (size+2); c.setFillColor(colors.black)
    def wrap_paragraph(text, width=W-2*M, font="Helvetica", size=10, leading=LEAD):
        nonlocal y; c.setFont(font,size)
        for para in text.split("\n"):
            para = para.rstrip()
            if not para: ensure(leading); y -= leading; continue
            words = para.split(); line=""
            for w in words:
                test = (line+" "+w).strip() if line else w
                if pdfmetrics.stringWidth(test, font, size) <= width: line = test
                else: ensure(leading); c.drawString(M,y,line); y -= leading; line = w
            if line: ensure(leading); c.drawString(M,y,line); y -= leading
    def bullet_list(items, bullet="•", font="Helvetica", size=10, leading=LEAD):
        nonlocal y
        for it in (items or []):
            s = f"{bullet} {it}"; ensure(leading); c.setFont(font,size); c.drawString(M,y,s); y -= leading

    # Title row with icon
    header(f"LPG Pre-Check — ///{words}")
    try:
        c.drawImage(ImageReader("icon.png"), W-M-32, H-52, width=24, height=24, mask="auto")
    except Exception:
        pass

    addr_line = ", ".join([p for p in [addr.get('road'), addr.get('city'), addr.get('postcode')] if p])
    if addr_line: text_line(addr_line, grey)
    if addr.get("display_name"): text_line(addr["display_name"], grey)

    # Map
    if map_png and os.path.exists(map_png):
        try:
            from PIL import Image as PILImage
            iw, ih = PILImage.open(map_png).size
            maxw, maxh = W - 2*M, 240
            sc = min(maxw/iw, maxh/ih)
            ensure(ih*sc + 12)
            c.drawImage(ImageReader(map_png), M, y-ih*sc, width=iw*sc, height=ih*sc)
            y -= ih*sc + 12
        except Exception:
            pass

    # Key metrics
    section("Key Metrics")
    text_line(f"Wind: {values['wind_mps']:.1f} m/s from {values['wind_deg']}°")
    text_line(f"Slope: {values['slope_pct']:.1f} %   |   Approach avg/max: {values['approach_avg']:.1f} / {values['approach_max']:.1f} %")
    text_line(f"Route indirectness: {'n/a' if values['route_ratio'] is None else f'{values['route_ratio']:.2f}×'}")

    section("Separations (~400 m)")
    def fmt(v): return "n/a" if v is None else f"{v:.1f} m"
    bullet_list([
        f"Building: {fmt(values['building_m'])}",
        f"Boundary: {fmt(values['boundary_m'])}",
        f"Road/footpath: {fmt(values['road_m'])}",
        f"Drain/manhole: {fmt(values['drain_m'])}",
        f"Overhead power lines: {fmt(values['overhead_m'])}",
        f"Railway: {fmt(values['rail_m'])}",
        f"Watercourse: {fmt(values['water_m'])}",
        f"Land use: {values['land_use']}",
    ])

    section("Risk score")
    text_line(f"Total: {rs['score']}/100 → {rs['status']}")
    bullet_list(breakdown)

    section("Recommended controls")
    bullet_list(controls)

    for head in ["Safety Risk Profile", "Environmental Considerations", "Access & Logistics", "Overall Site Suitability"]:
        ensure(LEAD); y -= LEAD
        section(head)
        wrap_paragraph(ai.get(head,""))

    c.showPage(); c.save()
    return out

# -------------------------
# Results rendering
# -------------------------
if submitted:
    values = {
        "wind_mps": wind_mps, "wind_deg": wind_deg, "slope_pct": slope_pct,
        "approach_avg": approach_avg, "approach_max": approach_max, "route_ratio": route_ratio,
        "building_m": building_m, "boundary_m": boundary_m, "road_m": road_m, "drain_m": drain_m,
        "overhead_m": overhead_m, "rail_m": rail_m, "water_m": water_m, "land_use": land_use,
    }

    # Access notes/surface from OSM vs chosen vehicle
    notes = restriction_notes(auto.get("restrictions",[]), VEHICLES[vehicle_name])
    surf  = surface_info(auto.get("surfaces",[]))
    access = {"notes": notes, "surface": surf}

    # Score
    rs = risk_score(values, access, veg_3m, los_issue)
    breakdown = rs["explain"][:]
    if values["building_m"] and values["building_m"] >= CoP["to_building_m"]:
        breakdown.append(f"+0 Adequate building separation ({values['building_m']} m ≥ {CoP['to_building_m']} m)")
    if values["overhead_m"] and values["overhead_m"] >= CoP["overhead_info_m"]:
        breakdown.append(f"+0 Overhead outside attention band ({values['overhead_m']} m ≥ {CoP['overhead_info_m']} m)")

    # Controls
    controls = make_controls(rs["status"])

    # Layout: left (metrics) / right (AI + controls)
    L, R = st.columns([0.48, 0.52])
    with L:
        st.markdown("## Key metrics")
        km1, km2, km3 = st.columns(3)
        km1.metric("Wind (m/s)", f"{wind_mps:.1f}")
        km2.metric("Wind dir", f"{wind_deg}°")
        km3.metric("Slope (%)", f"{slope_pct:.1f}")
        k2, k3, k4 = st.columns(3)
        k2.metric("Approach avg", f"{approach_avg:.1f}%")
        k3.metric("Approach max", f"{approach_max:.1f}%")
        k4.metric("Indirectness", "—" if route_ratio is None else f"{route_ratio:.2f}×")

        st.markdown("### Separations (~400 m)")
        def fmt(v): return "—" if v is None else f"{v:.1f} m"
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"Building: {fmt(building_m)}")
            st.info(f"Road/footpath: {fmt(road_m)}")
            st.info(f"Overhead power lines: {fmt(overhead_m)}")
            st.info(f"Watercourse: {fmt(water_m)}")
        with c2:
            st.info(f"Boundary: {fmt(boundary_m)}")
            st.info(f"Drain/manhole: {fmt(drain_m)}")
            st.info(f"Railway: {fmt(rail_m)}")
            st.info(f"Land use: {land_use}")

        st.markdown("### Access suitability (vehicle vs restrictions)")
        if notes:
            st.warning("ATTENTION — check the following against the selected vehicle:")
            for n in notes: st.write(f"• {n}")
        else:
            st.success("PASS — no blocking restrictions detected for the selected vehicle.")

        st.markdown("## Risk result")
        badge = ("🟢 PASS", "🟡 ATTENTION", "🔴 BLOCKER")[ (0 if rs['status']=="PASS" else (1 if rs['status']=="ATTENTION" else 2)) ]
        st.metric("Score", f"{rs['score']}/100", badge)
        st.write("**Top contributing factors**")
        for m in rs["explain"]:
            st.write(f"- {m}")

        # Map preview
        if lat and lon:
            try:
                map_png = make_map_card(st.session_state.get("w3w","site"), lat, lon)
                st.image(map_png, caption="Site map (3 m and 6 m rings)")
            except Exception:
                st.info("Map preview unavailable.")

    with R:
        st.markdown("## AI commentary")
        ctx = {**values, "risk":rs}
        ai = ai_sections(ctx)
        with st.expander("[1] Safety Risk Profile", expanded=True):
            st.write(ai["Safety Risk Profile"])
        with st.expander("[2] Environmental Considerations", expanded=False):
            st.write(ai["Environmental Considerations"])
        with st.expander("[3] Access & Logistics", expanded=False):
            st.write(ai["Access & Logistics"])
        with st.expander("[4] Overall Site Suitability", expanded=False):
            st.write(ai["Overall Site Suitability"])

        st.markdown("## Recommended controls")
        for c in controls: st.write(f"• {c}")

        # PDF
        map_png_for_pdf = None
        try:
            map_png_for_pdf = make_map_card(st.session_state.get("w3w","site"), lat, lon)
        except Exception:
            pass
        pdf_path = build_pdf(st.session_state.get("w3w","site"), addr, lat, lon, values, rs, breakdown, controls, ai, map_png_for_pdf)
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download PDF report", f, file_name=os.path.basename(pdf_path), type="secondary")
