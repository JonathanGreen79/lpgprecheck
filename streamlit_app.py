# streamlit_app.py
import os, io, math, json, requests, re
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

import streamlit as st
from PIL import Image

# ------------------------- Page config -------------------------
PAGE_ICON = "icon.png" if os.path.exists("icon.png") else None
COMPANY_LOGO = "companylogo.png"  # shown top-right after auth, if present
st.set_page_config(
    page_title="LPG Customer Tank — Pre-Check",
    page_icon=PAGE_ICON,
    layout="wide",
)

# ------------------------- Secrets -------------------------
def get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets[name]  # type: ignore[attr-defined]
    except Exception:
        return os.getenv(name, default)

W3W_API_KEY    = get_secret("W3W_API_KEY")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")  # used for online AI
MAPBOX_TOKEN   = get_secret("MAPBOX_TOKEN")
APP_PASSWORD   = get_secret("APP_PASSWORD", "")  # <-- password lives with your API keys
AI_MODEL       = get_secret("AI_MODEL", "gpt-4o-mini")  # optional override

UA = {"User-Agent": "LPG-Precheck/1.9"}

# ------------------------- Auth + status helpers -------------------------
def is_authed() -> bool:
    if not APP_PASSWORD:
        return True
    return bool(st.session_state.get("__auth_ok__", False))

def sidebar_secrets_status():
    def tick(flag: bool) -> str:
        return "✅" if flag else "⚠️"
    st.sidebar.markdown("#### API and Token Access")
    st.sidebar.write(f"{tick(bool(W3W_API_KEY))} What3Words API")
    st.sidebar.write(f"{tick(bool(MAPBOX_TOKEN))} Mapbox Token")
    st.sidebar.write(f"{tick(bool(OPENAI_API_KEY))} OpenAI key")
    st.sidebar.write(f"{tick(bool(APP_PASSWORD))} App Authenticator")
    mode = "Online" if OPENAI_API_KEY else "Offline (fallback)"
    st.sidebar.caption(f"AI commentary mode: **{mode}**")

def sidebar_access():
    st.sidebar.markdown("#### Access")
    if not APP_PASSWORD:
        st.sidebar.warning("No APP_PASSWORD set — access is open.")
        return

    if st.session_state.get("__auth_ok__", False):
        st.sidebar.success("🔓 Access authenticated")
        return

    def _try_unlock():
        ok = (st.session_state.get("__pw_input__", "") == APP_PASSWORD)
        st.session_state["__auth_ok__"] = ok
        if ok:
            st.session_state["__pw_input__"] = ""

    st.sidebar.text_input(
        "Password",
        type="password",
        key="__pw_input__",
        on_change=_try_unlock,
    )
    if st.sidebar.button("Unlock", key="__unlock_btn__"):
        _try_unlock()

    if not st.session_state.get("__auth_ok__", False):
        st.sidebar.info("Enter the password to continue.")

# ------------------------- Vehicle presets -------------------------
VEHICLE_PRESETS = {
    "3.5t Van":               {"length_m": 6.0,  "width_m": 2.1, "height_m": 2.6, "turning_circle_m": 12.0},
    "7.5t Rigid":             {"length_m": 8.0,  "width_m": 2.4, "height_m": 3.3, "turning_circle_m": 16.0},
    "18t Rigid":              {"length_m": 10.0, "width_m": 2.5, "height_m": 3.8, "turning_circle_m": 20.0},
    "26t Rigid":              {"length_m": 11.0, "width_m": 2.55,"height_m": 4.0, "turning_circle_m": 22.0},
    "Artic (44t)":            {"length_m": 16.5, "width_m": 2.55,"height_m": 4.9, "turning_circle_m": 24.0},
    "LPG Tanker (Urban)":     {"length_m": 9.2,  "width_m": 2.5, "height_m": 3.6, "turning_circle_m": 19.0},
    "LPG Tanker (Long Rigid)":{"length_m": 11.0, "width_m": 2.5, "height_m": 3.7, "turning_circle_m": 21.0},
}

# ------------------------- Geom helpers -------------------------
def meters_per_degree(lat_deg: float) -> Tuple[float, float]:
    lat = math.radians(lat_deg)
    mlat = 111132.92 - 559.82*math.cos(2*lat) + 1.175*math.cos(4*lat)
    mlon = 111412.84*math.cos(lat) - 93.5*math.cos(3*lat)
    return mlat, mlon

def ll_to_xy(lat0, lon0, lat, lon):
    mlat, mlon = meters_per_degree(lat0)
    return (lon - lon0) * mlon, (lat - lat0) * mlat

def _dist_m(lat0, lon0, lat1, lon1) -> float:
    dx, dy = ll_to_xy(lat0, lon0, lat1, lon1)
    return math.hypot(dx, dy)

def dist_line(lat0, lon0, line):
    if not line or len(line) < 2:
        return None
    px, py = 0.0, 0.0
    verts = [ll_to_xy(lat0, lon0, la, lo) for la, lo in line]
    best = None
    for (ax, ay), (bx, by) in zip(verts, verts[1:]):
        apx, apy = px - ax, py - ay
        abx, aby = bx - ax, by - ay
        ab2 = abx*abx + aby*aby
        t = 0.0 if ab2 == 0 else max(0.0, min(1.0, (apx*abx + apy*aby) / ab2))
        cx, cy = ax + t*abx, ay + t*aby
        d = math.hypot(px - cx, py - cy)
        best = d if best is None else min(best, d)
    return best

def dist_poly(lat0, lon0, poly):
    if not poly or len(poly) < 3:
        return None
    return dist_line(lat0, lon0, poly + poly[:1])

# ------------------------- External data -------------------------
def w3w_to_latlon(words: str) -> Tuple[Optional[float], Optional[float]]:
    if not W3W_API_KEY or not words:
        return None, None
    try:
        r = requests.get(
            "https://api.what3words.com/v3/convert-to-coordinates",
            params={"words": words.strip().lstrip("/").lstrip("/") , "key": W3W_API_KEY},
            timeout=12,
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
            headers=UA, timeout=15
        )
        if r.status_code == 200:
            j = r.json()
            a = j.get("address") or {}
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

def open_meteo(lat, lon) -> Dict:
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current": "windspeed_10m,winddirection_10m"},
            timeout=12
        )
        if r.status_code == 200:
            cur = r.json().get("current", {})
            spd = cur.get("windspeed_10m")
            deg = cur.get("winddirection_10m")
            comp = None
            if deg is not None:
                comp = ["N","NE","E","SE","S","SW","W","NW"][round((deg%360)/45)%8]
            return {"speed_mps": spd, "deg": deg, "compass": comp}
    except Exception:
        pass
    return {"speed_mps": None, "deg": None, "compass": None}

def overpass_near(lat, lon, radius=400) -> Dict:
    q = f"""
[out:json][timeout:60];
(
  way(around:{radius},{lat},{lon})["building"];
  relation(around:{radius},{lat},{lon})["building"];
  way(around:{radius},{lat},{lon})["highway"];
  node(around:{radius},{lat},{lon})["man_made"="manhole"];
  node(around:{radius},{lat},{lon})["manhole"];
  way(around:{radius},{lat},{lon})["waterway"="drain"];
  way(around:{radius},{lat},{lon})["tunnel"="culvert"];
  way(around:{radius},{lat},{lon})["power"="line"];
  node(around:{radius},{lat},{lon})["power"~"tower|pole"];
  way(around:{radius},{lat},{lon})["railway"]["railway"!="abandoned"]["railway"!="disused"];
  way(around:{radius},{lat},{lon})["waterway"~"river|stream|ditch"];
  way(around:{radius},{lat},{lon})["natural"="water"];
  way(around:{radius},{lat},{lon})["landuse"];
);
out tags geom;
""".strip()
    try:
        r = requests.post(OVERPASS, data={"data": q}, headers=UA, timeout=90)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"elements": []}

def nearest_hospital(lat: float, lon: float) -> Dict:
    """Return nearest hospital-like feature up to 10 km."""
    radii = [400, 1000, 3000, 10000]
    best = None
    def centroid(coords):
        if not coords: return (lat, lon)
        xs = sum(c[0] for c in coords) / len(coords)
        ys = sum(c[1] for c in coords) / len(coords)
        return xs, ys
    for r in radii:
        q = f"""
[out:json][timeout:60];
(
  node(around:{r},{lat},{lon})["amenity"="hospital"];
  node(around:{r},{lat},{lon})["healthcare"="hospital"];
  way(around:{r},{lat},{lon})["amenity"="hospital"];
  way(around:{r},{lat},{lon})["healthcare"="hospital"];
  relation(around:{r},{lat},{lon})["amenity"="hospital"];
  relation(around:{r},{lat},{lon})["healthcare"="hospital"];
);
out tags geom 20;
"""
        try:
            resp = requests.post(OVERPASS, data={"data": q}, headers=UA, timeout=90)
            resp.raise_for_status()
            data = resp.json().get("elements", [])
        except Exception:
            data = []
        for el in data:
            tags = el.get("tags", {}) or {}
            name = tags.get("name") or tags.get("official_name")
            if el.get("type") == "node":
                la, lo = el.get("lat"), el.get("lon")
            else:
                geom = el.get("geometry") or []
                if not geom:
                    continue
                la, lo = centroid([(g["lat"], g["lon"]) for g in geom])
            d = _dist_m(lat, lon, la, lo)
            if best is None or d < best["distance_m"]:
                best = {
                    "name": name or "Nearest hospital",
                    "distance_m": round(d, 1),
                    "lat": la, "lon": lo,
                }
        if best:
            break
    return best or {"name": "n/a", "distance_m": None, "lat": None, "lon": None}

def parse_osm(lat0, lon0, data) -> Dict:
    bpolys, roads, drains, manholes, plines, pnodes, rails, wlines, wpolys, land_polys = [],[],[],[],[],[],[],[],[],[]
    for el in data.get("elements", []):
        t = el.get("type")
        tags = el.get("tags", {}) or {}
        geom = el.get("geometry")
        coords = [(g["lat"], g["lon"]) for g in (geom or [])]
        if tags.get("building") and t in ("way", "relation"):
            bpolys.append(coords)
        elif tags.get("highway") and t == "way":
            roads.append(coords)
        elif t == "way" and (tags.get("waterway")=="drain" or tags.get("tunnel")=="culvert"):
            drains.append(coords)
        elif t == "node" and (tags.get("man_made")=="manhole" or "manhole" in tags):
            manholes.append((el.get("lat"), el.get("lon")))
        elif t == "way" and tags.get("power") == "line":
            plines.append(coords)
        elif t == "node" and tags.get("power") in ("tower","pole"):
            pnodes.append((el.get("lat"), el.get("lon")))
        elif t == "way" and tags.get("railway") and tags.get("railway") not in ("abandoned","disused"):
            rails.append(coords)
        elif t == "way" and tags.get("waterway") in ("river","stream","ditch"):
            wlines.append(coords)
        elif t == "way" and tags.get("natural") == "water":
            wpolys.append(coords)
        elif t in ("way","relation") and tags.get("landuse"):
            land_polys.append({"tag": tags.get("landuse"), "coords": coords})

    def _min_clean(vals):
        vals = [v for v in vals if v is not None]
        return min(vals) if vals else None

    d_build = _min_clean([dist_poly(lat0, lon0, p) for p in bpolys])
    d_road  = _min_clean([dist_line(lat0, lon0, l) for l in roads])

    drain_candidates = [dist_line(lat0, lon0, l) for l in drains]
    if manholes:
        mh = min(_dist_m(lat0, lon0, la, lo) for la,lo in manholes)
        drain_candidates.append(mh)
    d_drain = _min_clean(drain_candidates)

    over_candidates = [dist_line(lat0, lon0, l) for l in plines]
    if pnodes:
        pn = min(_dist_m(lat0, lon0, la, lo) for la,lo in pnodes)
        over_candidates.append(pn)
    d_over = _min_clean(over_candidates)

    d_rail  = _min_clean([dist_line(lat0, lon0, l) for l in rails])
    d_water = _min_clean([dist_line(lat0, lon0, l) for l in wlines] +
                         [dist_poly(lat0, lon0, p) for p in wpolys])

    land_counts = {}
    for lp in land_polys:
        tag = lp["tag"]
        land_counts[tag] = land_counts.get(tag, 0) + 1
    if land_counts:
        top = max(land_counts, key=lambda k: land_counts[k])
        if top in ("residential","commercial","retail"):
            land_class = "Domestic/Urban"
        elif top in ("industrial","industrial;retail"):
            land_class = "Industrial"
        else:
            land_class = "Rural/Agricultural"
    else:
        land_class = "Domestic/Urban" if len(bpolys) > 80 else ("Rural/Agricultural" if len(bpolys) < 20 else "Mixed")

    return {
        "building_m": round(d_build,1) if d_build is not None else None,
        "road_m":     round(d_road,1)  if d_road  is not None else None,
        "drain_m":    round(d_drain,1) if d_drain is not None else None,
        "overhead_m": round(d_over,1)  if d_over  is not None else None,
        "rail_m":     round(d_rail,1)  if d_rail  is not None else None,
        "water_m":    round(d_water,1) if d_water is not None else None,
        "land_class": land_class,
    }

# ------------------------- Depots -------------------------
DEPOTS = [
    ("Blaydon", -1.744101, 54.9756187),
    ("Buckfastleigh", -3.7833235, 50.4869034),
    ("Burton", -1.559383, 52.7768106),
    ("Cairnhill", -2.556494, 57.3820238),
    ("Carlisle", -2.950747, 54.898308),
    ("Conwy", -3.8574551, 53.2876319),
    ("Defford", -2.161253, 52.0883951),
    ("Evanton", -4.3069473, 57.6723499),
    ("Fawley", -1.3778735, 50.8490674),
    ("Grangemouth", -3.7212715, 56.0224567),
    ("Knowsley", -2.8682086, 53.4630737),
    ("Launceston", -4.3922685, 50.6278182),
    ("Leeds", -1.5059716, 53.7829647),
    ("Llandarcy", -3.8451237, 51.6391619),
    ("Ludham", 1.5499137, 52.7241592),
    ("Newport", -2.9782096, 51.5675306),
    ("Paisley", -4.424505, 55.8532784),
    ("Perth", -3.4755149, 56.4169938),
    ("Peterborough", -0.2142451, 52.5776289),
    ("Presteigne", -2.9945084, 52.2688513),
    ("Rainham", 0.1698276, 51.509824),
    ("Sittingbourne", 0.7530866, 51.3850881),
    ("Skegness", 0.2666851, 53.2554345),
    ("Staveley", -1.355699, 53.2761913),
    ("Stoke", -2.1676655, 52.9648079),
    ("Swinton", -2.2697802, 55.7230767),
    ("Witney", -1.5068047, 51.7926625),
    ("Wrexham", -2.9252352, 53.0264358),
]

def nearest_depots(lat: float, lon: float, n: int = 3):
    rows = []
    for name, dlon, dlat in DEPOTS:
        m = _dist_m(lat, lon, dlat, dlon)
        rows.append({"name": name, "miles": m / 1609.344})
    rows.sort(key=lambda r: r["miles"])
    return rows[:n]

# ------------------------- Routing (OSRM) + quick analysis -------------------------
def _bearing_deg(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dlon)
    brng = (math.degrees(math.atan2(x, y)) + 360) % 360
    return brng

def _compass_from_deg(deg: float) -> str:
    return ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"][round((deg%360)/22.5)%16]

def osrm_route(lat1, lon1, lat2, lon2, overview=True, steps=True) -> Dict:
    try:
        url = f"https://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
        params = {
            "alternatives": "false",
            "overview": "full" if overview else "false",
            "steps": "true" if steps else "false",
            "geometries": "geojson",
            "annotations": "false",
        }
        r = requests.get(url, params=params, headers=UA, timeout=20)
        r.raise_for_status()
        j = r.json()
        if j.get("routes"):
            return j["routes"][0]
    except Exception:
        pass
    return {}

def quick_route_snapshot(dep: Dict, site_lat: float, site_lon: float) -> Dict:
    out = {
        "miles": None,
        "eta_min": None,
        "final_road": None,
        "approach_deg": None,
        "approach_compass": None,
        "winding": None,
        "counts": {},
        "full_miles": None,
    }
    if not dep:
        return out

    route = osrm_route(dep["lat"], dep["lon"], site_lat, site_lon)
    if not route:
        return out

    total_m = route.get("distance") or 0.0
    total_s = route.get("duration") or 0.0
    out["full_miles"] = round(total_m / 1609.344, 1)

    miles = round(total_m / 1609.344, 1)
    eta_min = round(total_s / 60.0)
    out["miles"] = miles
    out["eta_min"] = eta_min

    steps = []
    legs = route.get("legs") or []
    if legs:
        for leg in legs:
            for stp in leg.get("steps", []):
                steps.append(stp)

    if steps:
        last = steps[-1]
        out["final_road"] = (last.get("name") or "").strip() or None

    coords = route.get("geometry", {}).get("coordinates") or []
    if len(coords) >= 2:
        a = (coords[-2][1], coords[-2][0])
        b = (coords[-1][1], coords[-1][0])
        br = _bearing_deg(a, b)
        out["approach_deg"] = round(br)
        out["approach_compass"] = _compass_from_deg(br)

    if len(coords) >= 3:
        total = 0.0
        changes = 0.0
        last_bear = None
        for i in range(len(coords)-1, 0, -1):
            latb, lonb = coords[i][1], coords[i][0]
            lata, lona = coords[i-1][1], coords[i-1][0]
            seg = _dist_m(latb, lonb, lata, lona)
            total += seg
            bear = _bearing_deg((lata,lona), (latb,lonb))
            if last_bear is not None:
                d = abs((bear - last_bear + 180) % 360 - 180)
                changes += d
            last_bear = bear
            if total >= 1609.344:
                break
        if changes < 120:
            out["winding"] = "Low"
        elif changes < 240:
            out["winding"] = "Medium"
        else:
            out["winding"] = "High"

    keywords = {
        "barrier_gate": r"\b(gate|barrier)\b",
        "bollard": r"\b(bollard|bollards)\b",
        "tunnel": r"\b(tunnel|underpass)\b",
        "ford": r"\b(ford)\b",
        "narrow": r"\b(narrow)\b",
        "weight": r"\b(weight|tonnage)\b",
        "height": r"\b(height|low\s+bridge|clearance)\b",
        "width": r"\b(width)\b",
        "private": r"\b(private\s+road|no\s+through)\b",
        "construction": r"\b(construction|roadworks|closure|closed)\b",
        "rail": r"\b(level\s+crossing|rail)\b",
    }
    counts = {k: 0 for k in keywords}
    try:
        for stp in steps:
            txt = " ".join([
                str(stp.get("name") or ""),
                str(stp.get("ref") or ""),
                str(stp.get("maneuver", {}).get("instruction") or "")
            ]).lower()
            for k, rx in keywords.items():
                if re.search(rx, txt):
                    counts[k] += 1
    except Exception:
        pass
    out["counts"] = counts
    return out

# ------------------------- Risk scoring -------------------------
CoP = {
    "to_building_m": 3.0, "to_boundary_m": 3.0, "to_ignition_m": 3.0, "to_drain_m": 3.0,
    "overhead_info_m": 10.0, "overhead_block_m": 5.0, "rail_attention_m": 30.0,
}

@dataclass
class RiskResult:
    score: float
    status: str
    explain: List[Tuple[int, str]]

def risk_score(
    feats: Dict, wind: Dict, slope_pct: Optional[float],
    enclosure_sides: int, los_issue: bool, veg_3m: int,
    open_field_m: Optional[float]
) -> RiskResult:
    pts: float = 0.0
    why: List[Tuple[int,str]] = []

    def add(x: int, msg: str):
        nonlocal pts, why
        pts += x
        why.append((x, msg))

    def penal(dist, lim, msg, base=18, per=6, cap=40):
        if dist is None or dist >= lim: return
        p = min(cap, base + per*(lim - dist))
        add(int(p), f"{msg} below {lim} m (≈ {dist} m)")

    penal(feats.get("building_m"), CoP["to_building_m"], "Below 3.0 m")
    penal(feats.get("road_m"),     CoP["to_ignition_m"], "Ignition proxy (road/footpath)")
    penal(feats.get("drain_m"),    CoP["to_drain_m"],    "Drain/manhole within 3 m")

    d_ov = feats.get("overhead_m")
    if d_ov is not None:
        if d_ov < CoP["overhead_block_m"]:
            add(28, "Overhead in no-go band")
        elif d_ov < CoP["overhead_info_m"]:
            add(10, "Overhead within 10 m")

    d_rail = feats.get("rail_m")
    if d_rail is not None and d_rail < CoP["rail_attention_m"]:
        add(10, "Railway within 30.0 m")

    if feats.get("water_m") is not None and feats["water_m"] < 50:
        add(8, "Watercourse within 50 m")

    if slope_pct and slope_pct >= 3.0:
        add(8, f"Local slope {slope_pct:.1f}%")

    if wind.get("speed_mps") is not None and wind["speed_mps"] < 1.0:
        add(6, f"Low wind {wind['speed_mps']:.1f} m/s")

    if enclosure_sides >= 3:
        add(12, f"Enclosure effect: {enclosure_sides} solid side(s)")
    elif enclosure_sides == 2:
        add(8, "Partial enclosure: 2 solid sides")

    if los_issue:
        add(8, "Restricted line-of-sight at stand")

    if veg_3m >= 2:
        add(6, f"Vegetation within 3 m (level {veg_3m})")

    if open_field_m is not None and open_field_m < 20:
        add(6, f"Open field within {open_field_m:.0f} m")

    pts = round(min(100.0, pts), 1)
    status = "PASS" if pts < 20 else ("ATTENTION" if pts < 50 else "BLOCKER")
    why.sort(key=lambda x: -x[0])
    return RiskResult(pts, status, why)

# ------------------------- Map -------------------------
def fetch_map(lat, lon, zoom=17, size=(1000, 700)) -> Optional[Image.Image]:
    if not MAPBOX_TOKEN:
        return None
    style = "light-v11"
    marker = f"pin-l+f30({lon},{lat})"
    url = (f"https://api.mapbox.com/styles/v1/mapbox/{style}/static/"
           f"{marker}/{lon},{lat},{zoom},0/{size[0]}x{size[1]}?access_token={MAPBOX_TOKEN}")
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGBA")
    except Exception:
        return None

# ------------------------- AI commentary (ONLINE by default, OFFLINE fallback) -------------------------
def _offline_ai_sections(ctx: Dict) -> Dict[str, str]:
    feats, wind = ctx["feats"], ctx["wind"]
    slope_pct = ctx["slope_pct"] or 0.0
    risk = ctx["risk"]
    sides = ctx["enclosure_sides"]
    los = ctx["los_issue"]
    veg = ctx["veg_3m"]
    land = feats.get("land_class", "n/a")
    route = ctx.get("route", {}) or {}

    s1 = (
        f"The local slope is {slope_pct:.1f}%. Key separations (m): "
        f"building {feats.get('building_m','n/a')}, boundary n/a, road {feats.get('road_m','n/a')}, "
        f"drain {feats.get('drain_m','n/a')}, overhead {feats.get('overhead_m','n/a')}, rail {feats.get('rail_m','n/a')}. "
        f"Wind {wind.get('speed_mps') or 0:.1f} m/s from {wind.get('compass') or 'n/a'}. "
        f"Heuristic {risk.score:.1f}/100 → {risk.status}. Drivers: "
        + "; ".join([f"{p} {m}" for p, m in risk.explain[:5]]) + "."
    )
    counts = route.get("counts") or {}
    issues_line = "No route flags detected." if not counts or sum(counts.values()) == 0 else (
        "Route flags — " + ", ".join([f"{k.replace('_',' ')}: {v}" for k, v in counts.items() if v])
    )
    s2 = (
        f"Flood Low (No mapped watercourse nearby). Watercourse ~{feats.get('water_m','n/a')} m; "
        f"drains/manholes {feats.get('drain_m','n/a')}. Land use {land}. Vegetation within 3 m: level {veg}. "
        f"{issues_line}"
    )
    s3 = (
        f"Access lines of sight: {'restricted' if los else 'clear'}; enclosure {sides} side(s). "
        f"Nearest depot drive ~{route.get('miles','n/a')} mi, ETA {route.get('eta_min','n/a')} min. "
        f"Final approach via {route.get('final_road') or 'n/a'} from {route.get('approach_compass') or 'n/a'} "
        f"({route.get('approach_deg') or 'n/a'}°). Last-mile winding: {route.get('winding') or 'n/a'}."
    )
    s4 = (
        "Attention required — ensure separation compliance, ignition control, drainage protection, "
        "and safe approach/egress. Confirm route restrictions before mobilisation."
    )
    return {
        "Safety Risk Profile": s1,
        "Environmental Considerations": s2,
        "Access & Logistics": s3,
        "Overall Site Suitability": s4,
    }

def _online_ai_sections(ctx: Dict, model: str = None) -> Dict[str, str]:
    if not OPENAI_API_KEY:
        raise RuntimeError("No OpenAI key")

    model = model or AI_MODEL
    feats = ctx.get("feats", {})
    wind = ctx.get("wind", {})
    risk = ctx.get("risk")
    route = ctx.get("route", {}) or {}

    site = {
        "wind": {"mps": wind.get("speed_mps"), "dir_deg": wind.get("deg"), "dir_compass": wind.get("compass")},
        "slope_pct": ctx.get("slope_pct"),
        "separations_m": {
            "building": feats.get("building_m"),
            "boundary": feats.get("boundary_m"),
            "road": feats.get("road_m"),
            "drain": feats.get("drain_m"),
            "overhead": feats.get("overhead_m"),
            "rail": feats.get("rail_m"),
            "watercourse": feats.get("water_m"),
            "land_use": feats.get("land_class"),
        },
        "enclosure_sides": ctx.get("enclosure_sides"),
        "los_restricted": bool(ctx.get("los_issue")),
        "vegetation_3m_level": ctx.get("veg_3m"),
        "risk": {
            "score": getattr(risk, "score", None),
            "status": getattr(risk, "status", None),
            "drivers": getattr(risk, "explain", []),
        },
        "route": {
            "driving_miles": route.get("miles"),
            "eta_min": route.get("eta_min"),
            "final_road": route.get("final_road"),
            "approach_deg": route.get("approach_deg"),
            "approach_compass": route.get("approach_compass"),
            "winding": route.get("winding"),
            "counts": route.get("counts"),
        }
    }

    system = (
        "You are a safety and logistics assistant writing concise LPG site pre-check commentary. "
        "Audience is field engineers; keep it professional, UK terminology. "
        "Use the route snapshot (miles/ETA/approach/winding and counts) to tailor the Access & Logistics section. "
        "Output exactly four sections with these exact headings, each as a single short paragraph:\n"
        "### Safety Risk Profile\n"
        "### Environmental Considerations\n"
        "### Access & Logistics\n"
        "### Overall Site Suitability"
    )
    user = (
        "Using this site context, write those four sections (short, specific, actionable). "
        "Avoid repeating every number; focus on implications and what to do. "
        f"Context:\n{json.dumps(site, ensure_ascii=False)}"
    )

    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.3,
                "max_tokens": 450,
            },
            timeout=30,
        )
        r.raise_for_status()
        content = (r.json()["choices"][0]["message"]["content"] or "").strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI error: {e}")

    text = (content or "").replace("\r\n", "\n").strip()
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)

    section_keys = [
        "Safety Risk Profile",
        "Environmental Considerations",
        "Access & Logistics",
        "Overall Site Suitability",
    ]
    sections = {k: "" for k in section_keys}

    head_line_re = re.compile(
        r"(?im)^\s*(?:#+\s*)?(?:\*\*)?(Safety Risk Profile|Environmental Considerations|Access & Logistics|Overall Site Suitability)(?:\*\*)?\s*:?\s*"
    )

    def slice_by_matches(txt: str) -> Dict[str, str]:
        found = {}
        matches = list(head_line_re.finditer(txt))
        if not matches:
            return found
        for idx, m in enumerate(matches):
            key = m.group(1)
            start = m.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(txt)
            body = txt[start:end].strip()
            found[key] = body
        return found

    found = slice_by_matches(text)
    if sum(bool(v) for v in found.values()) < 2:
        inline = text
        for k in section_keys:
            inline = re.sub(
                rf"(?i)(?<!\n)\s+({re.escape(k)})\s*[:\-–—]\s*",
                r"\n\1: ",
                inline,
            )
        found = slice_by_matches(inline)

    for k in section_keys:
        v = (found.get(k) or "").strip()
        v = v.lstrip(" .,:;–—-")
        sections[k] = v or "No additional notes for this section."

    return sections

def ai_sections(ctx: Dict) -> Dict[str, str]:
    if OPENAI_API_KEY:
        try:
            return _online_ai_sections(ctx)
        except Exception:
            pass
    return _offline_ai_sections(ctx)

# ------------------------- UI helper: seeded, editable distance -------------------------
def nm_distance(
    label: str,
    key: str,
    auto_val: Optional[float],
    max_val: float = 2000.0,
    seed_tag: Optional[str] = None,
) -> Optional[float]:
    tag = seed_tag or "__default__"
    if st.session_state.get(f"{key}__seed") != tag:
        st.session_state[f"{key}__nm"]  = (auto_val is None)
        st.session_state[f"{key}__val"] = 0.0 if auto_val is None else float(auto_val)
        st.session_state[f"{key}__seed"] = tag

    c1, c2 = st.columns([0.78, 0.22])
    with c1:
        val = st.number_input(
            label,
            min_value=0.0, max_value=float(max_val), step=0.1,
            value=float(st.session_state[f"{key}__val"]),
            key=f"{key}__val_input",
        )
    with c2:
        nm = st.checkbox("Not mapped", value=st.session_state[f"{key}__nm"], key=f"{key}__nm_chk")

    st.session_state[f"{key}__val"] = val
    st.session_state[f"{key}__nm"]  = nm
    return None if nm else float(val)

# ------------------------- Pretty key/value block -------------------------
def kv_block(title: str, data: Dict, cols: int = 2, fmt: Dict[str, str] | None = None):
    st.markdown(f"### {title}" if title.lower().startswith("key") else f"#### {title}")
    keys = list(data.keys())
    rows = (len(keys) + cols - 1) // cols
    fmt = fmt or {}
    def show(k, v):
        if v is None:
            return "—"
        if isinstance(v, (int, float)) and k in fmt:
            try:
                return format(v, fmt[k])
            except Exception:
                return str(v)
        return str(v)
    for r in range(rows):
        cs = st.columns(cols)
        for c in range(cols):
            i = r + c*rows
            if i < len(keys):
                k = keys[i]
                v = data[k]
                with cs[c]:
                    st.markdown(
                        f"<div style='line-height:1.4'><b>{k}:</b> {show(k,v)}</div>",
                        unsafe_allow_html=True,
                    )

def winding_badge(level: Optional[str]) -> str:
    color = {
        "Low": "#289e41",
        "Medium": "#e0a10b",
        "High": "#cc2b2b"
    }.get(level or "", "#7a7a7a")
    label = level or "n/a"
    return f"<span style='background:{color};color:white;padding:2px 8px;border-radius:999px;font-weight:600'>{label}</span>"

# ------------------------- Sidebar (status & access) -------------------------
sidebar_secrets_status()
sidebar_access()
if not is_authed():
    st.stop()

# ------------------------- App -------------------------
# Title + right-aligned company logo (shown only after auth)
if PAGE_ICON:
    header_cols = st.columns([0.08, 0.72, 0.20])
    with header_cols[0]:
        st.image(PAGE_ICON, use_container_width=True)
    with header_cols[1]:
        st.title("LPG Customer Tank — Pre-Check")
    with header_cols[2]:
        if os.path.exists(COMPANY_LOGO) and is_authed():
            st.image(COMPANY_LOGO, use_container_width=True)
else:
    header_cols = st.columns([0.80, 0.20])
    with header_cols[0]:
        st.title("LPG Customer Tank — Pre-Check")
    with header_cols[1]:
        if os.path.exists(COMPANY_LOGO) and is_authed():
            st.image(COMPANY_LOGO, use_container_width=True)

st.caption("Enter a what3words location, review/edit auto-filled data, then confirm to assess.")

# ------------------------- W3W input THEN buttons -------------------------
w3w_input = st.text_input(
    "what3words (word.word.word):",
    value=st.session_state.get("w3w", ""),
    key="w3w_entry",
)

c_run, c_reset, _ = st.columns([0.18, 0.14, 0.68])
with c_run:
    run = st.button("Run Pre-Check", type="primary", use_container_width=True, key="run_btn")
with c_reset:
    reset = st.button("Reset", type="secondary", use_container_width=True, key="reset_btn")

# Reset: clear data below and W3W, hide edit/results until new W3W entered
if reset:
    for k in list(st.session_state.keys()):
        if k.startswith("d_") or k.endswith("__val") or k.endswith("__nm") or k.endswith("__seed") or k.startswith("veh_"):
            st.session_state.pop(k, None)
    st.session_state.pop("auto", None)
    st.session_state.pop("w3w", None)
    st.rerun()

# ------------------------- Run handler -------------------------
if run:
    w3w_clean = (w3w_input or "").strip().removeprefix("///").removeprefix("/")
    if not w3w_clean or w3w_clean.count(".") != 2:
        st.error("Please enter a valid what3words (word.word.word).")
    else:
        st.session_state.pop("auto", None)
        st.session_state["w3w"] = w3w_clean
        with st.status("Fetching site data…", expanded=False):
            lat, lon = w3w_to_latlon(w3w_clean)
            if lat is None:
                st.error("what3words lookup failed.")
                st.stop()

            addr = reverse_geocode(lat, lon)
            wind = open_meteo(lat, lon)
            osm  = overpass_near(lat, lon, radius=400)
            feats = parse_osm(lat, lon, osm)
            hosp = nearest_hospital(lat, lon)

            # nearest depot set
            dep3 = nearest_depots(lat, lon, n=3)
            dep_sel = dep3[0] if dep3 else None
            dep_sel_full = {"name": dep_sel["name"], "lat": next(d[2] for d in DEPOTS if d[0]==dep_sel["name"]),
                            "lon": next(d[1] for d in DEPOTS if d[0]==dep_sel["name"])} if dep_sel else None

            # quick route snapshot
            route_snap = {}
            if dep_sel_full:
                route_snap = quick_route_snapshot(dep_sel_full, lat, lon)

            st.session_state["auto"] = {
                "lat": lat, "lon": lon,
                "addr": addr,
                "hospital": hosp,
                "wind_mps": wind.get("speed_mps") or 0.0,
                "wind_deg": wind.get("deg") or 0,
                "wind_comp": wind.get("compass") or "n/a",
                "slope_pct": 3.5,
                "approach_avg": 0.9,
                "approach_max": 3.5,
                **feats,
                "nearest_depots": dep3,
                "route_snap": route_snap,
            }
            st.success("Auto data ready.")

# --- Use persisted auto for the form and results ---
auto = st.session_state.get("auto", {})
if auto:
    seed = st.session_state.get("w3w", "")
    submitted = False

    st.markdown("### Edit & confirm")
    with st.form("inputs"):
        # ---------------- Location & address (FIRST) ----------------
        st.subheader("Location & address")
        a1, a2 = st.columns([0.6, 0.4])
        with a1:
            addr_road     = st.text_input("Road / street", auto.get("addr", {}).get("road", ""), key="addr_road")
            addr_city     = st.text_input("Town / City",   auto.get("addr", {}).get("city", ""), key="addr_city")
            addr_postcode = st.text_input("Postcode",      auto.get("addr", {}).get("postcode", ""), key="addr_postcode")
            addr_local    = st.text_input("Local authority", auto.get("addr", {}).get("local_authority", ""), key="addr_local")
            hosp_name     = st.text_input("Nearest hospital", (auto.get("hospital", {}) or {}).get("name", ""), key="hosp_name")
        with a2:
            st.text_input("what3words", st.session_state.get("w3w", ""), disabled=True, key="w3w_display")
            st.text_input("Latitude", f"{auto.get('lat', '')}", disabled=True, key="lat_display")
            st.text_input("Longitude", f"{auto.get('lon', '')}", disabled=True, key="lon_display")
            hosp_dist = (auto.get("hospital", {}) or {}).get("distance_m", None)
            hosp_km = f"{(hosp_dist/1000):.2f} km" if isinstance(hosp_dist, (int, float)) else "—"
            st.text_input("Hospital distance (approx.)", hosp_km, disabled=True, key="hosp_dist_ro")
        st.markdown("---")

        # ---------------- Environment & approach ----------------
        st.subheader("Environment & approach")
        e1, e2, e3 = st.columns(3)
        with e1:
            wind_mps = st.number_input("Wind (m/s)", 0.0, 60.0, float(auto.get("wind_mps", 0.0)), 0.1, key="wind_mps_in")
        with e2:
            wind_deg = st.number_input("Wind dir (°)", 0, 359, int(auto.get("wind_deg", 0)), 1, key="wind_deg_in")
        with e3:
            slope_pct = st.number_input("Slope (%)", 0.0, 100.0, float(auto.get("slope_pct", 0.0)), 0.1, key="slope_pct_in")

        a1c, a2c, a3c = st.columns(3)
        with a1c:
            approach_avg = st.number_input("Approach avg (%)", 0.0, 100.0, float(auto.get("approach_avg", 0.0)), 0.1, key="approach_avg_in")
        with a2c:
            approach_max = st.number_input("Approach max (%)", 0.0, 100.0, float(auto.get("approach_max", 0.0)), 0.1, key="approach_max_in")
        with a3c:
            rr_str = st.text_input("Route indirectness (× crow-fly) — optional", value="", placeholder="leave blank", key="route_ratio_input")
            try:
                route_ratio = float(rr_str) if rr_str.strip() else None
            except:
                route_ratio = None

        st.markdown("---")
        st.subheader("Separations (~400 m)")
        s1, s2 = st.columns(2)
        with s1:
            building_m = nm_distance("Building (m)", "d_building_m", auto.get("building_m"), seed_tag=seed)
            road_m     = nm_distance("Road/footpath (m)", "d_road_m", auto.get("road_m"), seed_tag=seed)
            overhead_m = nm_distance("Overhead power lines (m)", "d_overhead_m", auto.get("overhead_m"), seed_tag=seed)
            water_m    = nm_distance("Watercourse (m)", "d_water_m", auto.get("water_m"), seed_tag=seed)
        with s2:
            boundary_m = nm_distance("Boundary (m)", "d_boundary_m", auto.get("boundary_m"), seed_tag=seed)
            drain_m    = nm_distance("Drain/manhole (m)", "d_drain_m", auto.get("drain_m"), seed_tag=seed)
            rail_m     = nm_distance("Railway (m)", "d_rail_m", auto.get("rail_m"), seed_tag=seed)
            land_use   = st.selectbox(
                "Land use",
                ["Domestic/Urban", "Industrial", "Mixed", "Rural/Agricultural"],
                index=["Domestic/Urban", "Industrial", "Mixed", "Rural/Agricultural"].index(
                    auto.get("land_class", "Domestic/Urban")
                ),
                key="land_use_sel"
            )

        st.markdown("---")
        st.subheader("Vehicle")
        vcol1, vcol2, vcol3, vcol4 = st.columns(4)
        with vcol1:
            vehicle_type = st.selectbox(
                "Type",
                list(VEHICLE_PRESETS.keys()),
                index=list(VEHICLE_PRESETS.keys()).index("LPG Tanker (Urban)") if "LPG Tanker (Urban)" in VEHICLE_PRESETS else 0,
                key="vehicle_type_sel"
            )
        preset = VEHICLE_PRESETS[vehicle_type]
        basekey = f"veh_{vehicle_type}"
        if f"{basekey}_seeded" not in st.session_state:
            for k, v in preset.items():
                st.session_state[f"{basekey}_{k}"] = v
            st.session_state[f"{basekey}_seeded"] = True
        with vcol2:
            veh_length_m = st.number_input("Length (m)", 3.0, 25.0, float(st.session_state.get(f"{basekey}_length_m", preset["length_m"])), 0.1, key="veh_len_in")
        with vcol3:
            veh_width_m = st.number_input("Width (m)", 2.0, 3.0, float(st.session_state.get(f"{basekey}_width_m", preset["width_m"])), 0.01, key="veh_w_in")
        with vcol4:
            veh_height_m = st.number_input("Height (m)", 2.0, 6.5, float(st.session_state.get(f"{basekey}_height_m", preset["height_m"])), 0.01, key="veh_h_in")
        tc_col = st.columns(4)[0]
        with tc_col:
            turning_circle_m = st.number_input("Turning circle (m)", 8.0, 30.0, float(st.session_state.get(f"{basekey}_turning_circle_m", preset["turning_circle_m"])), 0.1, key="veh_tc_in")
        st.session_state[f"{basekey}_length_m"] = veh_length_m
        st.session_state[f"{basekey}_width_m"]  = veh_width_m
        st.session_state[f"{basekey}_height_m"] = veh_height_m
        st.session_state[f"{basekey}_turning_circle_m"] = turning_circle_m

        # ---------------- Nearest Depot & Logistics ----------------
        st.markdown("---")
        st.subheader("Nearest Depot & Logistics ↪")

        ndc1, ndc2 = st.columns([0.6, 0.4])
        depots3 = auto.get("nearest_depots") or nearest_depots(auto["lat"], auto["lon"], n=3)
        with ndc1:
            st.text_input("Nearest depot", depots3[0]["name"] if depots3 else "—", disabled=True)
        with ndc2:
            st.text_input("Distance (miles)", f"{depots3[0]['miles']:.1f} miles" if depots3 else "—", disabled=True)

        st.caption("Top 3 nearest depots (crow-fly)")
        st.table({
            "Depot": [d["name"] for d in depots3],
            "Distance (miles)": [f"{d['miles']:.1f}" for d in depots3],
        })

        # quick route snapshot
        st.markdown("**Quick route snapshot (nearest depot)**")
        rs = auto.get("route_snap") or {}
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Driving distance**")
            st.markdown(f"🛣️  <span style='font-size:28px;font-weight:700'>{rs.get('miles','n/a')} mi</span>", unsafe_allow_html=True)
        with c2:
            st.markdown("**ETA (typical)**")
            st.markdown(f"⏱️  <span style='font-size:28px;font-weight:700'>{rs.get('eta_min','n/a')} min</span>", unsafe_allow_html=True)
        with c3:
            st.markdown("**Last-mile winding**")
            st.markdown("🌀 " + winding_badge(rs.get("winding")), unsafe_allow_html=True)

        appr = []
        if rs.get("final_road"): appr.append(f"Final approach via **{rs['final_road']}**")
        if rs.get("approach_compass") is not None and rs.get("approach_deg") is not None:
            appr.append(f"approach from **{rs['approach_compass']} ({rs['approach_deg']}°)**")
        if appr:
            st.markdown("• " + "; ".join(appr) + ".")

        st.markdown("---")
        st.subheader("Site options")
        v1, v2 = st.columns([0.5, 0.5])
        with v1:
            veg_3m = st.slider("Vegetation within 3 m of tank (0 none → 3 heavy)", 0, 3, 1, key="veg_slider")
            enclosure_sides = st.select_slider(
                "Number of solid sides enclosing tank/stand (fence/walls)",
                options=[0, 1, 2, 3, 4], value=0, key="enclosure_sides_sel"
            )
            los_issue = st.toggle(
                "Restricted line-of-sight at Delivery Point",
                value=False,
                key="los_tgl"
            )

        with v2:
            stand_surface = st.selectbox(
                "Stand surface",
                ["asphalt", "concrete", "block paving", "gravel", "grass", "other"],
                index=0,
                key="stand_surface_sel"
            )
            open_field_m = nm_distance("Distance to open field (m)", "d_open_field_m", None, seed_tag=seed)

        notes_txt = st.text_input("Notes (vegetation / sightlines / special instructions)", value="", key="notes_input")

        submitted = st.form_submit_button("Confirm & assess", type="primary", key="confirm_btn")

    # ---------------- Confirm handler (with status queue) ----------------
    if submitted:
        feats = {
            "building_m": building_m, "road_m": road_m, "drain_m": drain_m,
            "overhead_m": overhead_m, "rail_m": rail_m, "water_m": water_m,
            "land_class": land_use,
        }
        wind = {"speed_mps": wind_mps, "deg": wind_deg,
                "compass": ["N","NE","E","SE","S","SW","W","NW"][round((wind_deg%360)/45)%8]}

        # Run heavy work first with a visible status panel
        with st.status("Assessing site…", expanded=True) as stat:
            stat.write("1/3 Scoring risk…")
            risk = risk_score(
                feats=feats, wind=wind, slope_pct=slope_pct,
                enclosure_sides=enclosure_sides, los_issue=los_issue,
                veg_3m=veg_3m, open_field_m=open_field_m
            )

            stat.update(label="Assessing site… • 2/3 Generating AI commentary…")
            ctx_for_ai = {
                "feats": feats, "wind": wind, "slope_pct": slope_pct,
                "enclosure_sides": enclosure_sides, "los_issue": los_issue,
                "veg_3m": veg_3m, "risk": risk,
                "route": (auto.get("route_snap") or {}),
            }
            ai = ai_sections(ctx_for_ai)

            stat.update(label="Assessing site… • 3/3 Preparing map & export…")
            # Prepare a static map once and reuse
            map_path = None
            if MAPBOX_TOKEN:
                img_for_export = fetch_map(st.session_state["auto"]["lat"], st.session_state["auto"]["lon"])
                if img_for_export:
                    try:
                        map_path = "map_preview.png"
                        img_for_export.save(map_path)
                    except Exception:
                        map_path = None

            stat.update(label="Assessment complete ✅", state="complete")

        addr_edited = {
            "road": addr_road, "city": addr_city, "postcode": addr_postcode,
            "local_authority": addr_local,
            "display_name": auto.get("addr", {}).get("display_name", ""),
            "hospital_name": hosp_name,
            "hospital_distance_m": (auto.get("hospital", {}) or {}).get("distance_m", None),
        }

        left, right = st.columns([0.45, 0.55])

        with left:
            kv_block(
                "Key metrics",
                {
                    "Wind (m/s)": round(wind_mps, 1),
                    "Wind dir (°/compass)": f"{wind_deg} / {wind['compass']}",
                    "Slope (%)": round(slope_pct, 1),
                    "Approach avg/max (%)": f"{approach_avg:.1f} / {approach_max:.1f}",
                    "Flood": "Low — No mapped watercourse nearby" if (water_m is None or water_m >= 150) else "Medium/High",
                    "Nearest hospital": hosp_name,
                    "Hospital distance (km)": round(((auto.get("hospital", {}) or {}).get("distance_m", 0.0) or 0.0)/1000.0, 2),
                },
                cols=2,
                fmt={"Wind (m/s)": ".1f", "Slope (%)": ".1f", "Hospital distance (km)": ".2f"}
            )

            kv_block(
                "Separations (~400 m)",
                {
                    "Building (m)": building_m,
                    "Boundary (m)": boundary_m,
                    "Road/footpath (m)": road_m,
                    "Drain/manhole (m)": drain_m,
                    "Overhead power lines (m)": overhead_m,
                    "Railway (m)": rail_m,
                    "Watercourse (m)": water_m,
                    "Land use": land_use,
                },
                cols=2,
                fmt={"Building (m)": ".1f", "Road/footpath (m)": ".1f", "Railway (m)": ".1f", "Watercourse (m)": ".0f"}
            )

            kv_block(
                "Vehicle",
                {
                    "Type": vehicle_type,
                    "Length (m)": veh_length_m,
                    "Width (m)": veh_width_m,
                    "Height (m)": veh_height_m,
                    "Turning circle (m)": turning_circle_m,
                },
                cols=2,
                fmt={"Length (m)": ".1f", "Width (m)": ".2f", "Height (m)": ".2f", "Turning circle (m)": ".1f"}
            )

            st.markdown("### Risk result")
            badge = ("✅ PASS" if risk.status=="PASS"
                     else "🟡 ATTENTION" if risk.status=="ATTENTION"
                     else "🟥 BLOCKER")
            st.subheader(f"{risk.score:.1f}/100  {badge}")
            st.markdown("#### Top contributing factors")
            for p, m in risk.explain[:6]:
                st.write(f"+{p} {m}")

            if MAPBOX_TOKEN and map_path and os.path.exists(map_path):
                st.markdown("#### Map")
                try:
                    st.image(map_path, caption="Map (centered on W3W)", use_container_width=True)
                except Exception:
                    pass

        with right:
            st.markdown("### AI commentary")
            sections = [
                "Safety Risk Profile",
                "Environmental Considerations",
                "Access & Logistics",
                "Overall Site Suitability",
            ]
            for i, k in enumerate(sections, start=1):
                with st.expander(f"[{i}] {k}", expanded=(i == 1)):
                    st.write(ai[k])

            st.markdown("### Recommended controls")
            controls_list = [
                "Use a trained banksman during manoeuvres and reversing.",
                "Add temporary cones/signage; consider a convex mirror or visibility aids.",
                "Plan approach/egress to avoid reversing where practicable.",
            ]
            if stand_surface in ("gravel", "grass"):
                controls_list.append("Ensure firm, level stand surface (temporary mats if required).")
            if overhead_m is not None and overhead_m < CoP["overhead_info_m"]:
                controls_list.append("Confirm isolation/clearance for overhead power; position tanker outside bands.")
            if (auto.get("route_snap", {}).get("counts", {}).get("tunnel", 0) or 0) > 0:
                controls_list.append("⚠️ LPG tankers: tunnels/underpasses noted on approach — plan a compliant diversion.")
            for b in controls_list:
                st.write("• " + b)

            st.markdown("---")
            st.subheader("Access suitability (vehicle vs restrictions)")
            if stand_surface in ("asphalt", "concrete", "block paving") and turning_circle_m <= 22.0:
                st.success("PASS — no blocking restrictions detected for the selected vehicle.")
            else:
                st.info("ATTENTION — check turning area / bearing capacity for the selected vehicle.")

            # ---------- Route analysis (last 20 miles) ----------
            st.markdown("### Route analysis (last 20 miles) ↪")
            rs = auto.get("route_snap") or {}
            full = rs.get("full_miles")
            st.caption(f"Full route: {full if full is not None else 'n/a'} miles • Analysed segment: ~last 20 miles (nearest to site)")

            counts = rs.get("counts") or {}
            pretty = {
                "Gates/barriers": counts.get("barrier_gate", 0),
                "Bollards": counts.get("bollard", 0),
                "Tunnels/underpass": counts.get("tunnel", 0),
                "Fords": counts.get("ford", 0),
                "Low bridge / Height": counts.get("height", 0),
                "Weight limits": counts.get("weight", 0),
                "Width limits / Narrow": counts.get("width", 0) + counts.get("narrow", 0),
                "Level crossing / Rail": counts.get("rail", 0),
                "Private / No through": counts.get("private", 0),
                "Construction/closures": counts.get("construction", 0),
            }
            def badge(v:int)->str:
                if v <= 0: col="#289e41"
                elif v==1: col="#e0a10b"
                else: col="#cc2b2b"
                return f"<span style='background:{col};color:#fff;border-radius:8px;padding:2px 8px;font-weight:600'>{v}</span>"

            ccols = st.columns(2)
            idx = 0
            for k, v in pretty.items():
                with ccols[idx%2]:
                    st.markdown(f"<div style='margin-bottom:6px'><b>{k}:</b> {badge(v)}</div>", unsafe_allow_html=True)
                idx += 1

            if sum(pretty.values()) == 0:
                st.info("No notable flags detected within the last 20 miles (based on directions text). A manual review is still recommended.")

            st.markdown("---")
            st.subheader("Export")
            st.caption("Generate a one-page PDF summary (includes key metrics, separations, vehicle, AI commentary, controls, and map if available).")

            def build_pdf_bytes(ctx: Dict, ai_text: Dict[str,str], controls: list, map_file: Optional[str]) -> bytes:
                try:
                    from reportlab.pdfgen import canvas
                    from reportlab.lib.pagesizes import A4
                    from reportlab.lib.utils import ImageReader
                    from reportlab.lib import colors
                    from reportlab.pdfbase import pdfmetrics
                except Exception:
                    return b""

                buf = io.BytesIO()
                W, H = A4
                M = 38
                y = H - 46
                blue = colors.HexColor("#1f4e79")
                grey = colors.HexColor("#555555")

                c = canvas.Canvas(buf, pagesize=A4)

                def header(txt, size=16, col=blue):
                    nonlocal y
                    c.setFillColor(col); c.setFont("Helvetica-Bold", size)
                    c.drawString(M, y, txt); y -= (size + 6); c.setFillColor(colors.black)

                def text_line(txt, size=10, col=colors.black, bold=False):
                    nonlocal y
                    c.setFillColor(col); c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
                    c.drawString(M, y, txt); y -= (size + 3); c.setFillColor(colors.black)

                def wrap_paragraph(text, width=W-2*M, size=10, leading=12):
                    nonlocal y
                    c.setFont("Helvetica", size)
                    for para in (text or "").split("\n"):
                        words = para.split()
                        line = ""
                        for w in words:
                            test = (line + " " + w).strip() if line else w
                            if pdfmetrics.stringWidth(test, "Helvetica", size) <= width:
                                line = test
                            else:
                                c.drawString(M, y, line); y -= leading; line = w
                        if line:
                            c.drawString(M, y, line); y -= leading

                # Title
                header(f"LPG Pre-Check — ///{st.session_state.get('w3w','')}")
                addr = ctx["addr"]
                addr_line = ", ".join([p for p in [addr.get("road"), addr.get("city"), addr.get("postcode")] if p])
                if addr_line: text_line(addr_line, col=grey)
                if addr.get("display_name"): text_line(addr["display_name"], col=grey)
                if addr.get("hospital_name"):
                    if addr.get("hospital_distance_m") is not None:
                        km = addr["hospital_distance_m"] / 1000.0
                        text_line(f"Nearest hospital: {addr['hospital_name']} (~{km:.2f} km)", col=grey)
                    else:
                        text_line(f"Nearest hospital: {addr['hospital_name']}", col=grey)

                # Map
                if map_file and os.path.exists(map_file):
                    try:
                        ir = ImageReader(map_file)
                        from PIL import Image as PILImage
                        iw, ih = PILImage.open(map_file).size
                        maxw, maxh = W - 2*M, 240
                        sc = min(maxw/iw, maxh/ih)
                        c.drawImage(ir, M, y - ih*sc, width=iw*sc, height=ih*sc)
                        y -= ih*sc + 12
                    except Exception:
                        pass

                # Key metrics
                header("Key Metrics", size=13)
                text_line(f"Wind: {ctx['wind']['speed_mps']:.1f} m/s from {ctx['wind']['compass']}", col=grey)
                text_line(f"Slope: {ctx['slope_pct']:.1f}%   |   Approach avg/max: {ctx['approach_avg']:.1f}% / {ctx['approach_max']:.1f}%", col=grey)
                rr_txt = ("n/a" if ctx.get("route_ratio") in (None, "", "n/a") else f"{ctx['route_ratio']:.2f}× crow-fly")
                text_line(f"Route indirectness: {rr_txt}", col=grey)

                # Vehicle
                header("Vehicle", size=13)
                text_line(f"Type: {ctx['vehicle']['type']}", col=grey)
                text_line(f"Dimensions (L×W×H m): {ctx['vehicle']['length_m']:.1f} × {ctx['vehicle']['width_m']:.2f} × {ctx['vehicle']['height_m']:.2f}", col=grey)
                text_line(f"Turning circle: {ctx['vehicle']['turning_circle_m']:.1f} m", col=grey)

                # Separations
                header("Separations (~400 m)", size=13)
                feats = ctx["feats"]
                sep_lines = [
                    f"Building: {feats.get('building_m','n/a')}",
                    f"Boundary: {feats.get('boundary_m','n/a')}",
                    f"Road/footpath: {feats.get('road_m','n/a')}",
                    f"Drain/manhole: {feats.get('drain_m','n/a')}",
                    f"Overhead power lines: {feats.get('overhead_m','n/a')}",
                    f"Railway: {feats.get('rail_m','n/a')}",
                    f"Watercourse: {feats.get('water_m','n/a')}",
                    f"Land use: {feats.get('land_class','n/a')}",
                ]
                for l in sep_lines: text_line(l)

                # Risk
                header("Risk score", size=13)
                text_line(f"Total: {ctx['risk'].score:.1f}/100 → {ctx['risk'].status}", bold=True)
                for p, m in ctx['risk'].explain[:7]:
                    text_line(f"+{p} {m}")

                # Controls
                header("Recommended controls", size=13)
                for b in ctx["controls"]:
                    wrap_paragraph("• " + b)

                # AI sections
                for head in [
                    "Safety Risk Profile",
                    "Environmental Considerations",
                    "Access & Logistics",
                    "Overall Site Suitability",
                ]:
                    y -= 8
                    header(head, size=13)
                    wrap_paragraph(ai_text.get(head, ""))

                c.showPage()
                c.save()
                return buf.getvalue()

            ctx_pdf = {
                "addr": addr_edited,
                "wind": wind,
                "slope_pct": slope_pct,
                "approach_avg": approach_avg,
                "approach_max": approach_max,
                "route_ratio": route_ratio,
                "feats": feats,
                "risk": risk,
                "vehicle": {
                    "type": vehicle_type,
                    "length_m": veh_length_m,
                    "width_m": veh_width_m,
                    "height_m": veh_height_m,
                    "turning_circle_m": turning_circle_m,
                },
                "controls": controls_list,
            }

            pdf_bytes = build_pdf_bytes(ctx_pdf, ai, controls_list, map_path if MAPBOX_TOKEN else None)
            if pdf_bytes:
                st.download_button(
                    "📄 Download PDF report",
                    data=pdf_bytes,
                    file_name=f"precheck_{st.session_state.get('w3w','site')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            else:
                st.caption("PDF generation unavailable on this host (ReportLab not installed).")
