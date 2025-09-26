# streamlit_app.py
import os, io, math, json, requests, re
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

import streamlit as st
import pandas as pd
import pydeck as pdk
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
# (simple gross weights for a couple of rules; used in route flags)
VEHICLE_GVW_T = {
    "3.5t Van": 3.5,
    "7.5t Rigid": 7.5,
    "18t Rigid": 18,
    "26t Rigid": 26,
    "Artic (44t)": 44,
    "LPG Tanker (Urban)": 26,
    "LPG Tanker (Long Rigid)": 26,
}

# ------------------------- Your depot list -------------------------
DEPOTS: List[Dict[str, float]] = [
    {"name":"Blaydon","lon":-1.744101,"lat":54.9756187},
    {"name":"Buckfastleigh","lon":-3.7833235,"lat":50.4869034},
    {"name":"Burton","lon":-1.559383,"lat":52.7768106},
    {"name":"Cairnhill","lon":-2.556494,"lat":57.3820238},
    {"name":"Carlisle","lon":-2.950747,"lat":54.898308},
    {"name":"Conwy","lon":-3.8574551,"lat":53.2876319},
    {"name":"Defford","lon":-2.161253,"lat":52.0883951},
    {"name":"Evanton","lon":-4.3069473,"lat":57.6723499},
    {"name":"Fawley","lon":-1.3778735,"lat":50.8490674},
    {"name":"Grangemouth","lon":-3.7212715,"lat":56.0224567},
    {"name":"Knowsley","lon":-2.8682086,"lat":53.4630737},
    {"name":"Launceston","lon":-4.3922685,"lat":50.6278182},
    {"name":"Leeds","lon":-1.5059716,"lat":53.7829647},
    {"name":"Llandarcy","lon":-3.8451237,"lat":51.6391619},
    {"name":"Ludham","lon":1.5499137,"lat":52.7241592},
    {"name":"Newport","lon":-2.9782096,"lat":51.5675306},
    {"name":"Paisley","lon":-4.424505,"lat":55.8532784},
    {"name":"Perth","lon":-3.4755149,"lat":56.4169938},
    {"name":"Peterborough","lon":-0.2142451,"lat":52.5776289},
    {"name":"Presteigne","lon":-2.9945084,"lat":52.2688513},
    {"name":"Rainham","lon":0.1698276,"lat":51.509824},
    {"name":"Sittingbourne","lon":0.7530866,"lat":51.3850881},
    {"name":"Skegness","lon":0.2666851,"lat":53.2554345},
    {"name":"Staveley","lon":-1.355699,"lat":53.2761913},
    {"name":"Stoke","lon":-2.1676655,"lat":52.9648079},
    {"name":"Swinton","lon":-2.2697802,"lat":55.7230767},
    {"name":"Witney","lon":-1.5068047,"lat":51.7926625},
    {"name":"Wrexham","lon":-2.9252352,"lat":53.0264358},
]

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

def hav_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2-lat1)
    dlmb = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2*R*math.asin(math.sqrt(a))

def cumulative_lengths_m(latlon: List[Tuple[float,float]]) -> List[float]:
    out=[0.0]
    for (a,b) in zip(latlon, latlon[1:]):
        out.append(out[-1]+hav_m(a[0],a[1],b[0],b[1]))
    return out

def nearest_depots(lat: float, lon: float, n: int = 3) -> List[Dict]:
    items=[]
    for d in DEPOTS:
        dm = hav_m(lat, lon, d["lat"], d["lon"])
        items.append({"name": d["name"], "lat": d["lat"], "lon": d["lon"], "miles": dm/1609.34})
    items.sort(key=lambda x: x["miles"])
    return items[:n]

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

# --- Nearest hospital (unchanged) ---
def nearest_hospital(lat: float, lon: float) -> Dict:
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

# ------------------------- Map utils -------------------------
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

# ------------------------- OSRM route + hazards (20 mi near site) -------------------------
def osrm_route(start_lon, start_lat, end_lon, end_lat) -> Dict:
    url = f"https://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}"
    params = {"overview":"full","geometries":"geojson","annotations":"false"}
    try:
        r = requests.get(url, params=params, headers=UA, timeout=25)
        r.raise_for_status()
        data = r.json()
        if not data.get("routes"):
            return {}
        route = data["routes"][0]
        coords = route["geometry"]["coordinates"]  # [lon,lat]
        return {"distance_m": route.get("distance"), "duration_s": route.get("duration"), "coords": coords}
    except Exception:
        return {}

def crop_last_miles(route_coords: List[List[float]], miles: float = 20.0) -> List[List[float]]:
    if not route_coords:
        return []
    latlon = [(c[1], c[0]) for c in route_coords]
    cum = cumulative_lengths_m(latlon)
    need = miles * 1609.34
    if cum[-1] <= need:
        return route_coords
    cut_start = 0
    target = cum[-1] - need
    # find first index where cum[i] >= target
    for i, v in enumerate(cum):
        if v >= target:
            cut_start = max(0, i-1)
            break
    return route_coords[cut_start:]

def overpass_hazards_along(coords_lonlat: List[List[float]], search_width_m: float = 35.0) -> List[Dict]:
    """Scan along a polyline and fetch OSM features that may be restrictions."""
    if not coords_lonlat:
        return []
    # sample every ~400 m along the line for a bbox based query
    latlon = [(c[1], c[0]) for c in coords_lonlat]
    cum = cumulative_lengths_m(latlon)
    pts = [0.0]
    total = cum[-1]
    step = 400.0
    x = step
    while x < total:
        pts.append(x)
        x += step
    pts.append(total)

    # helper: find approximate lat/lon at distance d
    def point_at(d):
        for i in range(len(cum)-1):
            if cum[i] <= d <= cum[i+1]:
                t = 0 if cum[i+1]==cum[i] else (d-cum[i])/(cum[i+1]-cum[i])
                la = latlon[i][0] + t*(latlon[i+1][0]-latlon[i][0])
                lo = latlon[i][1] + t*(latlon[i+1][1]-latlon[i][1])
                return la, lo
        return latlon[-1][0], latlon[-1][1]

    hazards=[]
    seen=set()
    for d in pts:
        la, lo = point_at(d)
        # radius ~ search_width_m
        rad = max(30, min(80, int(search_width_m)))
        q = f"""
[out:json][timeout:40];
(
  way(around:{rad},{la},{lo})["tunnel"];
  way(around:{rad},{la},{lo})["hazmat"];
  way(around:{rad},{la},{lo})["maxheight"];
  way(around:{rad},{la},{lo})["maxwidth"];
  way(around:{rad},{la},{lo})["maxweight"];
  way(around:{rad},{la},{lo})["barrier"];
  node(around:{rad},{la},{lo})["barrier"];
  way(around:{rad},{la},{lo})["ford"];
  way(around:{rad},{la},{lo})["highway"="living_street"];
  way(around:{rad},{la},{lo})["surface"~"gravel|ground|unpaved|grass|dirt"];
  node(around:{rad},{la},{lo})["railway"="level_crossing"];
);
out tags center 50;
"""
        try:
            r = requests.post(OVERPASS, data={"data": q}, headers=UA, timeout=60)
            r.raise_for_status()
            eles = r.json().get("elements", [])
        except Exception:
            eles = []
        for el in eles:
            eid = f"{el.get('type')}:{el.get('id')}"
            if eid in seen: 
                continue
            seen.add(eid)
            tags = el.get("tags", {}) or {}
            cla, clo = (el.get("lat"), el.get("lon")) if el.get("type")=="node" else ( (el.get("center") or {}).get("lat"), (el.get("center") or {}).get("lon") )
            if cla is None or clo is None:
                continue
            hazards.append({"id":eid,"tags":tags,"lat":cla,"lon":clo})
    return hazards

def classify_issue(tags: Dict[str,str], veh_h: float, veh_w: float, veh_t: float, is_lpg: bool) -> Tuple[str,str]:
    """Return (severity, pretty_label) from tags."""
    # tunnels are a hard BLOCKER for LPG
    if tags.get("tunnel") in ("yes","building_passage","culvert"):
        return ("BLOCKER","Tunnel (LPG prohibited)" if is_lpg else "Tunnel")

    # heights
    if "maxheight" in tags:
        # parse meters or imperial like 13'6"
        v = tags["maxheight"]
        mh = None
        try:
            if "m" in v:
                mh = float(re.findall(r"([0-9]+(?:\.[0-9]+)?)", v)[0])
            elif "'" in v:
                m_ = re.findall(r"(\d+)'(?:\s*(\d+))?\"", v)
                if m_:
                    ft = int(m_[0][0]); inch = int(m_[0][1]) if m_[0][1] else 0
                    mh = (ft*12 + inch) * 0.0254
            else:
                mh = float(re.findall(r"([0-9]+(?:\.[0-9]+)?)", v)[0])
        except Exception:
            pass
        if mh is not None:
            return ("BLOCKER" if veh_h>mh else "ATTENTION", f"Height restriction {mh:.2f} m")

    if "maxwidth" in tags:
        try:
            mw = float(re.findall(r"([0-9]+(?:\.[0-9]+)?)", tags["maxwidth"])[0])
            return ("BLOCKER" if veh_w>mw else "ATTENTION", f"Width restriction {mw:.2f} m")
        except Exception:
            return ("ATTENTION","Width restriction")

    if "maxweight" in tags:
        try:
            wt = float(re.findall(r"([0-9]+(?:\.[0-9]+)?)", tags["maxweight"])[0])
            return ("BLOCKER" if veh_t>wt else "ATTENTION", f"Weight limit {wt:.1f} t")
        except Exception:
            return ("ATTENTION","Weight limit")

    if tags.get("barrier") in ("gate","lift_gate"):
        return ("ATTENTION","Gate")
    if tags.get("barrier") in ("bollard","block"):
        return ("ATTENTION","Bollard")
    if tags.get("barrier") == "height_restrictor":
        return ("ATTENTION","Height restrictor")

    if tags.get("railway") == "level_crossing":
        return ("ATTENTION","Level crossing")

    if tags.get("ford") == "yes":
        return ("ATTENTION","Ford")

    if tags.get("hazmat") in ("no","designated"):
        return ("ATTENTION","Hazmat restriction")

    if tags.get("highway") == "living_street":
        return ("ATTENTION","Unsuitable road class")

    if tags.get("surface") in ("gravel","ground","unpaved","grass","dirt"):
        return ("ATTENTION","Poor surface")

    return ("INFO","Restriction")

# ------------------------- AI commentary -------------------------
def _offline_ai_sections(ctx: Dict) -> Dict[str, str]:
    feats, wind = ctx["feats"], ctx["wind"]
    slope_pct = ctx["slope_pct"] or 0.0
    risk = ctx["risk"]
    sides = ctx["enclosure_sides"]
    los = ctx["los_issue"]
    veg = ctx["veg_3m"]
    land = feats.get("land_class", "n/a")
    rs = ctx.get("route_summary", {})
    rtxt = ""
    if rs:
        rtxt = f" Nearest depot: {rs.get('nearest_depot','n/a')} ({rs.get('nearest_depot_miles','n/a')} mi). Last 20 mi flags: {rs.get('flags_text','none')}."

    s1 = (
        f"The local slope is {slope_pct:.1f}%. Key separations (m): "
        f"building {feats.get('building_m','n/a')}, boundary n/a, road {feats.get('road_m','n/a')}, "
        f"drain {feats.get('drain_m','n/a')}, overhead {feats.get('overhead_m','n/a')}, rail {feats.get('rail_m','n/a')}. "
        f"Wind {wind.get('speed_mps') or 0:.1f} m/s from {wind.get('compass') or 'n/a'}. "
        f"Heuristic {risk.score:.1f}/100 → {risk.status}. Drivers: "
        + "; ".join([f"{p} {m}" for p, m in risk.explain[:5]]) + "."
    )
    s2 = (
        f"Flood Low (No mapped watercourse nearby). Watercourse ~{feats.get('water_m','n/a')} m; "
        f"drains/manholes {feats.get('drain_m','n/a')} m. Land use {land}. Vegetation within 3 m: level {veg}."
    )
    s3 = (
        f"Access lines of sight: {'restricted' if los else 'clear'}; "
        f"enclosure effect {sides} solid side(s).{rtxt} "
        f"Validate signage/restrictions; provide sound hardstanding and clear sightlines."
    )
    s4 = (
        "Attention required — ensure separation compliance, ignition control, drainage protection, "
        "and safe approach/egress."
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
        "route_summary": ctx.get("route_summary", {}),
    }

    system = (
        "You are a safety and logistics assistant writing concise LPG site pre-check commentary. "
        "Audience is field engineers; keep it professional, UK terminology. "
        "Output exactly four sections with these headings on their own lines followed by a single paragraph: "
        "### Safety Risk Profile\n### Environmental Considerations\n### Access & Logistics\n### Overall Site Suitability"
    )
    user = (
        "Using this site context, write those four sections (short, specific, actionable). "
        "Focus Access & Logistics on approach/egress, nearest depot, and any route blockers (e.g., tunnels for LPG). "
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
        found={}
        ms=list(head_line_re.finditer(txt))
        if not ms:
            return found
        for i,m in enumerate(ms):
            k=m.group(1); s=m.end(); e=ms[i+1].start() if i+1<len(ms) else len(txt)
            found[k]=txt[s:e].strip()
        return found
    found = slice_by_matches(text)
    if sum(bool(v) for v in found.values()) < 2:
        inline = text
        for k in section_keys:
            inline = re.sub(rf"(?i)(?<!\n)\s+({re.escape(k)})\s*[:\-–—]\s*", r"\n\1: ", inline)
        found = slice_by_matches(inline)
    for k in section_keys:
        v=(found.get(k) or "").strip().lstrip(" .,:;–—-")
        sections[k]=v or "No additional notes for this section."
    return sections

def ai_sections(ctx: Dict) -> Dict[str, str]:
    if OPENAI_API_KEY:
        try:
            return _online_ai_sections(ctx)
        except Exception:
            pass
    return _offline_ai_sections(ctx)

# ------------------------- UI helpers -------------------------
def nm_distance(label: str, key: str, auto_val: Optional[float], max_val: float = 2000.0, seed_tag: Optional[str] = None) -> Optional[float]:
    tag = seed_tag or "__default__"
    if st.session_state.get(f"{key}__seed") != tag:
        st.session_state[f"{key}__nm"]  = (auto_val is None)
        st.session_state[f"{key}__val"] = 0.0 if auto_val is None else float(auto_val)
        st.session_state[f"{key}__seed"] = tag

    c1, c2 = st.columns([0.78, 0.22])
    with c1:
        val = st.number_input(label, min_value=0.0, max_value=float(max_val), step=0.1,
                              value=float(st.session_state[f"{key}__val"]), key=f"{key}__val_input")
    with c2:
        nm = st.checkbox("Not mapped", value=st.session_state[f"{key}__nm"], key=f"{key}__nm_chk")
    st.session_state[f"{key}__val"] = val
    st.session_state[f"{key}__nm"]  = nm
    return None if nm else float(val)

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

# Reset
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

            # nearest depots
            dep_3 = nearest_depots(lat, lon, n=3)

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
                "nearest_depots": dep_3,
            }
            st.success("Auto data ready.")

# --- Use persisted auto for the form and results ---
auto = st.session_state.get("auto", {})
if auto:
    seed = st.session_state.get("w3w", "")
    submitted = False

    st.markdown("### Edit & confirm")
    with st.form("inputs"):
        # ---------------- Location & address ----------------
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

        # ---------------- Nearest Depot & Logistics (quick) ----------------
        st.markdown("---")
        st.subheader("Nearest Depot & Logistics")
        nd = nearest_depots(auto["lat"], auto["lon"], 3)
        nearest_name = nd[0]["name"]
        nearest_mi = nd[0]["miles"]

        c1n, c2n = st.columns([0.6, 0.4])
        with c1n:
            st.text_input("Nearest depot", nearest_name, disabled=True)
        with c2n:
            st.text_input("Distance (miles)", f"{nearest_mi:.1f} miles", disabled=True)

        st.caption("Top 3 nearest depots (crow-fly)")
        rows = [{"Depot": d["name"], "Distance (miles)": d["miles"]} for d in nd]
        df = pd.DataFrame(rows)
        df.index = range(1, len(df)+1)
        df.index.name = ""
        df["Distance (miles)"] = df["Distance (miles)"].map(lambda x: f"{x:.1f}")
        st.table(df)

        # --- Light route analysis preview (last 20 mi) with clickable issues ---
        # Build route from nearest depot to site (OSRM)
        route = osrm_route(
            start_lon=next(d["lon"] for d in DEPOTS if d["name"]==nearest_name),
            start_lat=next(d["lat"] for d in DEPOTS if d["name"]==nearest_name),
            end_lon=auto["lon"], end_lat=auto["lat"]
        )
        cropped = crop_last_miles(route.get("coords", []), 20.0)
        hazards = overpass_hazards_along(cropped, 35.0)

        # Classify hazards for this vehicle
        is_lpg = "LPG" in vehicle_type
        issues = []
        # compute mile-from-site for each hazard
        latlon = [(c[1], c[0]) for c in cropped]
        cum = cumulative_lengths_m(latlon) if latlon else [0.0]
        for h in hazards:
            sev, lab = classify_issue(h.get("tags",{}), veh_height_m, veh_width_m, VEHICLE_GVW_T.get(vehicle_type, 26), is_lpg)
            # find nearest point along cropped to compute distance from site end
            dmin = 1e12
            if latlon:
                for (a,b) in zip(latlon, latlon[1:]):
                    dm = hav_m(h["lat"], h["lon"], a[0], a[1])
                    if dm<dmin: dmin=dm
            mile_from_site = (cum[-1] - dmin)/1609.34 if latlon else None
            issues.append({"severity":sev,"label":lab,"lat":h["lat"],"lon":h["lon"],"mile":mile_from_site})

        issues.sort(key=lambda x: (0 if x["severity"]=="BLOCKER" else 1, x["mile"] or 1e9))
        st.markdown("#### Route analysis (last 20 miles)")
        if route:
            st.caption(f"Full route: {(route['distance_m']/1609.34):.1f} miles, {(route['duration_s']/60):.0f} min • Analysed segment: {( (cum[-1] if latlon else 0)/1609.34):.1f} miles (nearest to site)")
        else:
            st.caption("Analysed segment near site (OSRM route not fully available).")

        if not issues:
            st.info("No notable restrictions found on the last 20 miles.")
        else:
            # clickable Issue -> show inline popup map
            for idx, it in enumerate(issues, start=1):
                cols = st.columns([0.12, 0.40, 0.20, 0.28])
                with cols[0]:
                    st.write(f"**{idx}**")
                with cols[1]:
                    key_btn = f"iss_{idx}"
                    if st.button(it["label"], key=key_btn):
                        st.session_state[f"show_map_{key_btn}"] = not st.session_state.get(f"show_map_{key_btn}", False)
                with cols[2]:
                    st.write("**" + it["severity"] + "**")
                with cols[3]:
                    st.write(f"{(it['mile'] or 0):.1f} mi from site")

                # inline popup map
                if st.session_state.get(f"show_map_{key_btn}", False) and cropped:
                    layer1 = pdk.Layer(
                        "PathLayer",
                        data=[{"path": [{"lon":c[0],"lat":c[1]} for c in cropped]}],
                        get_path="path", get_width=4, width_min_pixels=2
                    )
                    layer2 = pdk.Layer(
                        "ScatterplotLayer",
                        data=[{"lat": it["lat"], "lon": it["lon"]}],
                        get_position='[lon, lat]', get_radius=25, pickable=False
                    )
                    view = pdk.ViewState(latitude=it["lat"], longitude=it["lon"], zoom=14)
                    st.pydeck_chart(pdk.Deck(layers=[layer1, layer2], initial_view_state=view, map_style="mapbox://styles/mapbox/light-v11"))

        notes_txt = st.text_input("Notes (vegetation / sightlines / special instructions)", value="", key="notes_input")

        submitted = st.form_submit_button("Confirm & assess", type="primary", key="confirm_btn")

    if submitted:
        feats = {
            "building_m": building_m, "road_m": road_m, "drain_m": drain_m,
            "overhead_m": overhead_m, "rail_m": rail_m, "water_m": water_m,
            "land_class": land_use,
        }
        wind = {"speed_mps": wind_mps, "deg": wind_deg,
                "compass": ["N","NE","E","SE","S","SW","W","NW"][round((wind_deg%360)/45)%8]}

        risk = risk_score(
            feats=feats, wind=wind, slope_pct=slope_pct,
            enclosure_sides=enclosure_sides, los_issue=los_issue,
            veg_3m=veg_3m, open_field_m=open_field_m
        )

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

            # Recompute nearest depots for the record
            nd = nearest_depots(auto["lat"], auto["lon"], 3)
            st.markdown("#### Nearest depots")
            df2 = pd.DataFrame([{"Depot": d["name"], "Distance (miles)": d["miles"]} for d in nd])
            df2.index = range(1, len(df2)+1); df2.index.name=""
            df2["Distance (miles)"] = df2["Distance (miles)"].map(lambda x: f"{x:.1f}")
            st.table(df2)

            st.markdown("### Risk result")
            badge = ("✅ PASS" if risk.status=="PASS"
                     else "🟡 ATTENTION" if risk.status=="ATTENTION"
                     else "🟥 BLOCKER")
            st.subheader(f"{risk.score:.1f}/100  {badge}")
            st.markdown("#### Top contributing factors")
            for p, m in risk.explain[:6]:
                st.write(f"+{p} {m}")

            if MAPBOX_TOKEN:
                st.markdown("#### Map")
                img = fetch_map(st.session_state["auto"]["lat"], st.session_state["auto"]["lon"])
                if img:
                    st.image(img, caption="Map (centered on W3W)", use_container_width=True)

        # ---------------- RIGHT: AI + Logistics & Routing (rich) ----------------
        with right:
            # Build route + hazards again using nearest depot
            nd = nearest_depots(auto["lat"], auto["lon"], 1)
            nearest_name = nd[0]["name"]
            dep_lon = next(d["lon"] for d in DEPOTS if d["name"]==nearest_name)
            dep_lat = next(d["lat"] for d in DEPOTS if d["name"]==nearest_name)
            route = osrm_route(dep_lon, dep_lat, auto["lon"], auto["lat"])
            cropped = crop_last_miles(route.get("coords", []), 20.0)
            hazards = overpass_hazards_along(cropped, 35.0)

            is_lpg = "LPG" in vehicle_type
            issues=[]
            latlon=[(c[1],c[0]) for c in cropped] if cropped else []
            cum=cumulative_lengths_m(latlon) if latlon else [0.0]
            for h in hazards:
                sev, lab = classify_issue(h.get("tags",{}), veh_height_m, veh_width_m, VEHICLE_GVW_T.get(vehicle_type,26), is_lpg)
                dmin=1e12
                if latlon:
                    for a in latlon:
                        dm = hav_m(h["lat"], h["lon"], a[0], a[1])
                        if dm<dmin: dmin=dm
                mile_from_site=(cum[-1]-dmin)/1609.34 if latlon else None
                issues.append({"severity":sev,"label":lab,"lat":h["lat"],"lon":h["lon"],"mile":mile_from_site})
            issues.sort(key=lambda x: (0 if x["severity"]=="BLOCKER" else 1, x["mile"] or 1e9))

            # Route summary -> feed AI
            flags_text = ", ".join([i["label"] for i in issues[:4]]) if issues else "none"
            route_summary = {
                "nearest_depot": nearest_name,
                "nearest_depot_miles": f"{nd[0]['miles']:.1f}",
                "full_miles": f"{(route.get('distance_m',0)/1609.34):.1f}" if route else "n/a",
                "last20_mi_issue_count": len(issues),
                "flags_text": flags_text,
            }

            st.markdown("### AI commentary")
            ctx = {
                "feats": feats, "wind": wind, "slope_pct": slope_pct,
                "enclosure_sides": enclosure_sides, "los_issue": los_issue,
                "veg_3m": veg_3m, "risk": risk,
                "route_summary": route_summary,
            }
            ai = ai_sections(ctx)
            sections = [
                "Safety Risk Profile",
                "Environmental Considerations",
                "Access & Logistics",
                "Overall Site Suitability",
            ]
            for i, k in enumerate(sections, start=1):
                with st.expander(f"[{i}] {k}", expanded=(i == 1)):
                    st.write(ai[k])

            st.markdown("### Logistics & Routing")
            if route and cropped:
                # Big map
                layer_route = pdk.Layer(
                    "PathLayer",
                    data=[{"path":[{"lon":c[0],"lat":c[1]} for c in route["coords"]]}],
                    get_path="path", get_width=5, width_min_pixels=2
                )
                layer_last20 = pdk.Layer(
                    "PathLayer",
                    data=[{"path":[{"lon":c[0],"lat":c[1]} for c in cropped]}],
                    get_path="path", get_width=7, width_min_pixels=3
                )
                layer_issues = pdk.Layer(
                    "ScatterplotLayer",
                    data=[{"lon":i["lon"],"lat":i["lat"]} for i in issues],
                    get_position='[lon, lat]', get_radius=35
                )
                view = pdk.ViewState(latitude=auto["lat"], longitude=auto["lon"], zoom=9)
                st.pydeck_chart(pdk.Deck(layers=[layer_route,layer_last20,layer_issues], initial_view_state=view, map_style="mapbox://styles/mapbox/light-v11"))

                st.caption(f"Depot: {nearest_name} • Route: {(route['distance_m']/1609.34):.1f} miles, {(route['duration_s']/60):.0f} min • Analysed: {(cumulative_lengths_m([(c[1],c[0]) for c in cropped])[-1]/1609.34):.1f} miles")
            else:
                st.info("Route overview not available.")

            if issues:
                st.markdown("#### Last 20 miles — key restrictions")
                df_iss = pd.DataFrame(
                    [{"Severity":i["severity"], "Issue":i["label"], "Mile from site": f"{(i['mile'] or 0):.1f}"} for i in issues]
                )
                df_iss.index = range(1, len(df_iss)+1); df_iss.index.name=""
                st.table(df_iss)
            else:
                st.write("No notable restrictions detected.")

            st.markdown("---")
            st.subheader("Access suitability (vehicle vs restrictions)")
            if stand_surface in ("asphalt", "concrete", "block paving") and turning_circle_m <= 22.0 and not any(i["severity"]=="BLOCKER" for i in issues):
                st.success("PASS — route and stand look serviceable for the selected vehicle.")
            else:
                msgs=[]
                if stand_surface not in ("asphalt","concrete","block paving"):
                    msgs.append("stand surface")
                if turning_circle_m>22.0:
                    msgs.append("turning area")
                if any(i["severity"]=="BLOCKER" for i in issues):
                    msgs.append("route blockers")
                st.info("ATTENTION — check " + ", ".join(msgs) + ".")

            # ---------- PDF export ----------
            st.markdown("### Export")
            st.caption("Generate a one-page PDF summary (key metrics, separations, vehicle, AI commentary, route summary, controls, and map if available).")

            map_path = None
            if MAPBOX_TOKEN:
                img = fetch_map(st.session_state["auto"]["lat"], st.session_state["auto"]["lon"])
                if img:
                    map_path = "map_preview.png"
                    try:
                        img.save(map_path)
                    except Exception:
                        map_path = None

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
                for l in [
                    f"Building: {feats.get('building_m','n/a')}",
                    f"Boundary: {feats.get('boundary_m','n/a')}",
                    f"Road/footpath: {feats.get('road_m','n/a')}",
                    f"Drain/manhole: {feats.get('drain_m','n/a')}",
                    f"Overhead power lines: {feats.get('overhead_m','n/a')}",
                    f"Railway: {feats.get('rail_m','n/a')}",
                    f"Watercourse: {feats.get('water_m','n/a')}",
                    f"Land use: {feats.get('land_class','n/a')}",
                ]: text_line(l)

                # Risk
                header("Risk score", size=13)
                text_line(f"Total: {ctx['risk'].score:.1f}/100 → {ctx['risk'].status}", bold=True)
                for p, m in ctx['risk'].explain[:7]:
                    text_line(f"+{p} {m}")

                # Route summary
                rs = ctx.get("route_summary", {})
                header("Logistics & Route", size=13)
                text_line(f"Depot: {rs.get('nearest_depot','n/a')}  |  Depot distance: {rs.get('nearest_depot_miles','n/a')} mi", col=grey)
                text_line(f"Route length: {rs.get('full_miles','n/a')} mi", col=grey)
                wrap_paragraph("Last 20 mi flags: " + (rs.get("flags_text","none") or "none"))

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

            # Controls
            controls_list = [
                "Use a trained banksman during manoeuvres and reversing.",
                "Add temporary cones/signage; consider a convex mirror or visibility aids.",
                "Plan approach/egress to avoid reversing where practicable.",
            ]
            if stand_surface in ("gravel", "grass"):
                controls_list.append("Ensure firm, level stand surface (temporary mats if required).")
            if overhead_m is not None and overhead_m < CoP["overhead_info_m"]:
                controls_list.append("Confirm isolation/clearance for overhead power; position tanker outside bands.")
            if any(i["label"].lower().startswith("tunnel") for i in issues):
                controls_list.append("Re-route to avoid tunnels; LPG tankers prohibited in tunnels.")

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
                "route_summary": {
                    "nearest_depot": nearest_name,
                    "nearest_depot_miles": f"{nearest_depots(auto['lat'],auto['lon'],1)[0]['miles']:.1f}",
                    "full_miles": f"{(route.get('distance_m',0)/1609.34):.1f}" if route else "n/a",
                    "flags_text": ", ".join([i["label"] for i in issues[:6]]) if issues else "none",
                }
            }

            pdf_bytes = build_pdf_bytes(ctx_pdf, ai, controls_list, map_path)
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
