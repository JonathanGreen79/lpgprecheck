from __future__ import annotations
import os, io, math, json
from typing import Dict, List, Tuple, Optional

import requests
import streamlit as st
from PIL import Image, ImageDraw
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics

# ============================ App constants/secrets ============================

APP_TITLE    = "LPG Customer Tank — Pre-Check"
APP_PASSWORD = "Flogas2025"

W3W_API_KEY    = st.secrets.get("W3W_API_KEY","")
MAPBOX_TOKEN   = st.secrets.get("MAPBOX_TOKEN","")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY","")  # optional

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

OVERPASS = "https://overpass-api.de/api/interpreter"
UA = {"User-Agent": "LPG-Precheck-Pro/2.0"}

# =============================== Small utilities ==============================

def meters_per_degree(lat_deg: float) -> Tuple[float, float]:
    lat = math.radians(lat_deg)
    return (
        111132.92 - 559.82*math.cos(2*lat) + 1.175*math.cos(4*lat),
        111412.84*math.cos(lat) - 93.5*math.cos(3*lat),
    )

def ll_to_xy(lat0, lon0, lat, lon):
    mlat, mlon = meters_per_degree(lat0)
    return (lon - lon0)*mlon, (lat - lat0)*mlat

def dist_line(lat0, lon0, line: List[Tuple[float,float]]) -> Optional[float]:
    if not line or len(line) < 2:
        return None
    px, py = 0.0, 0.0
    verts = [ll_to_xy(lat0, lon0, la, lo) for la, lo in line]
    best = None
    for (ax,ay),(bx,by) in zip(verts, verts[1:]):
        apx, apy = px-ax, py-ay
        abx, aby = bx-ax, by-ay
        ab2 = abx*abx + aby*aby
        t = 0.0 if ab2 == 0 else max(0.0, min(1.0, (apx*abx + apy*aby)/ab2))
        cx, cy = ax + t*abx, ay + t*aby
        d = math.hypot(px-cx, py-cy)
        best = d if best is None else min(best, d)
    return best

def dist_poly(lat0, lon0, poly: List[Tuple[float,float]]) -> Optional[float]:
    if not poly or len(poly) < 2: return None
    return dist_line(lat0, lon0, poly + poly[:1])

def parse_num(s):
    if s is None: return None
    s = str(s).strip().lower()
    for u in ("m","meter","metre","meters","metres","t","ton","tons","tonne","tonnes"):
        if s.endswith(u):
            s = s[:-len(u)].strip()
            break
    s = s.replace(",",".")
    try: return float(s)
    except: return None

# =============================== External calls ===============================

def w3w_to_latlon(words: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        r = requests.get(
            "https://api.what3words.com/v3/convert-to-coordinates",
            params={"words": words, "key": W3W_API_KEY}, timeout=15
        )
        if r.status_code == 200:
            c = r.json().get("coordinates",{})
            return c.get("lat"), c.get("lng")
    except Exception:
        pass
    return None, None

def reverse_geocode(lat, lon) -> Dict:
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format":"jsonv2"},
            headers={"User-Agent":"LPG-Precheck"}, timeout=15
        )
        if r.status_code == 200:
            j = r.json(); a = j.get("address") or {}
            return {
                "display_name": j.get("display_name"),
                "road": a.get("road"), "postcode": a.get("postcode"),
                "city": a.get("town") or a.get("city") or a.get("village"),
                "county": a.get("county"),
                "state_district": a.get("state_district"),
                "local_authority": a.get("municipality") or a.get("county") or a.get("state_district"),
            }
    except Exception:
        pass
    return {}

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
    except Exception:
        return {"elements":[]}

def parse_osm(lat0, lon0, data) -> Dict:
    bpolys, roads, drains, manholes, plines, pnodes, rails, wlines, wpolys, land_polys = [],[],[],[],[],[],[],[],[],[]
    rest_ways, surf_ways = [],[]
    for el in data.get("elements",[]):
        t = el.get("type"); tags = el.get("tags",{}) or {}
        geom = el.get("geometry"); coords = [(g["lat"], g["lon"]) for g in (geom or [])]
        if tags.get("building") and t in ("way","relation"): bpolys.append(coords)
        elif tags.get("highway") and t=="way":
            roads.append(coords)
            if any(k in tags for k in ("maxheight","maxwidth","maxweight","hgv","access","oneway")):
                rest_ways.append({"tags":tags,"coords":coords})
            if ("surface" in tags) or ("smoothness" in tags):
                surf_ways.append({"tags":tags,"coords":coords})
        elif t=="way" and (tags.get("waterway")=="drain" or tags.get("tunnel")=="culvert"): drains.append(coords)
        elif t=="node" and (tags.get("man_made")=="manhole" or "manhole" in tags): manholes.append((el.get("lat"),el.get("lon")))
        elif t=="way" and tags.get("power")=="line": plines.append(coords)
        elif t=="node" and tags.get("power") in ("tower","pole"): pnodes.append((el.get("lat"),el.get("lon")))
        elif t=="way" and tags.get("railway") and tags.get("railway") not in ("abandoned","disused"): rails.append(coords)
        elif t=="way" and tags.get("waterway") in ("river","stream","ditch"): wlines.append(coords)
        elif t=="way" and tags.get("natural")=="water": wpolys.append(coords)
        elif t in ("way","relation") and tags.get("landuse"): land_polys.append({"tag":tags.get("landuse"),"coords":coords})

    vb = [p for p in bpolys if p and len(p)>=2]
    vr = [l for l in roads  if l and len(l)>=2]
    vd = [l for l in drains if l and len(l)>=2]
    vpl= [l for l in plines if l and len(l)>=2]
    vrl = [l for l in rails  if l and len(l)>=2]
    vwl = [l for l in wlines if l and len(l)>=2]
    vwp = [p for p in wpolys if p and len(p)>=2]

    d_build = min([dist_poly(lat0,lon0,p) for p in vb] or [None])
    d_road  = min([dist_line(lat0,lon0,l) for l in vr] or [None])

    # drains + manholes: treat nodes as tiny segments so the same dist function works
    dmix = [dist_line(lat0,lon0,l) for l in vd]
    if manholes:
        for la,lo in manholes:
            dmix.append(dist_line(lat0,lon0,[(la,lo),(la,lo)]))
    d_drain = min(dmix or [None])

    omix = [dist_line(lat0,lon0,l) for l in vpl]
    if pnodes:
        for la,lo in pnodes:
            omix.append(dist_line(lat0,lon0,[(la,lo),(la,lo)]))
    d_over = min(omix or [None])

    d_rail  = min([dist_line(lat0,lon0,l) for l in vrl] or [None])
    d_water = min(([dist_line(lat0,lon0,l) for l in vwl] + [dist_poly(lat0,lon0,p) for p in vwp]) or [None])

    land_counts={}
    for lp in land_polys:
        tag=lp["tag"]; land_counts[tag]=land_counts.get(tag,0)+1
    if land_counts:
        top=max(land_counts,key=lambda k: land_counts[k])
        if top in ("residential","commercial","retail"): land_class="Domestic/Urban"
        elif top in ("industrial","industrial;retail"):   land_class="Industrial"
        else: land_class="Rural/Agricultural"
    else:
        land_class="Domestic/Urban" if len(vb)>80 else ("Rural/Agricultural" if len(vb)<20 else "Mixed")

    return {
        "d_building_m": None if d_build is None else round(d_build,1),
        "d_road_m":     None if d_road  is None else round(d_road,1),
        "d_drain_m":    None if d_drain is None else round(d_drain,1),
        "d_overhead_m": None if d_over  is None else round(d_over,1),
        "d_rail_m":     None if d_rail  is None else round(d_rail,1),
        "d_water_m":    None if d_water is None else round(d_water,1),
        "land_class": land_class,
        "restrictions": rest_ways,
        "surfaces":     surf_ways,
    }

# ================================ Map helpers =================================

def fetch_map(lat, lon, zoom=17, size=(1000,750)):
    if not MAPBOX_TOKEN: return None
    try:
        w,h = size
        marker=f"pin-l+f30({lon},{lat})"; style="light-v11"
        url=(f"https://api.mapbox.com/styles/v1/mapbox/{style}/static/"
             f"{marker}/{lon},{lat},{zoom},0/{w}x{h}?access_token={MAPBOX_TOKEN}")
        r=requests.get(url,timeout=15); r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGBA")
    except Exception:
        return None

def overlay_rings(img: Image.Image, lat, zoom=17):
    if img is None: return None
    def mpp(lat,zoom): return 156543.03392*math.cos(math.radians(lat))/(2**zoom)
    scale=mpp(lat,zoom); cx,cy=img.width//2,img.height//2
    d=ImageDraw.Draw(img,"RGBA")
    for r,col in ((3,(220,0,0,180)),(6,(255,140,0,160))):
        px=max(1,int(r/scale)); d.ellipse((cx-px,cy-px,cx+px,cy+px),outline=col,width=4)
    return img

def make_map_card(words, lat, lon):
    img = fetch_map(lat,lon)
    if img is None:
        img = Image.new("RGBA",(1000,750),(245,247,250,255))
        ImageDraw.Draw(img).text((20,20),"Map unavailable", fill=(80,80,80))
    img = overlay_rings(img, lat, 17)
    out = f"map_{words.replace('.','_')}.png"
    img.save(out); return out

# =============================== Streamlit setup ==============================

st.set_page_config(page_title="LPG Pre-Check", page_icon="icon.png", layout="wide")
if "auth_ok" not in st.session_state: st.session_state.auth_ok = False

def header_with_icon(title: str):
    c1,c2 = st.columns([0.075, 0.925])
    with c1:
        try:
            st.image(Image.open("icon.png"), use_container_width=True)
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

# ================================ Sidebar =====================================

with st.sidebar:
    st.subheader("Location")
    words = st.text_input("what3words (word.word.word)", value=st.session_state.get("w3w",""))
    vehicle_name = st.selectbox("Vehicle", list(VEHICLES.keys()),
                                index=list(VEHICLES.keys()).index(DEFAULT_VEHICLE))
    run = st.button("Run Pre-Check", type="primary", use_container_width=True)

# ============================== Fetch auto data ===============================

auto, addr, lat, lon = {}, {}, None, None
if run and words.strip():
    with st.status("Fetching site data…", expanded=True) as s:
        la, lo = w3w_to_latlon(words.strip().lstrip("/"))
        if la is None:
            st.error("what3words lookup failed."); st.stop()
        lat, lon = la, lo
        s.update(label="Reverse geocoding…")
        addr = reverse_geocode(lat, lon)
        s.update(label="Reading OSM features…")
        osm  = overpass(lat, lon, int(CoP["poi_radius_m"]))
        feats = parse_osm(lat, lon, osm)
        # lightweight environment defaults
        auto = {
            "building_m": feats["d_building_m"], "boundary_m": None,
            "road_m": feats["d_road_m"], "drain_m": feats["d_drain_m"],
            "overhead_m": feats["d_overhead_m"], "rail_m": feats["d_rail_m"],
            "water_m": feats["d_water_m"], "land_class": feats["land_class"],
            "wind_mps": 6.8, "wind_deg": 191, "slope_pct": 3.5,
            "approach_avg": 0.9, "approach_max": 3.5, "rr": None,
            "restrictions": feats["restrictions"], "surfaces": feats["surfaces"]
        }
        st.session_state["w3w"] = words.strip()
        st.session_state["auto"] = auto
        st.session_state["addr"] = addr
        st.session_state["latlon"] = (lat, lon)
        st.session_state["vehicle"] = vehicle_name
        s.update(label="Auto data ready.", state="complete")

auto   = st.session_state.get("auto",{})
addr   = st.session_state.get("addr",{})
lat,lon= st.session_state.get("latlon",(None,None))
vehicle_name = st.session_state.get("vehicle", DEFAULT_VEHICLE)
vehicle = VEHICLES[vehicle_name]

# ============================== Form widgets ==================================

def nm_distance(label: str, key: str, auto_val: Optional[float], max_val=2000.0) -> Optional[float]:
    """One-line 'Not mapped' + number input that is always editable.
    The checkbox only decides whether we return None or the typed value,
    so it works inside st.form without relying on disabled=..."""
    # initial state
    if f"{key}_nm" not in st.session_state:
        st.session_state[f"{key}_nm"] = (auto_val is None)
    if f"{key}_val" not in st.session_state:
        st.session_state[f"{key}_val"] = 0.0 if auto_val is None else float(auto_val)

    c1, c2 = st.columns([0.78, 0.22])  # compact, single row
    with c1:
        # always editable; pre-filled with auto value (or 0.0 if unknown)
        val = st.number_input(
            label,
            min_value=0.0, max_value=float(max_val), step=0.1,
            value=float(st.session_state[f"{key}_val"]),
            key=f"{key}_val_input"
        )
    with c2:
        nm = st.checkbox("Not mapped", value=st.session_state[f"{key}_nm"], key=f"{key}_nm_chk")

    # persist the latest state
    st.session_state[f"{key}_val"] = val
    st.session_state[f"{key}_nm"] = nm

    # return None when 'Not mapped' is ticked; otherwise the typed float
    return None if nm else float(val)


# =============================== Risk & helpers ===============================

def restriction_notes(ways, vehicle) -> List[str]:
    out=[]
    for w in ways:
        t=w.get("tags",{})
        h=parse_num(t.get("maxheight"))
        wdt=parse_num(t.get("maxwidth"))
        wt=parse_num(t.get("maxweight"))
        if h is not None and h < vehicle["max_height_m"]: out.append(f"Max height {h} m")
        if wdt is not None and wdt < vehicle["max_width_m"]: out.append(f"Max width {wdt} m")
        if wt is not None and wt < vehicle["gross_weight_t"]: out.append(f"Max weight {wt} t")
        if (t.get("hgv") or "").lower() in ("no","destination"): out.append(f"HGV {t.get('hgv').lower()}")
        if (t.get("access") or "").lower() in ("no","private"): out.append(f"Access {t.get('access').lower()}")
        if (t.get("oneway") or "").lower() == "yes": out.append("One-way")
    seen=set(); out2=[]
    for s in out:
        if s not in seen:
            seen.add(s); out2.append(s)
    return out2

def surface_info(ways)->Dict:
    risky=0; samples=[]
    for w in ways:
        t=w.get("tags",{})
        surf=(t.get("surface") or "").lower()
        smooth=(t.get("smoothness") or "").lower()
        if any(k in surf for k in ("gravel","ground","dirt","grass","unpaved","compacted","sand")): risky+=1
        if any(k in smooth for k in ("bad","very_bad","horrible","impassable")): risky+=1
        if surf or smooth: samples.append(f"{surf or 'n/a'}/{smooth or 'n/a'}")
    return {"risky_count":risky, "samples":samples[:8]}

def risk_score(values: Dict, access: Dict, site: Dict) -> Dict:
    score=0.0; why=[]
    def add(x,msg): nonlocal score; score+=x; why.append(msg)
    def penal(dist, lim, msg, base=18, per=6, cap=40):
        if dist is None or dist >= lim: return
        add(min(cap, base + per*(lim-dist)), f"{msg} below {lim} m (≈ {dist:.1f} m)")

    penal(values["building_m"], CoP["to_building_m"], "Below 3.0 m")
    penal(values["road_m"],     CoP["to_ignition_m"], "Ignition proxy (road/footpath)")
    penal(values["drain_m"],    CoP["to_drain_m"],    "Drain/manhole <3 m")

    d_ov = values["overhead_m"]
    if d_ov is not None and d_ov < CoP["overhead_block_m"]: add(28,"Overhead in no-go band")
    elif d_ov is not None and d_ov < CoP["overhead_info_m"]: add(10,"Overhead within 10 m")

    d_rail=values["rail_m"]
    if d_rail is not None and d_rail < CoP["rail_attention_m"]: add(10,"Railway within 30 m")
    if values["water_m"] is not None and values["water_m"] < 50: add(8,"Watercourse within 50 m")

    if values["wind_mps"] is not None and values["wind_mps"] < CoP["wind_stagnant_mps"]: add(6,f"Low wind {values['wind_mps']:.1f} m/s")
    if values["slope_pct"] is not None and values["slope_pct"] >= CoP["slope_attention_pct"]: add(8,f"Local slope {values['slope_pct']:.1f}%")
    if values["approach_max"] is not None and values["approach_max"] >= CoP["approach_grade_warn_pct"]: add(12,f"Steep approach (max {values['approach_max']:.1f}%)")
    if values["route_ratio"] is not None and values["route_ratio"] > CoP["route_vs_crowfly_ratio_warn"]: add(10,f"Route length ≫ crow-fly ({values['route_ratio']:.2f}×)")

    veg_points = [0,2,4,6][min(max(site["veg_3m"],0),3)]
    if veg_points: add(veg_points, f"Vegetation near tank (level {site['veg_3m']})")

    if site["enclosure_sides"] > 1:
        add(min(12, 4*(site["enclosure_sides"]-1)), f"Enclosure effect: {site['enclosure_sides']} solid side(s)")

    if site["los_issue"]: add(8,"Restricted line-of-sight at stand")

    surface_penalty = {"asphalt":0,"concrete":0,"block paving":2,"gravel":4,"grass":6,"other":2}.get(site["stand_surface"],2)
    if surface_penalty: add(surface_penalty, f"Stand surface: {site['stand_surface']}")

    if site["open_field_m"] is not None:
        if site["open_field_m"] < 10: add(6, f"Open field very close ({site['open_field_m']:.1f} m)")
        elif site["open_field_m"] < 30: add(4, f"Open field nearby ({site['open_field_m']:.1f} m)")
        elif site["open_field_m"] < 60: add(2, f"Open field within 60 m ({site['open_field_m']:.1f} m)")

    if access["notes"]: add(min(12, 4*len(access["notes"])), "Access restrictions: "+", ".join(access["notes"]))
    if access["surface"]["risky_count"]>0: add(min(10,2*access["surface"]['risky_count']),
                                               f"Road surface flags={access['surface']['risky_count']}")

    score=round(min(100.0,score),1)
    status="PASS" if score<20 else ("ATTENTION" if score<50 else "BLOCKER")
    return {"score":score,"status":status,"explain":why[:7]}

def make_controls(status: str) -> List[str]:
    base = [
        "Use a trained banksman during manoeuvres and reversing.",
        "Add temporary cones/signage; consider a convex mirror or visibility aids.",
        "Plan approach/egress to avoid reversing where practicable.",
    ]
    if status!="PASS":
        base.append("Confirm separations to CoP1; protect drains within 3 m; manage vegetation and sightlines.")
    return base

# ================================ AI commentary ===============================

def ai_sections(context: Dict) -> Dict[str,str]:
    offline = {
        "Safety Risk Profile":
            f"Local slope {context['slope_pct']:.1f}%. Key separations (m): "
            f"bldg {context['building_m'] or 'n/a'}, boundary {context['boundary_m'] or 'n/a'}, "
            f"road {context['road_m'] or 'n/a'}, drain {context['drain_m'] or 'n/a'}, "
            f"overhead {context['overhead_m'] or 'n/a'}, rail {context['rail_m'] or 'n/a'}. "
            f"Wind {context['wind_mps']:.1f} m/s from {context['wind_deg']}°. "
            f"Heuristic {context['risk']['score']}/100 → {context['risk']['status']}.",
        "Environmental Considerations":
            f"Watercourse distance ~{context['water_m'] or 'n/a'} m; vegetation level {context.get('veg_3m',0)}. "
            "Protect drains during transfers; control vegetation and runoff.",
        "Access & Logistics":
            f"Approach avg/max {context['approach_avg']:.1f}/{context['approach_max']:.1f}%. "
            f"Stand surface {context.get('stand_surface','n/a')}; "
            f"line-of-sight {'restricted' if context.get('los_issue') else 'clear'}.",
        "Overall Site Suitability":
            "Site appears suitable with routine controls where PASS; when ATTENTION/BLOCKER, "
            "address separations, overheads, drains, vegetation, and sightlines before delivery."
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
Use the numeric context and be practical (≈120–180 words per section).

Context:
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
        if r.status_code!=200: return offline
        text=r.json()["choices"][0]["message"]["content"].strip()
        sections={"Safety Risk Profile":"","Environmental Considerations":"","Access & Logistics":"","Overall Site Suitability":""}
        current=None; mapping={"[1]":"Safety Risk Profile","[2]":"Environmental Considerations","[3]":"Access & Logistics","[4]":"Overall Site Suitability"}
        for line in text.splitlines():
            t=line.strip()
            for k,name in mapping.items():
                if t.startswith(k):
                    current=name; t=t[len(k):].lstrip(":- \t")
                    if t: sections[current]+=t+"\n"
                    break
            else:
                if current: sections[current]+=t+"\n"
        for k in sections:
            if not sections[k].strip(): sections[k]=offline[k]
        return sections
    except Exception:
        return offline

# ================================== PDF =======================================

def build_pdf(words, addr, lat, lon, values, rs, breakdown, controls, ai, map_png, site) -> str:
    W,H=A4; M=38; LEAD=12; y=H-46; PAGE_BOTTOM=40
    blue=colors.HexColor("#1f4e79"); grey=colors.HexColor("#555555")
    out=f"precheck_{words.replace('.','_')}.pdf"
    c=rl_canvas.Canvas(out, pagesize=A4)

    def ensure(h):
        nonlocal y
        if y-h < PAGE_BOTTOM:
            c.showPage(); y=H-46
    def text_line(txt,col=colors.black,font="Helvetica",size=10):
        nonlocal y; ensure(size+3); c.setFillColor(col); c.setFont(font,size); c.drawString(M,y,txt); y-=(size+3); c.setFillColor(colors.black)
    def header(txt,size=16,col=blue):
        nonlocal y; ensure(size+6); c.setFillColor(col); c.setFont("Helvetica-Bold",size); c.drawString(M,y,txt); y-=(size+6); c.setFillColor(colors.black)
    def section(txt,size=12,col=blue):
        nonlocal y; ensure(size+8); y-=4; c.setFillColor(col); c.setFont("Helvetica-Bold",size); c.drawString(M,y,txt); y-=(size+2); c.setFillColor(colors.black)
    def wrap_paragraph(text, width=W-2*M, font="Helvetica", size=10, leading=LEAD):
        nonlocal y; c.setFont(font,size)
        for para in text.split("\n"):
            p=para.rstrip()
            if not p: ensure(leading); y-=leading; continue
            words=p.split(); line=""
            for w in words:
                test=(line+" "+w).strip() if line else w
                if pdfmetrics.stringWidth(test,font,size)<=width: line=test
                else: ensure(leading); c.drawString(M,y,line); y-=leading; line=w
            if line: ensure(leading); c.drawString(M,y,line); y-=leading
    def bullet_list(items, bullet="•", font="Helvetica", size=10, leading=LEAD):
        nonlocal y
        for it in (items or []):
            ensure(leading); c.setFont(font,size); c.drawString(M,y,f"{bullet} {it}"); y-=leading

    # Title + icon
    header(f"LPG Pre-Check — ///{words}")
    try: c.drawImage(ImageReader("icon.png"), W-M-32, H-52, width=24, height=24, mask="auto")
    except Exception: pass

    line = ", ".join([p for p in [addr.get('road'),addr.get('city'),addr.get('postcode')] if p])
    if line: text_line(line, grey)
    if addr.get("display_name"): text_line(addr["display_name"], grey)

    # Map
    if map_png and os.path.exists(map_png):
        try:
            from PIL import Image as PILImage
            iw,ih=PILImage.open(map_png).size
            maxw,maxh=W-2*M,240
            sc=min(maxw/iw, maxh/ih)
            ensure(ih*sc+12)
            c.drawImage(ImageReader(map_png), M, y-ih*sc, width=iw*sc, height=ih*sc)
            y-=ih*sc+12
        except Exception:
            pass

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

    section("Site options")
    def fmto(x): 
        if x is None: return "n/a"
        return f"{x:.1f} m" if isinstance(x,(int,float)) else str(x)
    bullet_list([
        f"Vegetation within 3 m: level {site['veg_3m']}",
        f"Enclosure sides: {site['enclosure_sides']}",
        f"Restricted line-of-sight: {'yes' if site['los_issue'] else 'no'}",
        f"Stand surface: {site['stand_surface']}",
        f"Open field distance: {fmto(site['open_field_m'])}",
    ])

    section("Risk score")
    text_line(f"Total: {rs['score']}/100 → {rs['status']}")
    bullet_list(rs["explain"])

    section("Recommended controls")
    bullet_list(controls)

    for head in ["Safety Risk Profile","Environmental Considerations","Access & Logistics","Overall Site Suitability"]:
        ensure(LEAD); y-=LEAD
        section(head); wrap_paragraph(ai.get(head,""))

    c.showPage(); c.save()
    return out

# ============================== Render results ================================

if submitted:
    values = {
        "wind_mps": wind_mps, "wind_deg": wind_deg, "slope_pct": slope_pct,
        "approach_avg": approach_avg, "approach_max": approach_max, "route_ratio": route_ratio,
        "building_m": building_m, "boundary_m": boundary_m, "road_m": road_m, "drain_m": drain_m,
        "overhead_m": overhead_m, "rail_m": rail_m, "water_m": water_m, "land_use": land_use,
    }
    site = {
        "veg_3m": veg_3m, "enclosure_sides": enclosure_sides, "los_issue": los_issue,
        "stand_surface": stand_surface, "open_field_m": open_field_m, "notes": notes_txt
    }

    notes = restriction_notes(auto.get("restrictions",[]), VEHICLES[vehicle_name])
    surf  = surface_info(auto.get("surfaces",[]))
    access = {"notes": notes, "surface": surf}

    rs = risk_score(values, access, site)
    controls = make_controls(rs["status"])

    L,R = st.columns([0.48,0.52])
    with L:
        st.markdown("## Key metrics")
        m1,m2,m3 = st.columns(3)
        m1.metric("Wind (m/s)", f"{wind_mps:.1f}")
        m2.metric("Wind dir", f"{wind_deg}°")
        m3.metric("Slope (%)", f"{slope_pct:.1f}")
        n1,n2,n3 = st.columns(3)
        n1.metric("Approach avg", f"{approach_avg:.1f}%")
        n2.metric("Approach max", f"{approach_max:.1f}%")
        n3.metric("Indirectness", "—" if route_ratio is None else f"{route_ratio:.2f}×")

        st.markdown("### Separations (~400 m)")
        def fmt(v): return "—" if v is None else f"{v:.1f} m"
        c1,c2=st.columns(2)
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
            st.write(", ".join(notes))
        else:
            st.success("PASS — no blocking restrictions detected for the selected vehicle.")

        st.markdown("## Risk result")
        badge = {"PASS":"🟢 PASS","ATTENTION":"🟡 ATTENTION","BLOCKER":"🔴 BLOCKER"}[rs["status"]]
        st.metric("Score", f"{rs['score']}/100", badge)
        st.write("**Top contributing factors**")
        for m in rs["explain"]: st.write(f"- {m}")

        if lat and lon:
            try:
                map_png = make_map_card(st.session_state.get("w3w","site"), lat, lon)
                st.image(map_png, caption="Site map (3 m and 6 m rings)")
            except Exception:
                st.info("Map preview unavailable.")

    with R:
        st.markdown("## Site options")
        st.write(f"- Vegetation level (3 m): **{veg_3m}**")
        st.write(f"- Enclosure sides: **{enclosure_sides}**")
        st.write(f"- Line-of-sight restricted: **{'Yes' if los_issue else 'No'}**")
        st.write(f"- Stand surface: **{stand_surface}**")
        st.write(f"- Open field distance: **{'—' if open_field_m is None else f'{open_field_m:.1f} m'}**")
        if notes_txt: st.caption(f"Notes: {notes_txt}")

        st.markdown("## AI commentary")
        ctx = {**values, **site, "risk":rs}
        ai = ai_sections(ctx)
        with st.expander("[1] Safety Risk Profile", expanded=True):
            st.write(ai["Safety Risk Profile"])
        with st.expander("[2] Environmental Considerations"):
            st.write(ai["Environmental Considerations"])
        with st.expander("[3] Access & Logistics"):
            st.write(ai["Access & Logistics"])
        with st.expander("[4] Overall Site Suitability"):
            st.write(ai["Overall Site Suitability"])

        st.markdown("## Recommended controls")
        for c in controls: st.write(f"• {c}")

        pdf_map = None
        try: pdf_map = make_map_card(st.session_state.get("w3w","site"), lat, lon)
        except Exception: pass
        pdf_path = build_pdf(st.session_state.get("w3w","site"), addr, lat, lon, values, rs, rs["explain"], controls, ai, pdf_map, site)
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download PDF report", f, file_name=os.path.basename(pdf_path), type="secondary")

