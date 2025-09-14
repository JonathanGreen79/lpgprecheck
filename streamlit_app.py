import io, json, math, re, requests
from typing import Dict, List, Tuple, Optional, Any
from types import SimpleNamespace
import streamlit as st

# ─────────────────────────── Password Gate (Step 1) ───────────────────────────
PASSWORD = "Flogas2025"  # hard-coded per your request

st.set_page_config(page_title="LPG Customer Tank — Pre-Check", page_icon="icon.png", layout="wide")

def require_password():
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False

    if st.session_state.auth_ok:
        return True

    st.title("🔒 LPG Pre-Check — Restricted")
    st.caption("Step 1 — Enter password to continue.")
    pwd = st.text_input("Access password", type="password")
    col1, col2 = st.columns([0.25, 0.75])
    ok = col1.button("Unlock", type="primary", use_container_width=True)
    if ok:
        if pwd == PASSWORD:
            st.session_state.auth_ok = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

if not st.session_state.get("auth_ok", False):
    require_password()

# ─────────────────────────── Secrets (optional) ───────────────────────────────
W3W_API_KEY    = st.secrets.get("W3W_API_KEY", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
MAPBOX_TOKEN   = st.secrets.get("MAPBOX_TOKEN", "")

# ─────────────────────────── Config ───────────────────────────────────────────
CFG = {
    "cop": {
        "to_building_m": 3.0, "to_boundary_m": 3.0, "to_ignition_m": 3.0, "to_drain_m": 3.0,
        "overhead_info_m": 10.0, "overhead_block_m": 5.0, "rail_attention_m": 30.0,
        "poi_radius_m": 400.0, "wind_stagnant_mps": 1.0, "slope_attention_pct": 3.0,
        "approach_grade_warn_pct": 18.0, "route_vs_crowfly_ratio_warn": 1.7,
    },
    "bands": {"pass_lt": 20, "attention_lt": 50},
    "weights": {
        "building":   {"base":18,"per_m":6,"cap":40},
        "boundary":   {"base":14,"per_m":5,"cap":32},
        "ignition":   {"base":18,"per_m":6,"cap":40},
        "drain":      {"base":18,"per_m":6,"cap":40},
        "overhead_info": 10,
        "overhead_block": 28,
        "rail_near": 10,
        "water_near": 8,
        "low_wind": 6,
        "slope_ge6": 12,
        "slope_ge3": 8,
        "approach_steep": 12,
        "route_detour": 10,
        "surface_flag_per": 2,
        "excel_ignitions_3m": 6,
        "fence_side": 4,
        "los_restricted": 8,
        "vegetation": 6
    },
    "controls": {
        "drain_within_3m": {
            "when": "features.d_drain_m is not None and features.d_drain_m < cop.to_drain_m",
            "actions": [
                "Fit drain cover/insert or isolate drain during transfers.",
                "Install spill containment to prevent entry to drainage."
            ]
        },
        "onsite_ignitions": {
            "when": "answers.onsite_ignitions is True",
            "actions": [
                "Relocate ignition source or maintain ≥3 m separation.",
                "Mark ATEX zone and enforce no-smoking/no-ignition controls."
            ]
        },
        "surface_soft": {
            "when": "answers.surface_type in ['gravel','grass']",
            "actions": [
                "Upgrade stand area to hardstanding (concrete/asphalt) for stability and spill control."
            ]
        },
        "fence_enclosure_controls": {
            "when": "int(answers.fence_sides) >= 2",
            "actions": [
                "Avoid enclosed pockets around the tank: keep ventilation gaps or louvred panels.",
                "Ensure at least one low-level vented side to disperse heavier-than-air gas."
            ]
        },
        "vegetation_clearance": {
            "when": "(answers.veg_within3m_pct or 0) > 0 or bool(answers.vegetation_notes)",
            "actions": [
                "Maintain ≥3 m vegetation clearance around the vessel; remove leaf/brush build-up.",
                "Keep hardstanding clear for footing and spill control."
            ]
        },
        "los_restricted_controls": {
            "when": "answers.los_restricted is True",
            "actions": [
                "Use a trained banksman during manoeuvres and reversing.",
                "Add temporary cones/signage; consider a convex mirror or visibility aids.",
                "Plan approach/egress to avoid reversing where practicable."
            ]
        }
    },
    "vehicle_defaults": {"height_m":3.6, "width_m":2.55, "gross_weight_t":18.0, "length_m":10.0},
}

# ─────────────────────────── Styling / Header ────────────────────────────────
st.markdown("""
<style>
.smallcaps{font-variant:all-small-caps;letter-spacing:.04em;color:#6b7280}
.kv{display:flex;justify-content:space-between;gap:.75rem;padding:.5rem .75rem;border:1px solid #eee;border-radius:.5rem;margin:.25rem 0;background:#fafafa}
.kv-k{color:#6b7280;font-weight:600}
.kv-v{font-variant-numeric: tabular-nums}
.pill{display:inline-block;padding:.15rem .5rem;border-radius:999px;background:#eef2ff;border:1px solid #c7d2fe;color:#3730a3;font-size:.85rem;margin:.15rem .25rem 0 0}
.muted{color:#9ca3af}
.riskbar{height:10px;border-radius:8px;background:#eee;overflow:hidden}
.riskbar>div{height:100%}
</style>
""", unsafe_allow_html=True)

def _fmt(v, unit=""):
    if v is None: return "—"
    if isinstance(v,(int,float)): return f"{v:.1f}{unit}"
    return str(v)

def pills(items: List[str]):
    if not items:
        st.markdown("<span class='muted'>None</span>", unsafe_allow_html=True); return
    html = " ".join(f"<span class='pill'>{str(x)}</span>" for x in items)
    st.markdown(html, unsafe_allow_html=True)

def keyval(label, value):
    st.markdown(f"""
    <div class="kv">
      <div class="kv-k">{label}</div>
      <div class="kv-v">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# Header with icon
hdr_l, hdr_r = st.columns([0.07, 0.93])
with hdr_l:
    st.image("icon.png", width=40)
with hdr_r:
    st.markdown("<h1 style='margin:0 0 .25rem 0'>LPG Customer Tank — Pre-Check</h1>", unsafe_allow_html=True)

# ─────────────────────────── Geo helpers ─────────────────────────────────────
def meters_per_degree(lat_deg: float) -> Tuple[float,float]:
    lat = math.radians(lat_deg)
    return (111132.92 - 559.82*math.cos(2*lat) + 1.175*math.cos(4*lat),
            111412.84*math.cos(lat) - 93.5*math.cos(3*lat))
def ll_to_xy(lat0, lon0, lat, lon): mlat,mlon=meters_per_degree(lat0); return (lon-lon0)*mlon,(lat-lat0)*mlat
def dist_line(lat0, lon0, line):
    if not line or len(line)<2: return None
    px,py=0.0,0.0
    verts=[ll_to_xy(lat0,lon0,la,lo) for la,lo in line]
    best=None
    for (ax,ay),(bx,by) in zip(verts,verts[1:]):
        apx,apy=px-ax,py-ay; abx,aby=bx-ax,by-ay; ab2=abx*abx+aby*aby
        t=0.0 if ab2==0 else max(0.0,min(1.0),(apx*abx+apy*aby)/ab2))
        cx,cy=ax+t*abx,ay+t*aby
        d=math.hypot(px-cx,py-cy)
        best=d if best is None else min(best,d)
    return best
def dist_poly(lat0,lon0,poly): return dist_line(lat0,lon0,poly+poly[:1])
def dist_pts(lat0, lon0, pts):
    if not pts: return None
    mlat,mlon=meters_per_degree(lat0)
    return min(math.hypot((lo-lon0)*mlon,(la-lat0)*mlat) for la,lo in pts)
def _dist_m(lat0, lon0, lat1, lon1):
    mlat,mlon=meters_per_degree(lat0)
    return math.hypot((lon1-lon0)*mlon,(lat1-lat0)*mlat)

# ─────────────────────────── External APIs ───────────────────────────────────
def w3w_to_latlon(words:str)->Tuple[Optional[float],Optional[float]]:
    if not W3W_API_KEY: return None,None
    try:
        r=requests.get("https://api.what3words.com/v3/convert-to-coordinates",
                       params={"words":words,"key":W3W_API_KEY},timeout=15)
        if r.status_code==200:
            c=r.json().get("coordinates",{})
            return c.get("lat"), c.get("lng")
    except: pass
    return None,None

def reverse_geocode(lat,lon)->Dict:
    try:
        r=requests.get("https://nominatim.openstreetmap.org/reverse",
                       params={"lat":lat,"lon":lon,"format":"jsonv2"},
                       headers={"User-Agent":"LPG-Precheck"},timeout=15)
        if r.status_code==200:
            j=r.json(); a=j.get("address") or {}
            return {"display_name":j.get("display_name"),
                    "road":a.get("road"),"postcode":a.get("postcode"),
                    "city":a.get("town") or a.get("city") or a.get("village"),
                    "county":a.get("county"),"state_district":a.get("state_district"),
                    "local_authority": a.get("municipality") or a.get("county") or a.get("state_district")}
    except: pass
    return {}

OVERPASS="https://overpass-api.de/api/interpreter"; UA={"User-Agent":"LPG-Precheck-Streamlit/merge"}
def overpass(lat,lon,r)->Dict:
    q=f"""
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
        r=requests.post(OVERPASS,data={"data":q},headers=UA,timeout=90)
        r.raise_for_status(); return r.json()
    except: return {"elements":[]}

def parse_osm(lat0,lon0,data)->Dict:
    bpolys, roads, drains, manholes, plines, pnodes, rails, wlines, wpolys, land_polys = [],[],[],[],[],[],[],[],[],[]
    rest_ways, surf_ways = [],[]
    for el in data.get("elements",[]):
        t=el.get("type"); tags=el.get("tags",{}) or {}; geom=el.get("geometry")
        coords=[(g["lat"],g["lon"]) for g in (geom or [])]
        if tags.get("building") and t in ("way","relation"): bpolys.append(coords)
        elif tags.get("highway") and t=="way":
            roads.append(coords)
            if any(k in tags for k in ("maxheight","maxwidth","maxweight","hgv","access","oneway")):
                rest_ways.append({"tags":tags,"coords":coords})
            if "surface" in tags or "smoothness" in tags:
                surf_ways.append({"tags":tags,"coords":coords})
        elif t=="way" and (tags.get("waterway")=="drain" or tags.get("tunnel")=="culvert"): drains.append(coords)
        elif t=="node" and (tags.get("man_made")=="manhole" or "manhole" in tags): manholes.append((el.get("lat"),el.get("lon")))
        elif t=="way" and tags.get("power")=="line": plines.append(coords)
        elif t=="node" and tags.get("power") in ("tower","pole"): pnodes.append((el.get("lat"),el.get("lon")))
        elif t=="way" and tags.get("railway") and tags.get("railway") not in ("abandoned","disused"): rails.append(coords)
        elif t=="way" and tags.get("waterway") in ("river","stream","ditch"): wlines.append(coords)
        elif t=="way" and tags.get("natural")=="water": wpolys.append(coords)
        elif t in ("way","relation") and tags.get("landuse"): land_polys.append({"tag":tags.get("landuse"),"coords":coords})

    d_build=min([dist_poly(lat0,lon0,p) for p in bpolys] or [None])
    d_road =min([dist_line(lat0,lon0,l) for l in roads] or [None])
    d_drain=min([dist_line(lat0,lon0,l) for l in drains]+([dist_pts(lat0,lon0,manholes)] if manholes else []) or [None])
    d_over =min([dist_line(lat0,lon0,l) for l in plines]+([dist_pts(lat0,lon0,pnodes)] if pnodes else []) or [None])
    d_rail =min([dist_line(lat0,lon0,l) for l in rails] or [None])
    d_water=min([dist_line(lat0,lon0,l) for l in wlines]+[dist_poly(lat0,lon0,p) for p in wpolys] or [None])

    # nearest landuse tag for 'open field' concept
    open_tags = {"meadow","grass","farmland","agriculture","pasture","greenfield","orchard","vineyard"}
    open_polys = [lp["coords"] for lp in land_polys if lp["tag"] in open_tags]
    d_open = min([dist_poly(lat0,lon0,p) for p in open_polys] or [None])

    land_counts={}
    for lp in land_polys: land_counts[lp["tag"]] = land_counts.get(lp["tag"],0)+1
    if land_counts:
        top=max(land_counts,key=lambda k:land_counts[k])
        land_class = "Domestic/Urban" if top in ("residential","commercial","retail") else ("Industrial" if top in ("industrial","industrial;retail") else "Rural/Agricultural")
    else:
        land_class = "Domestic/Urban" if len(bpolys)>80 else ("Rural/Agricultural" if len(bpolys)<20 else "Mixed")

    return {
        "d_building_m": round(d_build,1) if d_build is not None else None,
        "d_boundary_m": None,
        "d_road_m":     round(d_road,1)  if d_road  is not None else None,
        "d_drain_m":    round(d_drain,1) if d_drain is not None else None,
        "d_overhead_m": round(d_over,1)  if d_over  is not None else None,
        "d_rail_m":     round(d_rail,1)  if d_rail  is not None else None,
        "d_water_m":    round(d_water,1) if d_water is not None else None,
        "d_open_field_m": round(d_open,1) if d_open is not None else None,  # NEW
        "land_class": land_class,
        "counts": {"buildings":len(bpolys),"roads":len(roads),"drains":len(drains),"manholes":len(manholes),
                   "power_lines":len(plines),"power_structs":len(pnodes),"rail_lines":len(rails),
                   "water_lines":len(wlines),"water_polys":len(wpolys)},
        "restrictions": rest_ways, "surfaces": surf_ways, "nearest_road_line": roads[0] if roads else None
    }

def open_meteo(lat,lon)->Dict:
    try:
        r=requests.get("https://api.open-meteo.com/v1/forecast",
                       params={"latitude":lat,"longitude":lon,"current":"windspeed_10m,winddirection_10m"},timeout=12)
        cur=r.json().get("current",{}) if r.status_code==200 else {}
        spd,deg=cur.get("windspeed_10m"),cur.get("winddirection_10m")
        comp=["N","NE","E","SE","S","SW","W","NW"][round((deg or 0)%360/45)%8] if deg is not None else None
        return {"speed_mps":spd,"deg":deg,"compass":comp}
    except: return {"speed_mps":None,"deg":None,"compass":None}

def open_elevations(points):
    try:
        locs="|".join(f"{la},{lo}" for la,lo in points)
        r=requests.get("https://api.open-elevation.com/api/v1/lookup", params={"locations":locs},timeout=15)
        if r.status_code==200:
            return [it.get("elevation") for it in r.json().get("results",[])]
    except: pass
    return [None]*len(points)

def slope_calc(lat,lon,dx=20.0)->Dict:
    z=open_elevations([(lat,lon),
        (lat+dx/meters_per_degree(lat)[0],lon),
        (lat,lon+dx/meters_per_degree(lat)[1]),
        (lat-dx/meters_per_degree(lat)[0],lon),
        (lat,lon-dx/meters_per_degree(lat)[1])])
    if any(v is None for v in z): return {"elev_m":z[0] if z else None,"grade_pct":None,"aspect_deg":None}
    zc,zn,ze,zs,zw=z; dz_dy=(zn-zs)/(2*dx); dz_dx=(ze-zw)/(2*dx)
    grade=math.hypot(dz_dx,dz_dy)*100.0; aspect=(math.degrees(math.atan2(dz_dx,dz_dy))+360)%360
    return {"elev_m":zc,"grade_pct":round(grade,1),"aspect_deg":round(aspect,0)}

def approach_grade(lat,lon,road_line,N=6)->Dict:
    if not road_line: return {"avg_pct":None,"max_pct":None}
    mlat,mlon=meters_per_degree(lat)
    best,pt=None,None
    for la,lo in road_line:
        d=math.hypot((lo-lon)*mlon,(la-lat)*mlat)
        if best is None or d<best: best,pt=d,(la,lo)
    if pt is None: return {"avg_pct":None,"max_pct":None}
    pts=[(lat+(pt[0]-lat)*i/N, lon+(pt[1]-lon)*i/N) for i in range(N+1)]
    z=open_elevations(pts)
    if any(v is None for v in z): return {"avg_pct":None,"max_pct":None}
    grades=[]
    for i in range(N):
        run=math.hypot((pts[i+1][1]-pts[i][1])*mlon,(pts[i+1][0]-pts[i][0])*mlat)
        rise=z[i+1]-z[i]; grades.append(abs(rise/max(run,1e-3))*100.0)
    return {"avg_pct":round(sum(grades)/len(grades),1), "max_pct":round(max(grades),1)}

def osrm_ratio(lat,lon)->Optional[float]:
    try:
        r1=requests.get(f"https://router.project-osrm.org/nearest/v1/driving/{lon},{lat}",timeout=12)
        if r1.status_code!=200: return None
        snap_lon,snap_lat = r1.json()["waypoints"][0]["location"]
        r2=requests.get(f"https://router.project-osrm.org/route/v1/driving/{snap_lon},{snap_lat};{lon},{lat}",
                        params={"overview":"false"},timeout=15)
        if r2.status_code!=200: return None
        dist=float(r2.json()["routes"][0]["distance"])
        crow=math.hypot(lat-snap_lat,lon-snap_lon)*111000.0
        if crow<50 or dist<10: return None
        return dist/crow
    except: return None

def get_nearest_hospital_osm(lat: float, lon: float) -> dict:
    base = "https://overpass-api.de/api/interpreter"; headers = {"User-Agent": "LPG-Precheck-Streamlit/Hospital"}
    def query(r, filt):
        q=f"""[out:json][timeout:60];({filt});out tags center;"""
        try:
            resp=requests.post(base,data={"data":q.format(r=r,lat=lat,lon=lon)},headers=headers,timeout=60)
            resp.raise_for_status(); return resp.json().get("elements",[])
        except Exception: return []
    radii=[2000,5000,10000,20000,50000]
    best=None
    for r in radii:
        filt=f'node(around:{r},{lat},{lon})["amenity"="hospital"]["emergency"~"yes|designated|24_7|24/7"];way(around:{r},{lat},{lon})["amenity"="hospital"]["emergency"~"yes|designated|24_7|24/7"];relation(around:{r},{lat},{lon})["amenity"="hospital"]["emergency"~"yes|designated|24_7|24/7"]'
        for el in query(r,filt):
            tags=el.get("tags",{}) or {}
            la,lo=(el.get("lat"),el.get("lon")) if el.get("type")=="node" else ((el.get("center") or {}).get("lat"), (el.get("center") or {}).get("lon"))
            if la is None or lo is None: continue
            d=_dist_m(lat,lon,la,lo)
            cand={"name":tags.get("name") or "Unnamed hospital","distance_m":d,"lat":la,"lon":lo,"phone":tags.get("phone") or tags.get("contact:phone")}
            if (best is None) or (d<best["distance_m"]): best=cand
        if best: return best
    return {}

# ─────────────────────────── Helpers ─────────────────────────────────────────
def parse_num(s):
    if not s: return None
    s=str(s).lower().strip()
    for u in ("m","meter","metre","meters","metres","t","ton","tonne","tonnes"):
        if s.endswith(u): s=s[:-len(u)].strip()
    try: return float(s.replace(",", "."))
    except: return None

def humanize_restriction(s: str) -> str:
    if not s: return s
    t = s.strip().lower()
    m = re.search(r"maxheight\s*([\d.,]+)", t)
    if m: return f"Max height {m.group(1).replace(',', '.')} m"
    m = re.search(r"maxwidth\s*([\d.,]+)", t)
    if m: return f"Max width {m.group(1).replace(',', '.')} m"
    m = re.search(r"maxweight\s*([\d.,]+)", t)
    if m: return f"Max weight {m.group(1).replace(',', '.')} t"
    mapping = {
        "hgv=no": "No HGVs",
        "hgv=destination": "HGVs — destination only",
        "access=no": "No public access",
        "access=private": "Private access road",
        "oneway": "One-way",
    }
    return mapping.get(t, t.replace("=", " = "))

def restriction_notes(ways)->List[str]:
    TANKER = {"max_height_m": 3.6, "max_width_m": 2.55, "gross_weight_t": 18.0}
    out=[]
    for w in ways:
        t=w.get("tags",{})
        h=parse_num(t.get("maxheight")); wdt=parse_num(t.get("maxwidth")); wt=parse_num(t.get("maxweight"))
        if h is not None and h<TANKER["max_height_m"]: out.append(f"maxheight {h} m")
        if wdt is not None and wdt<TANKER["max_width_m"]: out.append(f"maxwidth {wdt} m")
        if wt is not None and wt<TANKER["gross_weight_t"]: out.append(f"maxweight {wt} t")
        if (t.get("hgv") or "").lower() in ("no","destination"): out.append(f"hgv={t.get('hgv').lower()}")
        if (t.get("access") or "").lower() in ("no","private"): out.append(f"access={t.get('access').lower()}")
        if (t.get("oneway") or "").lower()=="yes": out.append("oneway")
    seen=set(); out2=[]
    for s in out:
        if s not in seen: seen.add(s); out2.append(s)
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
    return {"risky_count":risky,"samples":samples[:8]}

def flood_risk(feats, slope, elev)->Dict:
    d=feats.get("d_water_m"); g=slope.get("grade_pct") or 0.0; z=elev or 0.0
    level="Low"; why=[]
    if d is None: why.append("No mapped watercourse nearby")
    else:
        if d<50: level="High"; why.append(f"Watercourse within {d} m")
        elif d<150: level="Medium"; why.append(f"Watercourse at {d} m")
        else: why.append(f"Watercourse at {d} m")
    if g>=6: why.append(f"Steep local slope {g}% (runoff/flow)")
    if z and z<10: why.append(f"Low elevation {int(z)} m a.s.l.")
    return {"level":level, "why":why}

# ─────────────────────────── Risk & AI ───────────────────────────────────────
def risk_score(feats, wind, slope, appr, rr, notes, surf, flood, answers=None, cfg=None)->Dict:
    answers = answers or {}
    cfg = cfg or CFG
    CoP = cfg["cop"]; W = cfg["weights"]; B = cfg["bands"]
    score=0.0; why=[]
    def add(x,msg): nonlocal score; score+=float(x); why.append((float(x),msg))
    def _safe_dist(x):
        if x is None: return None
        try: x=float(x)
        except: return None
        return x if x>0 else None
    def penal(dist, lim, w, label=None):
        if dist is None or lim is None: return
        try: lim=float(lim)
        except: return
        if dist>=lim: return
        if isinstance(w,(int,float)):
            add(w, f"{label or 'Below limit'} {lim} m (≈ {dist:.1f} m)"); return
        base, per_m, cap = w.get("base",0), w.get("per_m",0), w.get("cap",100)
        pts=min(cap, base + per_m*(lim-dist)); add(pts, f"{label or 'Below limit'} {lim} m (≈ {dist:.1f} m)")

    bld  = _safe_dist(feats.get("d_building_m"))
    shed = _safe_dist(feats.get("d_outbuilding_m"))
    eff_build = bld if shed is None else (shed if bld is None else min(bld,shed))
    penal(eff_build, CoP.get("to_building_m"), W.get("building"), "Below 3.0 m (≈")
    penal(_safe_dist(feats.get("d_boundary_m")), CoP.get("to_boundary_m"), W.get("boundary"), "Boundary <")
    penal(_safe_dist(feats.get("d_road_m")),  CoP.get("to_ignition_m"), W.get("ignition"), "Ignition proxy <")
    penal(_safe_dist(feats.get("d_drain_m")), CoP.get("to_drain_m"),    W.get("drain"), "Drain/manhole <")

    ov=_safe_dist(feats.get("d_overhead_m"))
    if ov is not None and ov < CoP["overhead_block_m"]:
        add(W.get("overhead_block",28), "Overhead power lines in no-go band")
    elif ov is not None and ov < CoP["overhead_info_m"]:
        add(W.get("overhead_info",10), "Overhead power lines in info band")

    rail=_safe_dist(feats.get("d_rail_m"))
    if rail is not None and rail < CoP["rail_attention_m"]:
        add(W.get("rail_near",10), f"Railway within {CoP['rail_attention_m']} m")

    wat=_safe_dist(feats.get("d_water_m"))
    if wat is not None and wat < 50: add(W.get("water_near",8), "Watercourse within 50 m")

    # NEW: open field proximity (mild)
    open_d = _safe_dist(feats.get("d_open_field_m"))
    if open_d is not None:
        if open_d < 10:  add(6, "Open field/farmland within 10 m")
        elif open_d < 30: add(3, "Open field/farmland within 30 m")

    spd=wind.get("speed_mps")
    if spd is not None and spd<CoP["wind_stagnant_mps"]: add(W.get("low_wind",6), f"Low wind {spd:.1f} m/s")

    g=slope.get("grade_pct")
    if g is not None and g>=6: add(W.get("slope_ge6",12), f"Local slope {g:.1f}%")
    elif g is not None and g>=CoP["slope_attention_pct"]: add(W.get("slope_ge3",8), f"Local slope {g:.1f}%")

    if appr.get("max_pct") is not None and appr["max_pct"]>=CoP["approach_grade_warn_pct"]:
        add(W.get("approach_steep",12), f"Steep approach (max {appr['max_pct']}%)")

    if rr is not None and rr>CoP["route_vs_crowfly_ratio_warn"]:
        add(W.get("route_detour",10), f"Route ≫ crow-fly ({rr:.2f}×)")

    if notes:
        nice = [humanize_restriction(n) for n in notes]
        add(min(12, 4*len(notes)),"Access restrictions: "+", ".join(nice))

    if surf.get("risky_count",0)>0:
        add(min(10, W.get("surface_flag_per",2)*surf['risky_count']), f"Surface flags={surf['risky_count']}")

    if answers.get("onsite_ignitions") is True: add(W.get("excel_ignitions_3m",6), "On-site ignition within 3 m (user)")
    stype=(answers.get("surface_type") or "").lower()
    if stype in ("gravel","grass"): add(min(10, W.get("surface_flag_per",2)*2), f"Soft surface: {stype}")
    fs = int(answers.get("fence_sides",0) or 0)
    if fs>0: add(min(12, W.get("fence_side",4)*fs), f"Enclosure effect: {fs} solid side(s)")
    if answers.get("los_restricted") is True: add(W.get("los_restricted",8), "Restricted line-of-sight at stand")

    # Vegetation slider: up to +8
    pct = int(answers.get("veg_within3m_pct", 0) or 0)
    if pct > 0:
        add(min(8, round(8 * pct / 100.0, 1)), f"Vegetation within 3 m ({pct}%)")

    if (answers.get("vegetation_notes") or "").strip(): add(W.get("vegetation",6), "Vegetation/combustible build-up")

    score=round(min(100.0,max(0.0,score)),1)
    status="PASS" if score<B["pass_lt"] else ("ATTENTION" if score<B["attention_lt"] else "BLOCKER")
    why.sort(key=lambda x:-x[0])
    return {"score":score,"status":status,"explain":why}

def ai_sections(context: Dict) -> Dict[str,str]:
    def offline(ctx: Dict) -> Dict[str,str]:
        feats,wind,slope,appr,rr,flood,risk = (ctx.get(k,{}) for k in
            ["features","wind","slope","approach","route_ratio","flood","risk"])
        S1 = (
            f"The local slope is {slope.get('grade_pct','?')}% (aspect {int((slope.get('aspect_deg') or 0))}°). "
            f"Key separations (m): building {feats.get('d_building_m')}, shed/structure {feats.get('d_outbuilding_m')}, "
            f"boundary {feats.get('d_boundary_m')}, road {feats.get('d_road_m')}, drain {feats.get('d_drain_m')}, "
            f"overhead power lines {feats.get('d_overhead_m')}, rail {feats.get('d_rail_m')}, open field {feats.get('d_open_field_m')}. "
            f"Wind {(wind.get('speed_mps') or 0):.1f} m/s from {wind.get('compass') or 'n/a'}. "
            f"Heuristic {risk.get('score','?')}/100 → {risk.get('status','?')}. Drivers: "
            + "; ".join([f"{int(p)} {m}" for p, m in (risk.get('explain') or [])[:4]]) + "."
        )
        S2 = (
            f"Flood {flood.get('level','n/a')} ({'; '.join(flood.get('why',[]))}). "
            f"Protect drains during transfers; land use {feats.get('land_class','n/a')}."
        )
        S3 = (
            f"Approach avg/max {appr.get('avg_pct','?')}/{appr.get('max_pct','?')}%. "
            + (f"Route indirectness {rr:.2f}× crow-fly. " if rr else "")
            + "Validate signage/restrictions; provide sound hardstanding and clear sightlines."
        )
        S4 = ("Site appears suitable with routine controls." if risk.get('status')=='PASS'
              else "Attention required: confirm separations, control ignition, protect drainage, plan safe approach/egress.")
        return {
            "Safety Risk Profile": S1, "Environmental Considerations": S2,
            "Access & Logistics": S3, "Overall Site Suitability": S4,
        }
    if not OPENAI_API_KEY:
        return offline(context)
    try:
        prompt = f"""
You are an expert LPG site assessor. Using the context JSON, write professional narrative commentary.
Use four sections exactly:
[1] Safety Risk Profile
[2] Environmental Considerations
[3] Access & Logistics
[4] Overall Site Suitability
2–4 sentences per section; numeric where relevant; include practical actions.

Context JSON:
{json.dumps(context, ensure_ascii=False)}
""".strip()
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type":"application/json"},
            json={"model":"gpt-4o-mini","temperature":0.25,"max_tokens":1100,
                  "messages":[{"role":"system","content":"You are an LPG safety and logistics assessor."},
                              {"role":"user","content":prompt}]},
            timeout=45
        )
        if r.status_code != 200:
            return offline(context)
        text = r.json()["choices"][0]["message"]["content"].strip()
        sections = {"Safety Risk Profile":"","Environmental Considerations":"","Access & Logistics":"","Overall Site Suitability":""}
        current=None; mapping={"[1]":"Safety Risk Profile","[2]":"Environmental Considerations","[3]":"Access & Logistics","[4]":"Overall Site Suitability"}
        for line in text.splitlines():
            t=line.strip()
            for key,name in mapping.items():
                if t.startswith(key):
                    current=name; t=t[len(key):].strip(":- \t")
                    if t: sections[current]+=t+"\n"; break
            else:
                if current: sections[current]+=t+"\n"
        clean={}; fb=offline(context)
        for k,v in sections.items():
            lines=[ln.rstrip() for ln in (v or "").splitlines()]
            while lines and not lines[0].strip(): lines.pop(0)
            clean[k]="\n".join(lines).strip() or fb[k]
        return clean
    except Exception:
        return offline(context)

def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    return obj
def _eval_when(expr: str, ctx: dict) -> bool:
    try:
        env = {
            "answers":  _to_ns(ctx.get("answers", {})),
            "features": _to_ns(ctx.get("features", {})),
            "cop":      _to_ns(CFG["cop"]),
            "wind":     _to_ns(ctx.get("wind", {})),
            "slope":    _to_ns(ctx.get("slope", {})),
            "approach": _to_ns(ctx.get("approach", {})),
            "risk":     _to_ns(ctx.get("risk", {})),
        }
        return bool(eval(expr, {"__builtins__": {}}, env))
    except Exception:
        return False

# ─────────────────────────── Map helpers ─────────────────────────────────────
def fetch_map(lat, lon, zoom=17, size=(900, 600)):
    if not MAPBOX_TOKEN: return None
    try:
        w,h=size
        marker=f"pin-l+f30({lon},{lat})"; style="light-v11"
        url=(f"https://api.mapbox.com/styles/v1/mapbox/{style}/static/"
             f"{marker}/{lon},{lat},{zoom},0/{w}x{h}?access_token={MAPBOX_TOKEN}")
        r=requests.get(url,timeout=15); r.raise_for_status()
        return r.content
    except Exception:
        return None
def overlay_rings(img_bytes, lat, zoom=17):
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return img_bytes
    try:
        img=Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        def mpp(lat,zoom): return 156543.03392*math.cos(math.radians(lat))/(2**zoom)
        scale=mpp(lat,zoom); cx,cy=img.width//2,img.height//2; d=ImageDraw.Draw(img,"RGBA")
        for r,col in ((3,(220,0,0,180)),(6,(255,140,0,160))):
            px=max(1,int(r/scale)); d.ellipse((cx-px,cy-px,cx+px,cy+px),outline=col,width=4)
        out=io.BytesIO(); img.save(out, format="PNG"); return out.getvalue()
    except Exception:
        return img_bytes

# ─────────────────────────── PDF builder ─────────────────────────────────────
def build_pdf_bytes(ctx: dict) -> bytes:
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfbase import pdfmetrics
    except Exception:
        return b""

    feats=ctx["features"]; wind=ctx["wind"]; slope=ctx["slope"]; appr=ctx["approach"]
    rr=ctx["route_ratio"]; flood=ctx["flood"]; risk=ctx["risk"]
    addr=ctx["address"]; la_name=ctx.get("authority"); hospital=ctx.get("hospital") or {}
    ai=ctx.get("ai") or {}; controls=ctx.get("controls") or []
    answers=ctx.get("answers") or {}; vehicle=ctx.get("vehicle") or {}
    notes=ctx.get("restrictions
