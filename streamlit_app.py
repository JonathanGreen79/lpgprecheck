# app.py — Streamlit UI for LPG Pre-Check (2-step: fetch → edit → assess)
# pip install -r requirements.txt

import os, io, re, math, json, requests
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional
import streamlit as st

# ===================== Secrets (Streamlit → local fallback → env) =====================
def _load_secrets():
    W3W = getattr(st, "secrets", {}).get("W3W_API_KEY", None)
    MAPBOX = getattr(st, "secrets", {}).get("MAPBOX_TOKEN", None)
    OPENAI = getattr(st, "secrets", {}).get("OPENAI_API_KEY", None)
    if not any([W3W, MAPBOX, OPENAI]):
        try:
            from secrets_local import W3W_API_KEY as _W3W, MAPBOX_TOKEN as _MAP, OPENAI_API_KEY as _OA
            W3W, MAPBOX, OPENAI = _W3W, _MAP, _OA
        except Exception:
            import os
            W3W  = os.getenv("W3W_API_KEY", "")
            MAPBOX = os.getenv("MAPBOX_TOKEN", "")
            OPENAI = os.getenv("OPENAI_API_KEY", "")
    return W3W or "", MAPBOX or "", OPENAI or ""

W3W_API_KEY, MAPBOX_TOKEN, OPENAI_API_KEY = _load_secrets()

# ===================== Page config & CSS =====================
st.set_page_config(page_title="LPG Pre-Check", page_icon="🛢️", layout="wide")
st.markdown("""
<style>
.smallcaps{font-variant:all-small-caps;letter-spacing:.04em;color:#6b7280}
.kv{display:flex;justify-content:space-between;gap:.75rem;padding:.5rem .75rem;border:1px solid #eee;border-radius:.5rem;margin:.25rem 0;background:#fafafa}
.kv-k{color:#6b7280;font-weight:600}
.kv-v{font-variant-numeric: tabular-nums}
.pill{display:inline-block;padding:.15rem .5rem;border-radius:999px;background:#eef2ff;border:1px solid #c7d2fe;color:#3730a3;font-size:.85rem;margin:.15rem .25rem 0 0}
.muted{color:#9ca3af}
hr{border:none;border-top:1px solid #eee;margin:1rem 0}
</style>
""", unsafe_allow_html=True)

# ===================== Defaults: CoP, weights, controls, questions =====================
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
        "excel_drain_3m": 8,
    },
    "controls": {
        "drain_within_3m": {
            "when": "answers.drain_within_3m is True",
            "actions": [
                "Fit drain cover/insert or isolate drain during transfers.",
                "Install/confirm spill containment to prevent entry to drainage."
            ]
        },
        "onsite_ignitions": {
            "when": "answers.onsite_ignitions is True",
            "actions": [
                "Increase separation or relocate ignition source beyond 3 m.",
                "Mark ATEX zone and enforce no-smoking/no-ignition controls."
            ]
        },
        "surface_soft": {
            "when": "answers.surface_type in ['gravel','grass']",
            "actions": [
                "Upgrade stand area to hardstanding (concrete/asphalt) for stability and spill control."
            ]
        }
    }
}

# ===================== Helpers: numbers, table, pills =====================
def _fmt(v, unit=""):
    if v is None: return "—"
    if isinstance(v, (int,float)):
        return f"{v:.1f}{unit}"
    return str(v)

def pills(items):
    if not items:
        st.markdown("<span class='muted'>None</span>", unsafe_allow_html=True)
        return
    html = " ".join(f"<span class='pill'>{str(x)}</span>" for x in items)
    st.markdown(html, unsafe_allow_html=True)

def keyval(label, value):
    st.markdown(f"""
    <div class="kv">
      <div class="kv-k">{label}</div>
      <div class="kv-v">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# ===================== Geo + API functions =====================
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
        t=0.0 if ab2==0 else max(0.0,min(1.0,(apx*abx+apy*aby)/ab2))
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

def w3w_to_latlon(words:str)->Tuple[Optional[float],Optional[float]]:
    if not W3W_API_KEY: return None, None
    try:
        r=requests.get("https://api.what3words.com/v3/convert-to-coordinates",
                       params={"words":words,"key":W3W_API_KEY},timeout=15)
        if r.status_code==200:
            c=r.json().get("coordinates",{})
            return c.get("lat"), c.get("lng")
    except Exception: pass
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
                    "county":a.get("county"),
                    "state_district":a.get("state_district"),
                    "local_authority": a.get("municipality") or a.get("county") or a.get("state_district")}
    except Exception: pass
    return {}

OVERPASS="https://overpass-api.de/api/interpreter"; UA={"User-Agent":"LPG-Precheck-Streamlit/2step"}
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
    except Exception: return {"elements":[]}

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
    land_counts={}; 
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
    except Exception: return {"speed_mps":None,"deg":None,"compass":None}

def open_elevations(points):
    try:
        locs="|".join(f"{la},{lo}" for la,lo in points)
        r=requests.get("https://api.open-elevation.com/api/v1/lookup", params={"locations":locs},timeout=15)
        if r.status_code==200:
            return [it.get("elevation") for it in r.json().get("results",[])]
    except Exception: pass
    return [None]*len(points)

def slope_aspect(lat,lon,dx=20.0)->Dict:
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
        # note: avg grade is avg of segment grades
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
    except Exception: return None

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

# ===================== Risk model & controls =====================
def parse_num(s):
    if not s: return None
    s=str(s).lower().strip()
    for u in ("m","meter","metre","meters","metres","t","ton","tonne","tonnes"):
        if s.endswith(u): s=s[:-len(u)].strip()
    try: return float(s.replace(",","."))  # type: ignore
    except: return None

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

def risk_score(feats, wind, slope, appr, rr, notes, surf, flood, answers=None, cfg=None)->Dict:
    answers = answers or {}
    cfg = cfg or CFG
    CoP = cfg["cop"]; W = cfg["weights"]; B = cfg["bands"]

    score=0.0; why=[]
    def add(x,msg):
        nonlocal score
        score+=float(x); why.append((float(x),msg))
    def penal(dist, lim, w):
        if dist is None or lim is None: return
        try: lim=float(lim)
        except: return
        if dist>=lim: return
        if isinstance(w,(int,float)):
            add(w, f"Below {lim} m (≈ {dist:.1f} m)"); return
        base, per_m, cap = w.get("base",0), w.get("per_m",0), w.get("cap",100)
        pts=min(cap, base + per_m*(lim-dist))
        add(pts, f"Below {lim} m (≈ {dist:.1f} m)")

    penal(feats.get("d_building_m"), CoP.get("to_building_m"), W.get("building"))
    penal(feats.get("d_road_m"),     CoP.get("to_ignition_m"), W.get("ignition"))
    penal(feats.get("d_drain_m"),    CoP.get("to_drain_m"),    W.get("drain"))
    d_ov=feats.get("d_overhead_m")
    if d_ov is not None and d_ov<CoP["overhead_block_m"]: add(W.get("overhead_block",28), "Overhead in no-go band")
    elif d_ov is not None and d_ov<CoP["overhead_info_m"]:  add(W.get("overhead_info",10),  "Overhead in info band")
    d_rail=feats.get("d_rail_m")
    if d_rail is not None and d_rail<CoP["rail_attention_m"]: add(W.get("rail_near",10), f"Railway within {CoP['rail_attention_m']} m")
    if feats.get("d_water_m") is not None and feats["d_water_m"]<50: add(W.get("water_near",8), "Watercourse within 50 m")
    spd=wind.get("speed_mps")
    if spd is not None and spd<CoP["wind_stagnant_mps"]: add(W.get("low_wind",6), f"Low wind {spd:.1f} m/s")
    g=slope.get("grade_pct")
    if g is not None and g>=6: add(W.get("slope_ge6",12), f"Local slope {g:.1f}%")
    elif g is not None and g>=CoP["slope_attention_pct"]: add(W.get("slope_ge3",8), f"Local slope {g:.1f}%")
    if appr.get("max_pct") is not None and appr["max_pct"]>=CoP["approach_grade_warn_pct"]:
        add(W.get("approach_steep",12), f"Steep approach (max {appr['max_pct']}%)")
    if rr is not None and rr>CoP["route_vs_crowfly_ratio_warn"]:
        add(W.get("route_detour",10), f"Route ≫ crow-fly ({rr:.2f}×)")
    if notes: add(min(12, 4*len(notes)),"Access restrictions: "+", ".join(notes))
    if surf.get("risky_count",0)>0: add(min(10, W.get("surface_flag_per",2)*surf['risky_count']), f"Surface flags={surf['risky_count']}")
    if answers.get("onsite_ignitions") is True: add(W.get("excel_ignitions_3m",6), "On-site ignition within 3 m (user)")
    if answers.get("drain_within_3m") is True: add(W.get("excel_drain_3m",8), "Drain/manhole within 3 m (user)")
    stype=(answers.get("surface_type") or "").lower()
    if stype in ("gravel","grass"): add(min(10, W.get("surface_flag_per",2)*2), f"Soft surface: {stype}")

    score=round(min(100.0,max(0.0,score)),1)
    status="PASS" if score<B["pass_lt"] else ("ATTENTION" if score<B["attention_lt"] else "BLOCKER")
    why.sort(key=lambda x:-x[0])
    return {"score":score,"status":status,"explain":why}

# ===================== AI sections =====================
def ai_sections(context: Dict) -> Dict[str,str]:
    def offline(ctx: Dict) -> Dict[str,str]:
        feats,wind,slope,appr,rr,flood,risk = (ctx.get(k,{}) for k in
            ["features","wind","slope","approach","route_ratio","flood","risk"])
        S1=(f"Local slope {slope.get('grade_pct','?')}% (aspect {int((slope.get('aspect_deg') or 0))}°). "
            f"Separations (m): bldg {feats.get('d_building_m')}, boundary {feats.get('d_boundary_m')}, "
            f"road {feats.get('d_road_m')}, drain {feats.get('d_drain_m')}, overhead {feats.get('d_overhead_m')}, rail {feats.get('d_rail_m')}. "
            f"Wind {(wind.get('speed_mps') or 0):.1f} m/s from {wind.get('compass') or 'n/a'}. "
            f"Heuristic {risk.get('score','?')}/100 → {risk.get('status','?')}. "
            f"Top factors: " + "; ".join(f"{int(p)} {m}" for p,m in risk.get("explain",[])[:5]) + ".")
        S2=(f"Flood {flood.get('level','n/a')} ({'; '.join(flood.get('why',[]))}). "
            f"Watercourse ~{feats.get('d_water_m','n/a')} m; drains/manholes {feats.get('d_drain_m','n/a')} m. Land use {feats.get('land_class','n/a')}.")
        S3=(f"Approach avg/max {appr.get('avg_pct','?')}/{appr.get('max_pct','?')}%. " +
            (f"Route {rr:.2f}× crow-fly. " if rr else "") +
            f"Validate restrictions and stand surface.")
        S4=("Suitable with routine controls" if risk.get('status')=='PASS' else
            "Attention required — ensure separation compliance, ignition control, drainage protection, and safe approach/egress.")
        return {"Safety Risk Profile":S1,"Environmental Considerations":S2,"Access & Logistics":S3,"Overall Site Suitability":S4}

    if not OPENAI_API_KEY:
        return offline(context)

    try:
        prompt = f"""
You are an LPG siting assessor. Produce FOUR sections labelled exactly:
[1] Safety Risk Profile
[2] Environmental Considerations
[3] Access & Logistics
[4] Overall Site Suitability
Use the context JSON (auto + edited values). Keep it practical and numeric.
Context:
{json.dumps(context, ensure_ascii=False)}
""".strip()
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type":"application/json"},
            json={"model":"gpt-4o-mini","temperature":0.25,"max_tokens":900,
                  "messages":[{"role":"system","content":"You are an LPG safety and logistics assessor."},
                              {"role":"user","content":prompt}]},
            timeout=45
        )
        if r.status_code != 200: return offline(context)
        text = r.json()["choices"][0]["message"]["content"].strip()
        sections = {"Safety Risk Profile":"","Environmental Considerations":"","Access & Logistics":"","Overall Site Suitability":""}
        current=None; mapping={"[1]":"Safety Risk Profile","[2]":"Environmental Considerations","[3]":"Access & Logistics","[4]":"Overall Site Suitability"}
        for line in text.splitlines():
            t=line.strip()
            for key,name in mapping.items():
                if t.startswith(key):
                    current=name; t=t[len(key):].strip(":- \t"); 
                    if t: sections[current]+=t+"\n"; break
            else:
                if current: sections[current]+=t+"\n"
        clean={}
        for heading, body in sections.items():
            lines=[ln.rstrip() for ln in (body or "").splitlines()]
            while lines and not lines[0].strip(): lines.pop(0)
            clean[heading]="\n".join(lines).strip()
        fb=offline(context)
        for k in clean:
            if not clean[k]: clean[k]=fb[k]
        return clean
    except Exception:
        return offline(context)

# ===================== Controls evaluator (SAFE) =====================
def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    return obj

def _eval_when(expr: str, ctx: dict) -> bool:
    try:
        env = {
            "answers":  _to_ns(ctx.get("answers", {})),
            "features": _to_ns(ctx.get("features", {})),
            "cop":      _to_ns(ctx.get("cop", {})),
            "wind":     _to_ns(ctx.get("wind", {})),
            "slope":    _to_ns(ctx.get("slope", {})),
            "approach": _to_ns(ctx.get("approach", {})),
            "risk":     _to_ns(ctx.get("risk", {})),
        }
        return bool(eval(expr, {"__builtins__": {}}, env))
    except Exception as e:
        st.caption(f"⚠ control rule error: {e}")
        return False

# ===================== Map (optional, uses Mapbox) =====================
def fetch_map(lat, lon, zoom=17, size=(900, 600)):
    if not MAPBOX_TOKEN:
        return None
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

# ===================== PDF (minimal) =====================
def build_pdf_bytes(ctx: dict) -> bytes:
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.pdfbase import pdfmetrics
    except Exception:
        return b""
    feats=ctx["features"]; wind=ctx["wind"]; slope=ctx["slope"]; appr=ctx["approach"]
    rr=ctx["route_ratio"]; flood=ctx["flood"]; risk=ctx["risk"]
    addr=ctx["address"]; la_name=ctx.get("authority"); hospital=ctx.get("hospital") or {}
    ai=ctx.get("ai") or {}; controls=ctx.get("controls") or []

    buff=io.BytesIO(); c=canvas.Canvas(buff, pagesize=A4)
    W,H=A4; M=38; y=H-46; LEAD=12; blue=colors.HexColor("#1f4e79")
    def ensure(h): nonlocal y; 
    def ensure(h):
        nonlocal y
        if y-h<40: c.showPage(); y=H-46
    def head(txt, size=16):
        nonlocal y; ensure(size+6); c.setFillColor(blue); c.setFont("Helvetica-Bold",size); c.drawString(M,y,txt); y-=size+6; c.setFillColor(colors.black)
    def line(txt, size=10):
        nonlocal y; ensure(size+3); c.setFont("Helvetica",size); c.drawString(M,y,txt); y-=size+3
    def wrap(txt, size=10):
        nonlocal y
        width=W-2*M
        c.setFont("Helvetica",size)
        for para in (txt or "").split("\n"):
            para=para.rstrip()
            if not para: ensure(LEAD); y-=LEAD; continue
            cur=""
            for w in para.split():
                test=(cur+" "+w).strip() if cur else w
                if pdfmetrics.stringWidth(test,"Helvetica",size)<=width:
                    cur=test
                else:
                    ensure(LEAD); c.drawString(M,y,cur); y-=LEAD; cur=w
            if cur: ensure(LEAD); c.drawString(M,y,cur); y-=LEAD

    head(f"LPG Pre-Check — ///{ctx['words']}")
    addr_line=", ".join([p for p in [addr.get('road'),addr.get('city'),addr.get('postcode')] if p])
    if addr_line: line(addr_line,9)
    if addr.get("display_name"): line(addr["display_name"],9)
    line(f"Local authority: {la_name or 'n/a'}",9)
    if hospital: line(f"Nearest A&E: {hospital.get('name','n/a')} ({(hospital.get('distance_m',0)/1000):.1f} km)",9)

    head("Key Metrics",12)
    line(f"Wind: {_fmt(wind.get('speed_mps'),' m/s')} from {wind.get('compass') or 'n/a'}")
    line(f"Slope: {_fmt(slope.get('grade_pct'),' %')}  |  Approach avg/max: {_fmt(appr.get('avg_pct'),' %')} / {_fmt(appr.get('max_pct'),' %')}")
    line(f"Route sanity: {'n/a' if rr is None else f'{rr:.2f}×'}  |  Flood: {flood['level']} — {'; '.join(flood['why'])}")

    head("Separations",12)
    def fmt(v): return f"{v:.1f} m" if isinstance(v,(int,float)) else "n/a"
    for lbl,key in [("Building","d_building_m"),("Boundary","d_boundary_m"),("Road/footpath","d_road_m"),
                    ("Drain/manhole","d_drain_m"),("Overhead","d_overhead_m"),("Railway","d_rail_m"),("Watercourse","d_water_m")]:
        line(f"{lbl}: {fmt(feats.get(key))}")
    line(f"Land use: {feats.get('land_class','n/a')}")

    head("Risk score",12)
    line(f"Total: {risk['score']}/100 → {risk['status']}")
    for pts,msg in risk["explain"][:10]: line(f"+{int(pts)} {msg}",9)

    if controls:
        head("Recommended controls",12)
        for a in controls: wrap(f"• {a}")

    for t in ["Safety Risk Profile","Environmental Considerations","Access & Logistics","Overall Site Suitability"]:
        head(t,12); wrap(ai.get(t,""))

    c.showPage(); c.save(); return buff.getvalue()

# ===================== UI: Step 1 → Step 2 flow =====================
st.title("🛢️ LPG Customer Tank — Pre-Check")

with st.sidebar:
    st.subheader("Step 1 — Location")
    words = st.text_input("what3words (word.word.word)", value=st.session_state.get("words",""), placeholder="filled.count.soap")
    fetch_btn = st.button("Fetch", type="primary", use_container_width=True)
    st.markdown("---")
    st.subheader("Secrets")
    st.caption(f"W3W: {'✅' if W3W_API_KEY else '❌'} • Mapbox: {'✅' if MAPBOX_TOKEN else '❌'} • OpenAI: {'✅' if OPENAI_API_KEY else '❌'}")

# Fetch auto data
if fetch_btn:
    st.session_state["words"] = words
    if words.count(".")!=2 or not all(p.isalpha() for p in words.split(".")):
        st.error("Invalid what3words format."); st.stop()
    with st.status("Fetching site data…", expanded=False):
        lat,lon = w3w_to_latlon(words)
        if lat is None: st.error("what3words lookup failed."); st.stop()
        addr     = reverse_geocode(lat,lon)
        la_name  = addr.get("local_authority") or addr.get("county") or addr.get("state_district")
        wind     = open_meteo(lat,lon)
        slope    = slope_aspect(lat,lon,dx=20.0)
        osm      = overpass(lat,lon,int(CFG["cop"]["poi_radius_m"]))
        feats    = parse_osm(lat,lon,osm)
        hospital = get_nearest_hospital_osm(lat,lon)
        if hospital: feats["d_hospital_m"] = round(hospital["distance_m"],1)
        appr     = approach_grade(lat,lon,feats.get("nearest_road_line"),N=6)
        rr       = osrm_ratio(lat,lon)
        notes    = restriction_notes(feats.get("restrictions",[]))
        surf     = surface_info(feats.get("surfaces",[]))
        flood    = flood_risk(feats, slope, slope.get("elev_m"))
        # prefill editable values
        st.session_state["auto"] = dict(lat=lat, lon=lon, addr=addr, la=la_name,
                                        wind=wind, slope=slope, feats=feats, hospital=hospital,
                                        appr=appr, rr=rr, notes=notes, surf=surf, flood=flood)
        st.session_state["form"] = {
            # separations (m)
            "d_building_m": feats.get("d_building_m"),
            "d_boundary_m": feats.get("d_boundary_m"),
            "d_road_m":     feats.get("d_road_m"),
            "d_drain_m":    feats.get("d_drain_m"),
            "d_overhead_m": feats.get("d_overhead_m"),
            "d_rail_m":     feats.get("d_rail_m"),
            "d_water_m":    feats.get("d_water_m"),
            "land_class":   feats.get("land_class"),
            # environment
            "wind_speed_mps": wind.get("speed_mps"),
            "wind_deg":       wind.get("deg"),
            "slope_pct":      slope.get("grade_pct"),
            "approach_avg":   appr.get("avg_pct"),
            "approach_max":   appr.get("max_pct"),
            "route_ratio":    rr,
            # site answers
            "onsite_ignitions": False,
            "drain_within_3m":  False,
            "surface_type":     "(leave unset)"
        }
    st.success("Data fetched. Review & edit below, then press **Confirm & Assess**.")

# Step 2 form: show if we have auto data
auto = st.session_state.get("auto")
formvals = st.session_state.get("form")

if auto and formvals:
    st.markdown("### Step 2 — Edit values (everything is editable)")

    with st.form("edit_all"):
        # groups: Location summary
        lat,lon,addr = auto["lat"], auto["lon"], auto["addr"]
        c1,c2,c3 = st.columns([0.45,0.35,0.20])
        with c1:
            addr_line = ", ".join([p for p in [addr.get('road'),addr.get('city'),addr.get('postcode')] if p]) or "—"
            st.text_input("Address (OSM)", value=addr_line, disabled=True)
            st.text_input("Display name", value=addr.get("display_name") or "—", disabled=True)
        with c2:
            st.text_input("Local authority", value=(auto["la"] or "—"), disabled=True)
            st.text_input("Nearest A&E", value=(auto["hospital"] or {}).get("name","—"), disabled=True)
        with c3:
            st.text_input("Latitude", value=f"{lat:.6f}", disabled=True)
            st.text_input("Longitude", value=f"{lon:.6f}", disabled=True)

        st.markdown("<hr/>", unsafe_allow_html=True)

        # Separations
        st.subheader("Separations (~400 m)")
        sL, sR = st.columns(2)
        with sL:
            formvals["d_building_m"] = st.number_input("Building (m)", value=(formvals["d_building_m"] or 0.0), min_value=0.0, max_value=2000.0, step=0.1, format="%.1f")
            formvals["d_road_m"]     = st.number_input("Road/footpath (m)", value=(formvals["d_road_m"] or 0.0), min_value=0.0, max_value=2000.0, step=0.1, format="%.1f")
            formvals["d_overhead_m"] = st.number_input("Overhead (m)", value=(formvals["d_overhead_m"] or 0.0), min_value=0.0, max_value=2000.0, step=0.1, format="%.1f")
            formvals["d_water_m"]    = st.number_input("Watercourse (m)", value=(formvals["d_water_m"] or 0.0), min_value=0.0, max_value=5000.0, step=0.1, format="%.1f")
        with sR:
            formvals["d_boundary_m"] = st.number_input("Boundary (m)", value=(formvals["d_boundary_m"] or 0.0), min_value=0.0, max_value=2000.0, step=0.1, format="%.1f")
            formvals["d_drain_m"]    = st.number_input("Drain/manhole (m)", value=(formvals["d_drain_m"] or 0.0), min_value=0.0, max_value=2000.0, step=0.1, format="%.1f")
            formvals["d_rail_m"]     = st.number_input("Railway (m)", value=(formvals["d_rail_m"] or 0.0), min_value=0.0, max_value=2000.0, step=0.1, format="%.1f")
            formvals["land_class"]   = st.selectbox("Land use", ["Domestic/Urban","Industrial","Rural/Agricultural","Mixed"], index=["Domestic/Urban","Industrial","Rural/Agricultural","Mixed"].index(formvals["land_class"] or "Mixed"))

        st.markdown("<hr/>", unsafe_allow_html=True)

        # Environment
        st.subheader("Environment & approach")
        e1,e2,e3 = st.columns(3)
        with e1:
            formvals["wind_speed_mps"] = st.number_input("Wind speed (m/s)", value=(formvals["wind_speed_mps"] or 0.0), min_value=0.0, max_value=60.0, step=0.1, format="%.1f")
            formvals["slope_pct"]      = st.number_input("Local slope (%)", value=(formvals["slope_pct"] or 0.0), min_value=0.0, max_value=100.0, step=0.1, format="%.1f")
        with e2:
            formvals["wind_deg"]       = st.number_input("Wind direction (°)", value=(formvals["wind_deg"] or 0.0), min_value=0.0, max_value=359.0, step=1.0, format="%.0f")
            formvals["approach_avg"]   = st.number_input("Approach avg (%)", value=(formvals["approach_avg"] or 0.0), min_value=0.0, max_value=100.0, step=0.1, format="%.1f")
        with e3:
            formvals["approach_max"]   = st.number_input("Approach max (%)", value=(formvals["approach_max"] or 0.0), min_value=0.0, max_value=100.0, step=0.1, format="%.1f")
            rr_default = "—" if auto["rr"] is None else f"{auto['rr']:.2f}"
            rr_str = st.text_input("Route sanity (× crow-fly)", value=rr_default)
            try:
                formvals["route_ratio"] = float(rr_str) if rr_str not in ("", "—") else None
            except:
                formvals["route_ratio"] = None

        st.markdown("<hr/>", unsafe_allow_html=True)

        # Site answers (from Excel logic)
        st.subheader("Site answers")
        a1,a2,a3 = st.columns([0.33,0.33,0.34])
        with a1:
            formvals["onsite_ignitions"] = st.toggle("Ignition source within 3 m?", value=formvals["onsite_ignitions"])
        with a2:
            formvals["drain_within_3m"]  = st.toggle("Drain/manhole within 3 m?", value=formvals["drain_within_3m"])
        with a3:
            formvals["surface_type"]     = st.selectbox("Surface on tanker stand", ["(leave unset)","asphalt","concrete","gravel","grass","other"], index=["(leave unset)","asphalt","concrete","gravel","grass","other"].index(formvals["surface_type"]))

        confirmed = st.form_submit_button("Confirm & Assess", type="primary")

    if confirmed:
        # build merged dicts
        feats = auto["feats"].copy()
        for k in ["d_building_m","d_boundary_m","d_road_m","d_drain_m","d_overhead_m","d_rail_m","d_water_m","land_class"]:
            feats[k] = formvals[k]
        wind = {"speed_mps": formvals["wind_speed_mps"], "deg": formvals["wind_deg"],
                "compass": ["N","NE","E","SE","S","SW","W","NW"][round((formvals["wind_deg"] or 0)%360/45)%8] if formvals["wind_deg"] is not None else None}
        slope = {"grade_pct": formvals["slope_pct"], "aspect_deg": auto["slope"].get("aspect_deg"), "elev_m": auto["slope"].get("elev_m")}
        appr  = {"avg_pct": formvals["approach_avg"], "max_pct": formvals["approach_max"]}
        rr    = formvals["route_ratio"]
        answers = {
            "onsite_ignitions": formvals["onsite_ignitions"],
            "drain_within_3m":  formvals["drain_within_3m"],
            "surface_type":     formvals["surface_type"] if formvals["surface_type"]!="(leave unset)" else None
        }
        notes = auto["notes"]; surf = auto["surf"]; flood=auto["flood"]
        risk = risk_score(feats, wind, slope, appr, rr, notes, surf, flood, answers=answers, cfg=CFG)

        ctx = {
            "cop": CFG["cop"], "words": st.session_state["words"], "lat": auto["lat"], "lon": auto["lon"],
            "address": auto["addr"], "authority": auto["la"], "hospital": auto["hospital"],
            "wind": wind, "slope": slope, "features": feats, "approach": appr,
            "route_ratio": rr, "restrictions": notes, "surfaces": surf, "flood": flood,
            "answers": answers, "risk": risk
        }

        # controls
        controls=[]
        for key, rule in CFG["controls"].items():
            if _eval_when(rule.get("when","False"), ctx):
                controls += rule.get("actions",[])
        controls = list(dict.fromkeys(controls))  # dedupe
        ctx["controls"]=controls

        # AI commentary
        ai = ai_sections(ctx); ctx["ai"]=ai

        # ========== OUTPUT (two columns) ==========
        colL, colR = st.columns([0.55, 0.45], gap="large")

        with colL:
            st.subheader("Key metrics")
            r1, r2, r3 = st.columns(3)
            with r1: keyval("Wind (m/s)", _fmt(wind.get("speed_mps")))
            with r2: keyval("Wind dir", f"{_fmt(wind.get('deg'))}° / {wind.get('compass') or '—'}")
            with r3: keyval("Slope (%)", _fmt(slope.get("grade_pct")))
            r4, r5, r6 = st.columns(3)
            with r4: keyval("Approach avg (%)", _fmt(appr.get("avg_pct")))
            with r5: keyval("Approach max (%)", _fmt(appr.get("max_pct")))
            with r6: keyval("Route sanity", "—" if rr is None else f"{rr:.2f}×")
            st.markdown("**Separations**")
            sL,sR = st.columns(2)
            with sL:
                keyval("Building", _fmt(feats.get("d_building_m")," m"))
                keyval("Road/footpath", _fmt(feats.get("d_road_m")," m"))
                keyval("Overhead", _fmt(feats.get("d_overhead_m")," m"))
                keyval("Watercourse", _fmt(feats.get("d_water_m")," m"))
            with sR:
                keyval("Boundary", _fmt(feats.get("d_boundary_m")," m"))
                keyval("Drain/manhole", _fmt(feats.get("d_drain_m")," m"))
                keyval("Railway", _fmt(feats.get("d_rail_m")," m"))
                keyval("Land use", feats.get("land_class","—"))
            if notes:
                st.markdown("**Access restrictions**"); pills(notes)
            if surf.get("risky_count",0)>0:
                st.markdown("**Surface flags**"); pills([f"{surf['risky_count']} flags"] + surf["samples"])

            st.markdown("### Risk result")
            st.metric(label="Score", value=f"{risk['score']}/100", delta=risk["status"])
            st.markdown("**Top contributing factors**")
            st.markdown("\n".join(f"- +{int(pts)} {msg}" for pts,msg in risk["explain"][:8]))

            # Map
            if MAPBOX_TOKEN:
                img = fetch_map(auto["lat"], auto["lon"])
                if img:
                    img = overlay_rings(img, auto["lat"])
                    st.image(img, caption="Static map (rings: 3 m / 6 m)", use_column_width=True)

        with colR:
            st.subheader("AI commentary")
            with st.expander("[1] Safety Risk Profile", expanded=True):
                st.write(ai.get("Safety Risk Profile",""))
            with st.expander("[2] Environmental Considerations", expanded=False):
                st.write(ai.get("Environmental Considerations",""))
            with st.expander("[3] Access & Logistics", expanded=False):
                st.write(ai.get("Access & Logistics",""))
            with st.expander("[4] Overall Site Suitability", expanded=False):
                st.write(ai.get("Overall Site Suitability",""))
            st.subheader("Recommended controls")
            if controls:
                for a in controls: st.markdown(f"- {a}")
            else:
                st.info("No additional controls triggered.")

            # PDF
            pdf_bytes = build_pdf_bytes(ctx)
            if pdf_bytes:
                st.download_button("📄 Download PDF report", data=pdf_bytes,
                                   file_name=f"precheck_{st.session_state['words'].replace('.','_')}.pdf",
                                   mime="application/pdf")
            else:
                st.caption("Install `reportlab` to enable PDF downloads.")

else:
    st.info("Enter a what3words address on the left and click **Fetch**. Then review the editable boxes and press **Confirm & Assess**.")
