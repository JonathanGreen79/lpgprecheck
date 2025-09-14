# app.py — Streamlit UI for LPG Pre-Check (no Excel required)
# pip install -r requirements.txt
import os, io, re, math, json, requests, pathlib, textwrap
from typing import Dict, List, Tuple, Optional, Any

import streamlit as st

# ============== Keys (Streamlit secrets -> local secrets_local.py -> env) ==============
def _load_secrets():
    W3W = getattr(st, "secrets", {}).get("W3W_API_KEY", None)
    MAPBOX = getattr(st, "secrets", {}).get("MAPBOX_TOKEN", None)
    OPENAI = getattr(st, "secrets", {}).get("OPENAI_API_KEY", None)
    if not any([W3W, MAPBOX, OPENAI]):
        try:
            from secrets_local import W3W_API_KEY as _W3W, MAPBOX_TOKEN as _MAP, OPENAI_API_KEY as _OA
            W3W, MAPBOX, OPENAI = _W3W, _MAP, _OA
        except Exception:
            W3W = W3W or os.getenv("W3W_API_KEY", "")
            MAPBOX = MAPBOX or os.getenv("MAPBOX_TOKEN", "")
            OPENAI = OPENAI or os.getenv("OPENAI_API_KEY", "")
    return W3W or "", MAPBOX or "", OPENAI or ""

W3W_API_KEY, MAPBOX_TOKEN, OPENAI_API_KEY = _load_secrets()

# ============== UI config ==============
st.set_page_config(page_title="LPG Pre-Check", page_icon="🛢️", layout="wide")

# ============== Defaults: CoP, Risk Weights, Controls, Questions ==============
DEFAULT_CONFIG = {
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
        "surface_gravel": {
            "when": "answers.surface_type in ['gravel','grass']",
            "actions": [
                "Upgrade stand area to hardstanding (concrete/asphalt) for stability and spill control."
            ]
        }
    },
    "questions": [
        {"key":"d_boundary_m","prompt":"Measured separation to boundary (m)","type":"float","required":False},
        {"key":"onsite_ignitions","prompt":"Any fixed ignition sources within 3 m?","type":"bool","required":False},
        {"key":"drain_within_3m","prompt":"Any drain/manhole within 3 m?","type":"bool","required":False},
        {"key":"surface_type","prompt":"Surface on tanker stand","type":"choice","choices":["asphalt","concrete","gravel","grass","other"],"required":False},
    ]
}

# ============== Geo helpers ==============
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

# ============== APIs ==============
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

OVERPASS="https://overpass-api.de/api/interpreter"; UA={"User-Agent":"LPG-Precheck-Streamlit/1.0"}
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

# ============== Risk / notes / surfaces / flood ==============
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
    cfg = cfg or DEFAULT_CONFIG
    CoP = cfg["cop"]; W = cfg["weights"]; B = cfg["bands"]

    score=0.0; why=[]
    def add(x,msg):
        nonlocal score
        try: x=float(x)
        except: x=0.0
        score+=x; why.append((x,msg))
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

# ============== AI commentary (optional OpenAI) ==============
def ai_sections(context: Dict) -> Dict[str,str]:
    def offline(ctx: Dict) -> Dict[str,str]:
        feats,wind,slope,appr,rr,flood,risk = (ctx.get(k,{}) for k in
            ["features","wind","slope","approach","route_ratio","flood","risk"])
        S1=(f"Local slope {slope.get('grade_pct','?')}% (aspect {int((slope.get('aspect_deg') or 0))}°). "
            f"Separations (m): bldg {feats.get('d_building_m')}, boundary {feats.get('d_boundary_m')}, "
            f"road {feats.get('d_road_m')}, drain {feats.get('d_drain_m')}, overhead {feats.get('d_overhead_m')}, rail {feats.get('d_rail_m')}. "
            f"Wind {(wind.get('speed_mps') or 0):.1f} m/s from {wind.get('compass') or 'n/a'}. "
            f"Heuristic {risk.get('score','?')}/100 → {risk.get('status','?')}. "
            f"Priorities: " + "; ".join(f"{int(p)} {m}" for p,m in risk.get("explain",[])[:5]) + ".")
        S2=(f"Flood {flood.get('level','n/a')} ({'; '.join(flood.get('why',[]))}). "
            f"Watercourse ~{feats.get('d_water_m','n/a')} m; drains/manholes {feats.get('d_drain_m','n/a')} m. Land use {feats.get('land_class','n/a')}.")
        S3=(f"Approach avg/max {appr.get('avg_pct','?')}/{appr.get('max_pct','?')}%. " +
            (f"Route {rr:.2f}× crow-fly. " if rr else "") +
            f"Check restrictions and surface; maintain signage and tanker stand quality.")
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
Be numeric, site-specific, and practical. Use auto data + user answers + overrides + risk result.
Context JSON:
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

# ============== PDF builder (minimal) ==============
def build_pdf_bytes(ctx: dict) -> bytes:
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.pdfbase import pdfmetrics
        from reportlab.lib.utils import ImageReader
    except Exception:
        return b""

    feats=ctx["features"]; wind=ctx["wind"]; slope=ctx["slope"]; appr=ctx["approach"]
    rr=ctx["route_ratio"]; flood=ctx["flood"]; risk=ctx["risk"]; counts=feats["counts"]
    addr=ctx["address"]; la_name=ctx.get("authority"); hospital=ctx.get("hospital") or {}
    overrides=ctx.get("overrides") or {}; overridden_keys=set(overrides.keys())
    ai=ctx.get("ai") or {}; controls=ctx.get("controls") or []

    def flag(k): return "*" if k in overridden_keys else ""
    buff=io.BytesIO()
    c=canvas.Canvas(buff, pagesize=A4)
    W,H=A4; M=38; y=H-46; LEAD=12; blue=colors.HexColor("#1f4e79")

    def ensure(h):
        nonlocal y
        if y-h<40: c.showPage(); y=H-46
    def head(txt, size=16):
        nonlocal y; ensure(size+6); c.setFillColor(blue); c.setFont("Helvetica-Bold",size); c.drawString(M,y,txt); y-=size+6; c.setFillColor(colors.black)
    def line(txt, size=10):
        nonlocal y; ensure(size+3); c.setFont("Helvetica",size); c.drawString(M,y,txt); y-=size+3
    def wrap(txt, size=10):
        nonlocal y
        c.setFont("Helvetica",size)
        width=W-2*M
        for para in (txt or "").split("\n"):
            para=para.rstrip()
            if not para: ensure(LEAD); y-=LEAD; continue
            words=para.split(); cur=""
            for w in words:
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
    line(f"Wind: {(wind.get('speed_mps') or 0):.1f} m/s from {wind.get('compass') or 'n/a'}{flag('wind.speed_mps')}")
    line(f"Slope: {slope.get('grade_pct','n/a')}% (aspect {int(slope.get('aspect_deg') or 0)}°){flag('slope.grade_pct')}")
    line(f"Approach: avg {appr.get('avg_pct','?')}% / max {appr.get('max_pct','?')}%{flag('approach.avg_pct')}{flag('approach.max_pct')}")
    line(f"Route sanity: {rr:.2f}× crow-fly" if rr else "Route sanity: n/a")
    line(f"Flood: {flood['level']} — " + "; ".join(flood["why"]))

    head("Separations",12)
    def fmt(v): return f"{v:.1f} m" if isinstance(v,(int,float)) else "n/a"
    for lbl, key in [
        ("Building","d_building_m"),("Boundary","d_boundary_m"),("Road/footpath","d_road_m"),
        ("Drain/manhole","d_drain_m"),("Overhead","d_overhead_m"),("Railway","d_rail_m"),("Watercourse","d_water_m")
    ]:
        line(f"{lbl}: {fmt(feats.get(key))}{flag('features.'+key)}")
    line(f"Land use: {feats.get('land_class','n/a')}{flag('features.land_class')}")

    head("Risk Score",12)
    line(f"Total: {risk['score']}/100 → {risk['status']}")
    for pts,msg in risk["explain"][:10]:
        line(f"+{int(pts)} {msg}",9)

    if controls:
        head("Recommended Controls",12)
        for a in controls:
            wrap(f"• {a}")

    for title in ["Safety Risk Profile","Environmental Considerations","Access & Logistics","Overall Site Suitability"]:
        head(title,12); wrap(ai.get(title,""))

    c.showPage(); c.save()
    return buff.getvalue()

# ============== UI ==============
st.title("🛢️ LPG Customer Tank — Pre-Check")

with st.sidebar:
    st.subheader("Location")
    words = st.text_input("what3words (word.word.word)", value="", placeholder="filled.count.soap")
    run_btn = st.button("Run Pre-Check", type="primary", use_container_width=True)
    st.markdown("---")
    st.subheader("Options")
    ask_human = st.checkbox("Ask site questions (boundary, ignitions, drain, surface)", value=True)
    allow_overrides = st.checkbox("Allow manual overrides", value=True)
    st.markdown("---")
    st.subheader("Secrets status")
    st.caption(f"W3W: {'✅' if W3W_API_KEY else '❌'} • Mapbox: {'✅' if MAPBOX_TOKEN else '❌'} • OpenAI: {'✅' if OPENAI_API_KEY else '❌'}")

if run_btn and words:
    if words.count(".")!=2 or not all(p.isalpha() for p in words.split(".")):
        st.error("Invalid what3words format. Use word.word.word"); st.stop()

    with st.status("Fetching auto data…", expanded=False):
        lat,lon = w3w_to_latlon(words)
        if lat is None:
            st.error("what3words lookup failed (check API key & words)."); st.stop()
        addr     = reverse_geocode(lat,lon)
        la_name  = addr.get("local_authority") or addr.get("county") or addr.get("state_district")
        wind     = open_meteo(lat,lon)
        slope    = slope_aspect(lat,lon,dx=20.0)
        osm      = overpass(lat,lon,int(DEFAULT_CONFIG["cop"]["poi_radius_m"]))
        feats    = parse_osm(lat,lon,osm)
        hospital = get_nearest_hospital_osm(lat,lon)
        if hospital: feats["d_hospital_m"] = round(hospital["distance_m"],1)
        appr     = approach_grade(lat,lon,feats.get("nearest_road_line"),N=6)
        rr       = osrm_ratio(lat,lon)
        notes    = restriction_notes(feats.get("restrictions",[]))
        surf     = surface_info(feats.get("surfaces",[]))
        flood    = flood_risk(feats, slope, slope.get("elev_m"))

    st.success("Auto data ready.")
    st.markdown("### Auto-derived preview")
    st.write({
        "building_m":feats.get("d_building_m"), "road_m":feats.get("d_road_m"), "drain_m":feats.get("d_drain_m"),
        "overhead_m":feats.get("d_overhead_m"), "rail_m":feats.get("d_rail_m"), "water_m":feats.get("d_water_m"),
        "land_class":feats.get("land_class"),
        "wind_mps":wind.get("speed_mps"), "wind_dir":wind.get("deg"), "slope_pct":slope.get("grade_pct"),
        "approach_max_pct":appr.get("max_pct")
    })

    # ========== Human Q&A ==========
    answers={}
    if ask_human:
        st.markdown("### Site questions (optional)")
        colA,colB,colC = st.columns(3)
        with colA:
            d_boundary = st.number_input("Boundary separation (m)", min_value=0.0, max_value=100.0, value=feats.get("d_boundary_m") or 0.0, step=0.1, format="%.1f")
            provide_boundary = st.checkbox("Use boundary value above", value=False)
        with colB:
            onsite_ignitions = st.toggle("Ignition source within 3 m?", value=False)
            drain_3m = st.toggle("Drain/manhole within 3 m?", value=False)
        with colC:
            surface_type = st.selectbox("Surface on tanker stand", options=["(leave unset)","asphalt","concrete","gravel","grass","other"], index=0)
        if provide_boundary: answers["d_boundary_m"] = float(d_boundary)
        answers["onsite_ignitions"] = bool(onsite_ignitions)
        answers["drain_within_3m"] = bool(drain_3m)
        if surface_type != "(leave unset)":
            answers["surface_type"] = surface_type

        # apply boundary into features if provided
        if "d_boundary_m" in answers:
            feats["d_boundary_m"] = answers["d_boundary_m"]

    # ========== Overrides ==========
    overrides={}
    if allow_overrides:
        st.markdown("### Manual overrides (optional)")
        with st.expander("Override auto values"):
            def ov_float(lbl, cur, minv=None, maxv=None, key=None, suffix=""):
                val = st.text_input(f"{lbl} [{('unknown' if cur is None else cur)}{suffix}]", key=key)
                if not val: return None
                s = val.strip().lower().replace(",", ".").replace(" m","").replace(" %","")
                try:
                    x = float(s)
                    if minv is not None and x < minv: return None
                    if maxv is not None and x > maxv: return None
                    return x
                except: return None
            f = feats; w = wind; sl = slope; ap = appr
            upd = {}
            upd["features.d_building_m"] = ov_float("Separation to nearest building (m)", f.get("d_building_m"), 0, 1000, "ov_bldg"," m")
            upd["features.d_boundary_m"] = ov_float("Separation to boundary (m)", f.get("d_boundary_m"), 0, 1000, "ov_bound"," m")
            upd["features.d_road_m"]     = ov_float("Separation to road/footpath (m)", f.get("d_road_m"), 0, 1000, "ov_road"," m")
            upd["features.d_drain_m"]    = ov_float("Separation to drain/manhole (m)", f.get("d_drain_m"), 0, 1000, "ov_drain"," m")
            upd["features.d_overhead_m"] = ov_float("Separation to overhead (m)", f.get("d_overhead_m"), 0, 1000, "ov_ovh"," m")
            upd["features.d_rail_m"]     = ov_float("Separation to railway (m)", f.get("d_rail_m"), 0, 1000, "ov_rail"," m")
            upd["features.d_water_m"]    = ov_float("Separation to watercourse (m)", f.get("d_water_m"), 0, 5000, "ov_water"," m")
            upd["wind.speed_mps"]        = ov_float("Wind speed (m/s)", w.get("speed_mps"), 0, 60, "ov_wind"," m/s")
            upd["wind.deg"]              = ov_float("Wind direction (°)", w.get("deg"), 0, 359, "ov_wdir"," °")
            upd["slope.grade_pct"]       = ov_float("Local slope (%)", sl.get("grade_pct"), 0, 100, "ov_slope"," %")
            upd["approach.avg_pct"]      = ov_float("Approach avg (%)", ap.get("avg_pct"), 0, 100, "ov_apavg"," %")
            upd["approach.max_pct"]      = ov_float("Approach max (%)", ap.get("max_pct"), 0, 100, "ov_apmax"," %")
            land_choice = st.selectbox("Land class", ["(keep)","Domestic/Urban","Industrial","Rural/Agricultural","Mixed"], index=0, key="ov_land")
            if land_choice != "(keep)":
                overrides["features.land_class"] = land_choice
            for k,v in upd.items():
                if v is not None:
                    overrides[k]=v

        # apply overrides
        def apply_overrides(ctx, ovs):
            for path, val in ovs.items():
                cur = ctx
                parts = path.split(".")
                for p in parts[:-1]:
                    cur = cur.setdefault(p, {})
                cur[parts[-1]] = val
        ctx_temp={"features":feats,"wind":wind,"slope":slope,"approach":appr}
        apply_overrides(ctx_temp, overrides)
        feats,wind,slope,appr = ctx_temp["features"],ctx_temp["wind"],ctx_temp["slope"],ctx_temp["approach"]

    # ========== Final risk + AI + layout ==========
    risk = risk_score(feats, wind, slope, appr, rr, notes, surf, flood, answers=answers, cfg=DEFAULT_CONFIG)
    ctx = {
        "cop": DEFAULT_CONFIG["cop"], "words":words,"lat":lat,"lon":lon,
        "address":addr,"authority":la_name,"hospital":hospital,
        "wind":wind,"slope":slope,"features":feats,"approach":appr,
        "route_ratio":rr,"restrictions":notes,"surfaces":surf,"flood":flood,
        "answers":answers,"overrides":overrides,"risk":risk,
    }
    controls=[]
    # controls evaluation
    def _safe_get(dotted):
        cur=ctx
        for p in dotted.split("."):
            cur = cur.get(p) if isinstance(cur,dict) else None
        return cur
    def _eval_when(expr: str) -> bool:
        try:
            # very simple templating
            tokens=re.findall(r"(answers|features|cop|wind|slope|approach|risk)(?:\.[a-zA-Z0-9_]+)+", expr)
            expr2=expr
            for t in tokens:
                expr2=expr2.replace(t, repr(_safe_get(t)))
            return bool(eval(expr2, {"__builtins__": {}}, {}))
        except Exception:
            return False
    for key, rule in DEFAULT_CONFIG["controls"].items():
        if _eval_when(rule.get("when","False")):
            controls+=rule.get("actions",[])
    # dedupe
    dedup=[]; seen=set()
    for a in controls:
        if a not in seen: seen.add(a); dedup.append(a)
    controls=dedup
    ctx["controls"]=controls

    ai = ai_sections(ctx)
    ctx["ai"] = ai

    colL, colR = st.columns([0.55, 0.45], gap="large")

    with colL:
        st.subheader("Key metrics")
        st.write({
            "Wind (m/s)": wind.get("speed_mps"),
            "Wind dir (°/compass)": f"{wind.get('deg')} / {wind.get('compass')}",
            "Slope (%)": slope.get("grade_pct"),
            "Approach avg/max (%)": f"{appr.get('avg_pct')}/{appr.get('max_pct')}",
            "Route sanity (× crow-fly)": None if rr is None else round(rr,2),
            "Flood": f"{flood['level']} — {'; '.join(flood['why'])}",
        })
        st.markdown("**Separations (~400 m)**")
        st.write({
            "Building (m)": feats.get("d_building_m"),
            "Boundary (m)": feats.get("d_boundary_m"),
            "Road/footpath (m)": feats.get("d_road_m"),
            "Drain/manhole (m)": feats.get("d_drain_m"),
            "Overhead (m)": feats.get("d_overhead_m"),
            "Railway (m)": feats.get("d_rail_m"),
            "Watercourse (m)": feats.get("d_water_m"),
            "Land use": feats.get("land_class"),
        })
        if notes:
            st.markdown("**Access restrictions**")
            st.write(notes)
        sf=surf
        if sf.get("risky_count",0)>0:
            st.markdown("**Surface flags**")
            st.write({"count": sf["risky_count"], "samples": sf["samples"]})

        st.markdown("### Risk result")
        st.metric(label="Score", value=f"{risk['score']}/100", delta=risk["status"])
        st.markdown("**Top contributing factors**")
        st.write([f"+{int(pts)} {msg}" for pts,msg in risk["explain"][:8]])

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

        # PDF download
        pdf_bytes = build_pdf_bytes(ctx)
        if pdf_bytes:
            st.download_button("📄 Download PDF report", data=pdf_bytes, file_name=f"precheck_{words.replace('.','_')}.pdf", mime="application/pdf")
        else:
            st.caption("Install `reportlab` to enable PDF downloads.")

else:
    st.info("Enter a what3words address and click **Run Pre-Check**.")
