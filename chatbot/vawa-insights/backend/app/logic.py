"""
VAWA Insights Bot — non-API logic in one module.

Split into multiple files later only when this file grows hard to navigate.
"""

from __future__ import annotations

import csv
import json
import os
import ast
import re
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from app.schemas import ChatRequest, ChatResponse, MapEmbedPayload

# ---------------------------------------------------------------------------
# Data loader (CSV under app/data/)
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"

# Public ArcGIS Dashboard (override with ARCGIS_DASHBOARD_URL for staging/production).
DEFAULT_ARCGIS_DASHBOARD_URL = (
    "https://sdsugeo.maps.arcgis.com/apps/dashboards/7fa98d321eb94e9baaffc43fe65b973e"
)

_ARCGIS_TAB_WEBMAP_IDS: Dict[str, str] = {
    # Derived from the dashboard configuration (mapWidget itemIds).
    "Master": "e5e41a3c32084b33958bce654395327f",
    "Sexual Assault": "a0b8ad4cf3094db285b4733ce3f6c8b7",
    "Violent Crimes": "b4c812bc8b894c09977cd77fe2a18072",
    "Tribal": "58e315d44c1e422c99dd0f1044645c9f",
    "Domestic Violence": "21e1d8d284ff49758fe767aa6cfe0a57",
    "Firearm": "7f647a063be04721bf95603e31b6a26c",
    "Shelter Locations": "c7557b6ffa9049938605cba6c8d28b14",
    "Rural Locations": "5c2eda65c42242d9a2c31fa78ef967f2",
    "College": "45148dd88ee740f8b5cad9a2b12709c7",
    "Race": "b71d1048a8b84ad0b9b441471f1b85dc",
}

_METRIC_LABELS: Dict[str, str] = {
    "dv_rate": "Domestic violence rate",
    "sexual_assault_rate": "Sexual assault rate",
    "firearm_share": "Firearm involvement share",
    "dating_partner_share": "Dating partner share",
    "minority_victim_share": "Minority victim share",
    "native_american_victim_share": "Native American victim share",
    "reporting_proxy": "Reporting proxy",
    "risk_index": "Risk index",
}

_METRIC_DEFINITIONS: Dict[str, str] = {
    "dv_rate": "Project-defined domestic violence rate for the selected geography and period.",
    "sexual_assault_rate": "Project-defined sexual assault rate for the selected geography and period.",
    "firearm_share": "The share of incidents involving a firearm (a proportion from 0 to 1).",
    "dating_partner_share": "The share of incidents involving a dating/nonmarried partner (a proportion from 0 to 1).",
    "minority_victim_share": "The share of victims identified as a racial/ethnic minority (a proportion from 0 to 1).",
    "native_american_victim_share": "The share of victims identified as Native American (a proportion from 0 to 1).",
    "reporting_proxy": "A project-defined proxy indicating reporting/coverage intensity; interpret cautiously.",
    "risk_index": "A composite index summarizing multiple risk-related components (project-defined).",
}

_METRIC_UNITS: Dict[str, str] = {
    "dv_rate": "rate per 100,000 female population",
    "sexual_assault_rate": "rate per 100,000 female population",
    "firearm_share": "proportion (0–1)",
    "dating_partner_share": "proportion (0–1)",
    "minority_victim_share": "proportion (0–1)",
    "native_american_victim_share": "proportion (0–1)",
    "reporting_proxy": "rate per 100,000 female population (proxy)",
    "risk_index": "standardized index (unitless; higher means higher relative risk)",
}


def _metric_definition(metric: Optional[str]) -> str:
    m = (metric or "").strip()
    if not m:
        return ""
    return _METRIC_DEFINITIONS.get(m, "")


def _metric_units(metric: Optional[str]) -> str:
    m = (metric or "").strip()
    if not m:
        return ""
    return _METRIC_UNITS.get(m, "")


def _metric_explain_sentence(metric: Optional[str]) -> str:
    """
    Short suffix used in direct answers whenever we display numbers.
    """
    m = (metric or "").strip()
    if not m:
        return ""
    units = _metric_units(m)
    definition = _metric_definition(m)
    bits: List[str] = []
    if units:
        bits.append(f"Units are {units}.")
    if definition:
        bits.append(definition)
    if not bits:
        return ""
    return " " + " ".join(bits)


def _metric_display_name(metric: Optional[str]) -> str:
    m = (metric or "").strip()
    if not m:
        return ""
    return _METRIC_LABELS.get(m, m.replace("_", " "))

METRICS_FILE = DATA_DIR / "metrics.csv"
RISK_COMPONENTS_FILE = DATA_DIR / "risk_components.csv"
RESOURCES_FILE = DATA_DIR / "resources.csv"

# EDA outputs (kept outside chatbot to keep backend small).
REPO_ROOT = Path(__file__).resolve().parents[4]
EDA_OUTPUT_DIR = REPO_ROOT / "EDA" / "output"
POLICY_VARIABLE_LEVEL_FILE = EDA_OUTPUT_DIR / "policy_variable_level_summary.csv"
POLICY_STATE_LEVEL_CHANGES_FILE = EDA_OUTPUT_DIR / "policy_state_level_changes.csv"

METRIC_COLUMNS = [
    "dv_rate",
    "sexual_assault_rate",
    "firearm_share",
    "dating_partner_share",
    "minority_victim_share",
    "native_american_victim_share",
    "reporting_proxy",
    "risk_index",
]


@dataclass(frozen=True)
class MetricsRow:
    geo_id: str
    geo_name: str
    geo_type: str
    state: str
    county: str
    tribal_area: str
    year: int
    quarter: str
    month: str
    data_quality_flag: str
    metrics: Dict[str, Optional[float]]


@dataclass(frozen=True)
class RiskComponentRow:
    geo_id: str
    geo_name: str
    year: int
    component: str
    value: float
    note: str


@dataclass(frozen=True)
class ResourceRow:
    resource_id: str
    name: str
    category: str
    subcategory: str
    services: str
    address: str
    city: str
    state: str
    postal_code: str
    country: str
    phone: str
    website: str
    latitude: Optional[float]
    longitude: Optional[float]
    notes: str
    source: str


_CACHE: Dict[str, Any] = {}


def _to_float_or_none(x: str) -> Optional[float]:
    x = (x or "").strip()
    if x == "" or x.lower() in {"na", "nan", "none"}:
        return None
    return float(x)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_metrics_rows() -> List[MetricsRow]:
    if "metrics_rows" in _CACHE:
        return _CACHE["metrics_rows"]

    if not METRICS_FILE.exists():
        raise FileNotFoundError(f"Missing metrics file: {METRICS_FILE}")

    raw = _read_csv(METRICS_FILE)
    rows: List[MetricsRow] = []
    for r in raw:
        metrics: Dict[str, Optional[float]] = {}
        for m in METRIC_COLUMNS:
            metrics[m] = _to_float_or_none(r.get(m, ""))

        rows.append(
            MetricsRow(
                geo_id=r["geo_id"],
                geo_name=r["geo_name"],
                geo_type=r["geo_type"],
                state=r.get("state", ""),
                county=r.get("county", ""),
                tribal_area=r.get("tribal_area", ""),
                year=int(r["year"]),
                quarter=r.get("quarter", ""),
                month=r.get("month", ""),
                data_quality_flag=r.get("data_quality_flag", ""),
                metrics=metrics,
            )
        )

    _CACHE["metrics_rows"] = rows
    return rows


def load_risk_components_rows() -> List[RiskComponentRow]:
    if "risk_rows" in _CACHE:
        return _CACHE["risk_rows"]

    if not RISK_COMPONENTS_FILE.exists():
        raise FileNotFoundError(f"Missing risk components file: {RISK_COMPONENTS_FILE}")

    raw = _read_csv(RISK_COMPONENTS_FILE)
    rows: List[RiskComponentRow] = []
    for r in raw:
        rows.append(
            RiskComponentRow(
                geo_id=r["geo_id"],
                geo_name=r["geo_name"],
                year=int(r["year"]),
                component=r["component"],
                value=float(r["value"]),
                note=r.get("note", ""),
            )
        )

    _CACHE["risk_rows"] = rows
    return rows


def _to_float_or_none2(x: str) -> Optional[float]:
    x = (x or "").strip()
    if x == "" or x.lower() in {"na", "nan", "none"}:
        return None
    try:
        return float(x)
    except Exception:
        return None


def load_resource_rows() -> List[ResourceRow]:
    if "resource_rows" in _CACHE:
        return _CACHE["resource_rows"]

    if not RESOURCES_FILE.exists():
        _CACHE["resource_rows"] = []
        return []

    raw = _read_csv(RESOURCES_FILE)
    rows: List[ResourceRow] = []
    for r in raw:
        rows.append(
            ResourceRow(
                resource_id=(r.get("resource_id") or "").strip(),
                name=(r.get("name") or "").strip(),
                category=(r.get("category") or "").strip(),
                subcategory=(r.get("subcategory") or "").strip(),
                services=(r.get("services") or "").strip(),
                address=(r.get("address") or "").strip(),
                city=(r.get("city") or "").strip(),
                state=(r.get("state") or "").strip(),
                postal_code=(r.get("postal_code") or "").strip(),
                country=(r.get("country") or "").strip(),
                phone=(r.get("phone") or "").strip(),
                website=(r.get("website") or "").strip(),
                latitude=_to_float_or_none2(r.get("latitude", "")),
                longitude=_to_float_or_none2(r.get("longitude", "")),
                notes=(r.get("notes") or "").strip(),
                source=(r.get("source") or "").strip(),
            )
        )

    _CACHE["resource_rows"] = rows
    return rows


# ---------------------------------------------------------------------------
# Victim resources lookup (geocoding + distance search)
# ---------------------------------------------------------------------------


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 3958.7613  # Earth radius in miles
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def _extract_lat_lon(text: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract a (lat, lon) pair from text like:
    - "32.7157, -117.1611"
    - "lat 32.7157 lon -117.1611"
    """
    t = (text or "").strip()
    m = re.search(r"(-?\d{1,2}\.\d+)\s*,\s*(-?\d{1,3}\.\d+)", t)
    if m:
        try:
            return float(m.group(1)), float(m.group(2))
        except Exception:
            return None, None
    m = re.search(r"\blat(?:itude)?\s*[:=]?\s*(-?\d{1,2}\.\d+)\b.*\blon(?:gitude)?\s*[:=]?\s*(-?\d{1,3}\.\d+)\b", t, re.I)
    if m:
        try:
            return float(m.group(1)), float(m.group(2))
        except Exception:
            return None, None
    return None, None


def _extract_location_phrase(message: str) -> str:
    """
    Best-effort extraction of a location phrase from a natural-language query.
    """
    t = (message or "").strip()
    if not t:
        return ""

    m = re.search(r"\bnear\s+(.+)$", t, re.I)
    if m:
        loc = m.group(1).strip()
        loc = re.split(r"[?.!]", loc)[0].strip()
        return loc

    m = re.search(r"\bin\s+([A-Za-z][^?.!]+)$", t, re.I)
    if m and len(m.group(1).strip()) <= 80:
        loc = m.group(1).strip()
        loc = re.split(r"[?.!]", loc)[0].strip()
        return loc

    return ""


def _geocode_location(location: str) -> Dict[str, Any]:
    """
    Free geocoding via OpenStreetMap Nominatim.
    - Respects a tiny in-memory cache.
    - Requires network access at runtime.
    """
    loc = (location or "").strip()
    if not loc:
        return {"ok": False, "error": "Missing location.", "data": None}

    cache_key = f"geocode::{loc.lower()}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    try:
        import requests  # type: ignore

        # Small throttle to avoid accidental rapid-fire calls during dev refresh loops.
        now = time.time()
        last = float(_CACHE.get("_geocode_last_ts", 0.0) or 0.0)
        if now - last < 1.0:
            time.sleep(1.0 - (now - last))

        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": loc, "format": "json", "limit": 1, "addressdetails": 1}
        headers = {"User-Agent": "vawa-insights-bot/0.1 (educational project)"}
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        _CACHE["_geocode_last_ts"] = time.time()
        if resp.status_code != 200:
            out = {"ok": False, "error": f"Geocoding failed (status {resp.status_code}).", "data": None}
            _CACHE[cache_key] = out
            return out

        data = resp.json()
        if not isinstance(data, list) or not data:
            out = {"ok": False, "error": "No geocoding results found for that location.", "data": None}
            _CACHE[cache_key] = out
            return out

        hit = data[0]
        lat = _to_float_or_none2(str(hit.get("lat", "")))
        lon = _to_float_or_none2(str(hit.get("lon", "")))
        disp = str(hit.get("display_name") or loc).strip()
        if lat is None or lon is None:
            out = {"ok": False, "error": "Geocoding returned no usable coordinates.", "data": None}
            _CACHE[cache_key] = out
            return out

        out = {"ok": True, "data": {"display_name": disp, "latitude": lat, "longitude": lon}, "citation": {"citation_type": "geocoding", "provider": "OpenStreetMap Nominatim", "query": loc}}
        _CACHE[cache_key] = out
        return out
    except Exception:
        out = {"ok": False, "error": "Geocoding request failed.", "data": None}
        _CACHE[cache_key] = out
        return out


def find_victim_resources(
    *,
    query: str = "",
    location: str = "",
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    radius_miles: float = 25.0,
    limit: int = 8,
    categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    rows = load_resource_rows()
    if not rows:
        return {"ok": False, "error": f"No resources dataset found at {RESOURCES_FILE.name}.", "data": None}

    radius_miles = float(radius_miles or 25.0)
    radius_miles = max(1.0, min(radius_miles, 250.0))
    limit = int(limit or 8)
    limit = max(1, min(limit, 25))

    q = (query or "").lower()
    cats = [c.strip().lower() for c in (categories or []) if c and c.strip()]

    # Resolve coordinates
    resolved_location = (location or "").strip()
    lat = latitude
    lon = longitude
    citations: List[Dict[str, Any]] = [
        {"citation_type": "structured_data", "source_table": RESOURCES_FILE.name}
    ]

    if lat is None or lon is None:
        # Try to pull coords directly from free text
        lat2, lon2 = _extract_lat_lon(location or query or "")
        lat = lat if lat is not None else lat2
        lon = lon if lon is not None else lon2

    if (lat is None or lon is None) and resolved_location:
        geo = _geocode_location(resolved_location)
        if geo.get("ok"):
            lat = geo["data"]["latitude"]
            lon = geo["data"]["longitude"]
            resolved_location = geo["data"]["display_name"]
            if geo.get("citation"):
                citations.append(geo["citation"])
        else:
            # Still allow "national-only" results even if geocoding fails
            citations.append({"citation_type": "geocoding", "provider": "OpenStreetMap Nominatim", "query": resolved_location, "error": geo.get("error")})

    def row_matches_text(r: ResourceRow) -> bool:
        if not q:
            return True
        blob = " ".join([r.name, r.category, r.subcategory, r.services, r.city, r.state, r.notes]).lower()
        toks = [t for t in re.findall(r"[a-z0-9]+", q) if t not in {"find", "a", "an", "the", "near", "in", "for", "me", "please"}]
        if not toks:
            return True
        return any(tok in blob for tok in toks)

    def row_matches_category(r: ResourceRow) -> bool:
        if not cats:
            return True
        return (r.category or "").strip().lower() in cats

    out_rows: List[Dict[str, Any]] = []
    for r in rows:
        if not r.resource_id or not r.name:
            continue
        if not row_matches_category(r):
            continue
        if not row_matches_text(r):
            continue

        dist = None
        if lat is not None and lon is not None and r.latitude is not None and r.longitude is not None:
            dist = _haversine_miles(lat, lon, r.latitude, r.longitude)
            if dist > radius_miles:
                continue

        out_rows.append(
            {
                "resource_id": r.resource_id,
                "name": r.name,
                "category": r.category,
                "subcategory": r.subcategory,
                "services": r.services,
                "address": r.address,
                "city": r.city,
                "state": r.state,
                "postal_code": r.postal_code,
                "phone": r.phone,
                "website": r.website,
                "distance_miles": (round(dist, 2) if dist is not None else None),
                "notes": r.notes,
            }
        )

    if lat is not None and lon is not None:
        out_rows.sort(key=lambda x: (x["distance_miles"] is None, x["distance_miles"] or 9e9))
    else:
        # Without coordinates: show hotline-like/national resources first
        out_rows.sort(key=lambda x: (x.get("state") != "", x.get("city") != "", x.get("name", "")))

    out_rows = out_rows[:limit]

    return {
        "ok": True,
        "data": {
            "resolved_location": resolved_location,
            "latitude": lat,
            "longitude": lon,
            "radius_miles": radius_miles,
            "results": out_rows,
        },
        "citations": citations,
    }


# ---------------------------------------------------------------------------
# Policy stats tools (pre vs post 2022) from EDA/output
# ---------------------------------------------------------------------------

POLICY_VARIABLE_ALIASES: Dict[str, str] = {
    "total incidents": "total_incidents",
    "incidents": "total_incidents",
    "domestic violence": "dv_total",
    "dv total": "dv_total",
    "sexual assaults": "sex_assaults",
    "sexual assault": "sex_assaults",
    "firearm involvement": "involving_firearm",
    "gun involvement": "involving_firearm",
    "nonmarried partner": "victim_offender_nonmarried_partner",
    "unmarried partner": "victim_offender_nonmarried_partner",
    "dating partner": "victim_offender_nonmarried_partner",
    "spouse": "victim_offender_rel_spouse",
    "near schools": "near_school",
    "near school": "near_school",
    "tribal lands": "on_tribal_lands",
    "on tribal lands": "on_tribal_lands",
}


def _load_policy_rows(path: Path, cache_key: str) -> List[Dict[str, str]]:
    if cache_key in _CACHE:
        return _CACHE[cache_key]
    if not path.exists():
        _CACHE[cache_key] = []
        return []
    rows = _read_csv(path)
    _CACHE[cache_key] = rows
    return rows


def list_policy_variables() -> List[str]:
    rows = _load_policy_rows(POLICY_VARIABLE_LEVEL_FILE, "policy_variable_level_rows")
    vars_ = sorted({(r.get("variable") or "").strip() for r in rows if (r.get("variable") or "").strip()})
    return vars_


def detect_policy_variable(text: str) -> Optional[str]:
    t = (text or "").lower()
    for phrase, var in POLICY_VARIABLE_ALIASES.items():
        if phrase in t:
            return var
    for v in list_policy_variables():
        if v.lower() in t:
            return v
    return None


def _as_float(x: str) -> Optional[float]:
    try:
        return float((x or "").strip())
    except Exception:
        return None


def _as_int(x: str) -> Optional[int]:
    try:
        return int(float((x or "").strip()))
    except Exception:
        return None


def get_policy_variable_summary(variable: str) -> Dict[str, Any]:
    v = (variable or "").strip()
    if not v:
        return {"ok": False, "error": "Missing variable.", "data": None}

    rows = _load_policy_rows(POLICY_VARIABLE_LEVEL_FILE, "policy_variable_level_rows")
    match = next((r for r in rows if (r.get("variable") or "").strip() == v), None)
    if not match:
        return {"ok": False, "error": f"No policy summary found for variable '{v}'.", "data": None}

    def parse_top_list(s: str) -> List[Dict[str, Any]]:
        s = (s or "").strip()
        if not s:
            return []
        try:
            val = ast.literal_eval(s)
            return val if isinstance(val, list) else []
        except Exception:
            return []

    data = {
        "variable": v,
        "variable_type": (match.get("variable_type") or "").strip(),
        "states_with_valid_percent_change": _as_int(match.get("states_with_valid_percent_change", "")),
        "mean_absolute_change": _as_float(match.get("mean_absolute_change", "")),
        "median_absolute_change": _as_float(match.get("median_absolute_change", "")),
        "mean_percent_change": _as_float(match.get("mean_percent_change", "")),
        "median_percent_change": _as_float(match.get("median_percent_change", "")),
        "states_increased": _as_int(match.get("states_increased", "")),
        "states_decreased": _as_int(match.get("states_decreased", "")),
        "states_no_change": _as_int(match.get("states_no_change", "")),
        "top_5_largest_decreases": parse_top_list(match.get("top_5_largest_decreases", "")),
        "top_5_largest_increases": parse_top_list(match.get("top_5_largest_increases", "")),
        "source_file": POLICY_VARIABLE_LEVEL_FILE.name,
    }

    citation = {
        "citation_type": "structured_data",
        "source_table": POLICY_VARIABLE_LEVEL_FILE.name,
        "variable": v,
        "periods": ["pre_2022", "post_2022_avg"],
    }

    return {"ok": True, "data": data, "citation": citation}


def rank_states_by_policy_change(variable: str, direction: Literal["increase", "decrease"], top_n: int = 5, *, min_pre_value: float) -> Dict[str, Any]:
    v = (variable or "").strip()
    if not v:
        return {"ok": False, "error": "Missing variable.", "data": None}
    top_n = max(1, min(int(top_n or 5), 20))

    rows = _load_policy_rows(POLICY_STATE_LEVEL_CHANGES_FILE, "policy_state_level_changes_rows")
    filtered = [r for r in rows if (r.get("variable") or "").strip() == v]
    if not filtered:
        return {"ok": False, "error": f"No rows found for variable '{v}'.", "data": None}

    out = []
    for r in filtered:
        pc = _as_float(r.get("percent_change", ""))
        if pc is None:
            continue
        pre = _as_float(r.get("pre_2022", ""))
        if pre is not None and pre < float(min_pre_value):
            continue
        out.append(
            {
                "state": (r.get("state") or "").strip(),
                "percent_change": pc,
                "absolute_change": _as_float(r.get("absolute_change", "")),
                "pre_2022": pre,
                "post_2022_avg": _as_float(r.get("post_2022_avg", "")),
                "direction": (r.get("direction") or "").strip(),
            }
        )

    if not out:
        return {"ok": False, "error": f"No numeric percent_change values found for variable '{v}'.", "data": None}

    reverse = direction == "increase"
    out.sort(key=lambda x: x["percent_change"], reverse=reverse)
    ranked = out[:top_n]
    for i, r in enumerate(ranked, start=1):
        r["rank"] = i

    citation = {
        "citation_type": "structured_data",
        "source_table": POLICY_STATE_LEVEL_CHANGES_FILE.name,
        "variable": v,
        "sort": f"percent_change_{direction}",
        "top_n": top_n,
        "filters": {"min_pre_value": min_pre_value},
        "periods": ["pre_2022", "post_2022_avg"],
    }

    return {"ok": True, "data": {"variable": v, "direction": direction, "ranked": ranked}, "citation": citation}

def find_geo_ids_by_name(name: str) -> List[Tuple[str, str, str]]:
    name_norm = (name or "").strip().lower()
    if not name_norm:
        return []

    rows = load_metrics_rows()
    seen = {}
    for r in rows:
        key = (r.geo_id, r.geo_name, r.geo_type)
        if key in seen:
            continue
        gn = r.geo_name.lower()
        if name_norm == gn or name_norm in gn:
            seen[key] = True

    return list(seen.keys())


# ---------------------------------------------------------------------------
# Query parsing
# ---------------------------------------------------------------------------

METRIC_ALIASES: Dict[str, str] = {
    "dv rate": "dv_rate",
    "domestic violence": "dv_rate",
    "sexual assault": "sexual_assault_rate",
    "firearm": "firearm_share",
    "dating partner": "dating_partner_share",
    "minority": "minority_victim_share",
    "native american": "native_american_victim_share",
    "reporting": "reporting_proxy",
    "risk index": "risk_index",
}


def extract_years(text: str) -> List[int]:
    years = set()
    for m in re.finditer(r"\b(2021|2022|2023|2024)\b", text or ""):
        years.add(int(m.group(0)))
    return sorted(years)


def detect_metric(text: str) -> Optional[str]:
    t = (text or "").lower()

    for m in METRIC_COLUMNS:
        if m in t:
            return m

    for phrase, metric in METRIC_ALIASES.items():
        if phrase in t:
            return metric

    return None


def detect_geo_names(text: str) -> List[str]:
    t = (text or "").lower()
    geo_names = sorted({r.geo_name for r in load_metrics_rows()}, key=len, reverse=True)
    found: List[str] = []
    for name in geo_names:
        if name.lower() in t:
            found.append(name)
    out = []
    for x in found:
        if x not in out:
            out.append(x)
    return out


def resolve_geo(geo_name: str) -> Optional[Dict[str, str]]:
    candidates = find_geo_ids_by_name(geo_name)
    if not candidates:
        return {"state": geo_name}
    geo_id, name, geo_type = candidates[0]
    return {"geo_id": geo_id, "geo_name": name, "geo_type": geo_type}


# ---------------------------------------------------------------------------
# Structured data tools
# ---------------------------------------------------------------------------

Frequency = Literal["year", "quarter", "month"]
SortDirection = Literal["asc", "desc"]
GeoLevel = Literal["state", "county", "tribal"]


def _metric_exists(metric: str) -> bool:
    return metric in METRIC_COLUMNS


def _filter_rows_by_geo(rows, geo: Dict[str, str]):
    geo_id = (geo.get("geo_id") or "").strip()
    if geo_id:
        return [r for r in rows if r.geo_id == geo_id]

    geo_name = (geo.get("geo_name") or "").strip().lower()
    if geo_name:
        return [r for r in rows if r.geo_name.lower() == geo_name]

    state = (geo.get("state") or "").strip().lower()
    if state:
        return [r for r in rows if r.geo_type == "state" and r.geo_name.lower() == state]

    return []


def get_metric_timeseries(geo: Dict[str, str], metric: str, frequency: Frequency) -> Dict[str, Any]:
    if not _metric_exists(metric):
        return {"ok": False, "error": f"Unsupported metric '{metric}'.", "data": None}

    rows = _filter_rows_by_geo(load_metrics_rows(), geo)
    if not rows:
        return {"ok": False, "error": "No matching geography found in the dataset.", "data": None}

    points = []
    for r in rows:
        if frequency == "year":
            period = str(r.year)
        elif frequency == "quarter":
            period = f"{r.year} {r.quarter}"
        elif frequency == "month":
            period = f"{r.year}-{r.month}"
        else:
            return {"ok": False, "error": f"Unsupported frequency '{frequency}'.", "data": None}

        val = r.metrics.get(metric)
        if val is None:
            continue
        points.append({"period": period, "value": val, "data_quality_flag": r.data_quality_flag})

    if not points:
        return {"ok": False, "error": "Metric exists but no non-missing values were found for that geo.", "data": None}

    points.sort(key=lambda x: x["period"])

    first = rows[0]
    citation = {
        "citation_type": "structured_data",
        "source_table": "metrics.csv",
        "geo_id": first.geo_id,
        "geo_name": first.geo_name,
        "geo_type": first.geo_type,
        "metric": metric,
        "frequency": frequency,
        "years_covered": sorted({r.year for r in rows}),
    }

    return {"ok": True, "data": {"geo": asdict(first), "metric": metric, "frequency": frequency, "points": points}, "citation": citation}


def compare_geos(
    geo_a: Dict[str, str],
    geo_b: Dict[str, str],
    metric: str,
    start_period: str,
    end_period: str,
) -> Dict[str, Any]:
    if not _metric_exists(metric):
        return {"ok": False, "error": f"Unsupported metric '{metric}'.", "data": None}

    try:
        start_year = int(start_period)
        end_year = int(end_period)
    except ValueError:
        return {"ok": False, "error": "compare_geos supports year periods like '2021'..'2024'.", "data": None}

    rows = load_metrics_rows()
    a_rows = [r for r in _filter_rows_by_geo(rows, geo_a) if start_year <= r.year <= end_year and r.metrics.get(metric) is not None]
    b_rows = [r for r in _filter_rows_by_geo(rows, geo_b) if start_year <= r.year <= end_year and r.metrics.get(metric) is not None]

    if not a_rows or not b_rows:
        return {"ok": False, "error": "Not enough data to compare both geographies for the requested years.", "data": None}

    def yearly_avg(rs):
        vals = [r.metrics[metric] for r in rs if r.metrics.get(metric) is not None]
        return sum(vals) / len(vals) if vals else None

    a_avg = yearly_avg(a_rows)
    b_avg = yearly_avg(b_rows)
    if a_avg is None or b_avg is None:
        return {"ok": False, "error": "Metric values were missing in the requested range.", "data": None}

    a0 = a_rows[0]
    b0 = b_rows[0]
    citation = {
        "citation_type": "structured_data",
        "source_table": "metrics.csv",
        "metric": metric,
        "start_year": start_year,
        "end_year": end_year,
        "geos": [
            {"geo_id": a0.geo_id, "geo_name": a0.geo_name, "geo_type": a0.geo_type},
            {"geo_id": b0.geo_id, "geo_name": b0.geo_name, "geo_type": b0.geo_type},
        ],
    }

    return {
        "ok": True,
        "data": {
            "metric": metric,
            "start_year": start_year,
            "end_year": end_year,
            "geo_a": {"geo_id": a0.geo_id, "geo_name": a0.geo_name, "geo_type": a0.geo_type, "avg_value": a_avg},
            "geo_b": {"geo_id": b0.geo_id, "geo_name": b0.geo_name, "geo_type": b0.geo_type, "avg_value": b_avg},
            "difference": a_avg - b_avg,
        },
        "citation": citation,
    }


def rank_geos(metric: str, year: int, geo_level: GeoLevel, top_n: int, sort_direction: SortDirection) -> Dict[str, Any]:
    if not _metric_exists(metric):
        return {"ok": False, "error": f"Unsupported metric '{metric}'.", "data": None}

    rows = [r for r in load_metrics_rows() if r.year == year and r.geo_type == geo_level and r.metrics.get(metric) is not None]
    if not rows:
        return {"ok": False, "error": "No rows matched that year and geography level.", "data": None}

    reverse = sort_direction == "desc"
    rows.sort(key=lambda r: r.metrics[metric], reverse=reverse)  # type: ignore[index]

    top_n = max(1, min(top_n, len(rows)))
    ranked = []
    for i, r in enumerate(rows[:top_n], start=1):
        ranked.append(
            {
                "rank": i,
                "geo_id": r.geo_id,
                "geo_name": r.geo_name,
                "geo_type": r.geo_type,
                "value": r.metrics[metric],
                "data_quality_flag": r.data_quality_flag,
            }
        )

    citation = {
        "citation_type": "structured_data",
        "source_table": "metrics.csv",
        "metric": metric,
        "year": year,
        "geo_level": geo_level,
        "sort_direction": sort_direction,
    }

    return {"ok": True, "data": {"metric": metric, "year": year, "geo_level": geo_level, "ranked": ranked}, "citation": citation}


def get_risk_profile(geo: Dict[str, str], year: int) -> Dict[str, Any]:
    metrics_rows = _filter_rows_by_geo(load_metrics_rows(), geo)
    metrics_rows = [r for r in metrics_rows if r.year == year]
    if not metrics_rows:
        return {"ok": False, "error": "No matching geo/year found for risk profile.", "data": None}

    r0 = metrics_rows[0]
    risk_index = r0.metrics.get("risk_index")
    if risk_index is None:
        return {"ok": False, "error": "Risk index is missing for that geo/year.", "data": None}

    comps = [c for c in load_risk_components_rows() if c.geo_id == r0.geo_id and c.year == year]
    comp_out = [{"component": c.component, "value": c.value, "note": c.note} for c in comps]

    citations = [
        {
            "citation_type": "structured_data",
            "source_table": "metrics.csv",
            "geo_id": r0.geo_id,
            "geo_name": r0.geo_name,
            "geo_type": r0.geo_type,
            "metric": "risk_index",
            "year": year,
        },
        {
            "citation_type": "structured_data",
            "source_table": "risk_components.csv",
            "geo_id": r0.geo_id,
            "geo_name": r0.geo_name,
            "year": year,
        },
    ]

    return {
        "ok": True,
        "data": {
            "geo_id": r0.geo_id,
            "geo_name": r0.geo_name,
            "geo_type": r0.geo_type,
            "year": year,
            "risk_index": risk_index,
            "components": comp_out,
            "data_quality_flag": r0.data_quality_flag,
        },
        "citations": citations,
    }


# ---------------------------------------------------------------------------
# Knowledge base retriever
# ---------------------------------------------------------------------------

KB_DIR = Path(__file__).resolve().parent / "kb"
_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def _parse_frontmatter(md_text: str) -> Tuple[Dict[str, Any], str]:
    lines = (md_text or "").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, md_text

    meta: Dict[str, Any] = {}
    i = 1
    while i < len(lines):
        if lines[i].strip() == "---":
            body = "\n".join(lines[i + 1 :])
            return meta, body
        line = lines[i].strip()
        if line and ":" in line:
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip()
            if "," in v:
                meta[k] = [x.strip() for x in v.split(",") if x.strip()]
            else:
                meta[k] = v
        i += 1

    return {}, md_text


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]


def _chunk_markdown(body: str) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n+", body or "") if p.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0
    for p in paras:
        if buf_len + len(p) > 900 and buf:
            chunks.append("\n\n".join(buf).strip())
            buf = []
            buf_len = 0
        buf.append(p)
        buf_len += len(p)
    if buf:
        chunks.append("\n\n".join(buf).strip())
    return chunks


def _kb_strip_inline_markdown(s: str) -> str:
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = s.replace("**", "")
    s = s.replace("`", "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _kb_chunk_to_prose(text: str, *, max_chars: int = 1600) -> str:
    """
    Turn a KB markdown excerpt into plain, speakable sentences (no ## / --- / bullet markup).
    """
    t = (text or "").strip()
    if not t:
        return ""
    parts: List[str] = []

    for block in re.split(r"\n\s*\n+", t):
        lines_in = [
            ln.strip()
            for ln in block.splitlines()
            if ln.strip() and ln.strip() != "---" and not ln.strip().startswith("---")
        ]
        bullets: List[str] = []

        def flush_bullets() -> None:
            nonlocal bullets
            if not bullets:
                return
            clause = "; ".join(b.rstrip(".") for b in bullets if b)
            if clause:
                if not clause[0].isupper():
                    clause = clause[0].upper() + clause[1:] if len(clause) > 1 else clause.upper()
                parts.append(clause + ".")
            bullets = []

        for ln in lines_in:
            if ln.startswith("#"):
                flush_bullets()
                s = _kb_strip_inline_markdown(re.sub(r"^#+\s*", "", ln).strip())
                # Drop short section labels that read oddly as standalone “sentences.”
                if s and len(s) <= 48 and not re.search(r"\bvawa\b|\breauthorization\b|\bfederal\b|\bgrant\b", s, re.I):
                    low = s.lower().rstrip(".")
                    if low in {
                        "summary",
                        "source",
                        "purpose",
                        "overview",
                        "interpretation notes",
                        "related documents",
                        "key updates in 2022",
                        "core areas covered",
                    }:
                        s = ""
                if s:
                    parts.append(s if s.endswith((".", "?", "!")) else s + ".")
            elif re.match(r"^[\*\-]\s+", ln):
                bullets.append(_kb_strip_inline_markdown(re.sub(r"^[\*\-]\s+", "", ln).strip()))
            else:
                flush_bullets()
                if ln:
                    ln2 = _kb_strip_inline_markdown(ln)
                    parts.append(ln2 if ln2.endswith((".", "?", "!")) else ln2 + ".")
        flush_bullets()

    out = " ".join(parts)
    out = re.sub(r"\s+", " ", out).strip()
    out = re.sub(r":\s*\.\s*", ": ", out)
    out = re.sub(r"\s+\.\s+", ". ", out)
    if len(out) > max_chars:
        cut = out[: max_chars - 1]
        out = cut.rsplit(" ", 1)[0].rstrip(",;") + "…"
    return out


def _kb_chunk_para_index(ch: RetrievedChunk) -> int:
    m = re.search(r"::p(\d+)$", ch.chunk_id or "")
    return int(m.group(1)) if m else 999


def _synthesize_docs_answer(message: str, chunks: List[RetrievedChunk]) -> Tuple[str, List[str]]:
    """
    One main-bubble answer in natural language plus optional evidence lines (titles + short quotes).
    """
    if not chunks:
        return "", []
    ql = (message or "").lower()
    ordered = list(chunks)

    if "reauthorization" in ql or ("vawa" in ql and "2022" in ql and any(w in ql for w in ["what did", "what does", "reauthorize", "reauthorization"])):
        policy = [
            c
            for c in ordered
            if "vawa_2022_reauthorization" in str(c.metadata.get("source_file", "")).lower()
        ]
        if policy:
            ordered = sorted(policy, key=_kb_chunk_para_index)

    prose_bits: List[str] = []
    deferred_caveats: List[str] = []
    evidence: List[str] = []
    used_ids: set[str] = set()

    for ch in ordered[:5]:
        if ch.chunk_id in used_ids:
            continue
        raw = ch.text or ""
        lead_raw, main_raw = "", raw.strip()
        if "## Summary" in raw:
            lead_raw, sep, rest = raw.partition("## Summary")
            main_raw = (sep + rest).strip() if sep else raw.strip()

        main_p = _kb_chunk_to_prose(main_raw, max_chars=900) if main_raw else ""
        if lead_raw.strip():
            lead_p = _kb_chunk_to_prose(lead_raw.strip(), max_chars=420)
            if len(lead_p) > 30:
                deferred_caveats.append(lead_p)

        p = main_p
        if len(p) < 40:
            continue
        title = str(ch.metadata.get("title") or ch.metadata.get("source_file") or "Source").strip()
        used_ids.add(ch.chunk_id)
        prose_bits.append(p)
        evidence.append(f"{title} ({ch.chunk_id}): {p[:320]}{'…' if len(p) > 320 else ''}")

    body = " ".join(prose_bits).strip()
    if ("reauthorization" in ql or ("vawa" in ql and "2022" in ql)) and len(body) > 1180:
        cut = body[:1179]
        if "." in cut:
            body = cut.rsplit(".", 1)[0].rstrip() + "."
        else:
            body = cut.rstrip() + "…"
    if deferred_caveats:
        tail = " ".join(deferred_caveats).strip()
        body = (body + " " + tail).strip() if body else tail

    if len(body) > 1700:
        body = body[:1699].rsplit(" ", 1)[0].rstrip(",;") + "…"

    if not body:
        return "", evidence

    if "reauthorization" in ql or ("vawa" in ql and "2022" in ql):
        lead = "According to the project’s VAWA 2022 overview document:"
    else:
        lead = "In plain language, based on the closest matching knowledge-base excerpts:"

    return f"{lead} {body}", evidence


def retrieve(query: str, top_k: int = 4) -> List[RetrievedChunk]:
    """
    Retrieve KB chunks for a query.

    Default: keyword-overlap ranking.
    If OPENAI_API_KEY is set: semantic ranking using embeddings (more conversational).
    """
    query = (query or "").strip()
    if not query:
        return []

    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if api_key:
        try:
            return _retrieve_semantic(query, top_k=top_k, api_key=api_key)
        except Exception:
            # Fail closed to keyword retrieval
            pass
    return _retrieve_keyword(query, top_k=top_k)


def _kb_all_chunks() -> List[RetrievedChunk]:
    chunks_out: List[RetrievedChunk] = []
    for path in sorted(KB_DIR.glob("*.md")):
        md_text = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(md_text)
        chunks = _chunk_markdown(body)
        for idx, chunk_text in enumerate(chunks):
            chunks_out.append(
                RetrievedChunk(
                    chunk_id=f"{path.stem}::p{idx+1}",
                    text=chunk_text,
                    score=0.0,
                    metadata={**meta, "source_file": path.name},
                )
            )
    return chunks_out


def _retrieve_keyword(query: str, top_k: int = 4) -> List[RetrievedChunk]:
    q_tokens = set(_tokenize(query))
    if not q_tokens:
        return []

    ql = (query or "").lower()
    results: List[RetrievedChunk] = []
    for ch in _kb_all_chunks():
        c_tokens = set(_tokenize(ch.text))
        overlap = q_tokens.intersection(c_tokens)
        if not overlap:
            continue
        bonus = 0.0
        tags = ch.metadata.get("metric_tags", [])
        if isinstance(tags, list):
            for t in tags:
                if str(t).lower() in q_tokens:
                    bonus += 0.25
        sf = str(ch.metadata.get("source_file", "")).lower()
        # Prefer the dedicated policy doc when users ask about the 2022 reauthorization (limits_scope also mentions VAWA).
        if "reauthorization" in ql and "vawa_2022_reauthorization" in sf:
            bonus += 12.0
        if "reauthorization" in ql and "limitations_scope" in sf:
            bonus -= 5.0
        score = float(len(overlap)) + bonus
        results.append(RetrievedChunk(chunk_id=ch.chunk_id, text=ch.text, score=score, metadata=ch.metadata))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[: max(0, top_k)]


def _embed_texts_openai(texts: List[str], *, api_key: str, model: str = "text-embedding-3-small") -> List[List[float]]:
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def _retrieve_semantic(query: str, top_k: int, api_key: str) -> List[RetrievedChunk]:
    import numpy as np  # type: ignore

    cache_key = "kb_semantic_index_v1"
    index = _CACHE.get(cache_key)
    if not index:
        chunks = _kb_all_chunks()
        embs = _embed_texts_openai([c.text for c in chunks], api_key=api_key)
        mat = np.array(embs, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms
        index = {"chunks": chunks, "mat": mat}
        _CACHE[cache_key] = index

    chunks = index["chunks"]
    mat = index["mat"]

    q_emb = _embed_texts_openai([query], api_key=api_key)[0]
    q = np.array(q_emb, dtype=np.float32)
    qn = np.linalg.norm(q)
    if not np.isfinite(qn) or qn == 0:
        return _retrieve_keyword(query, top_k=top_k)
    q = q / qn

    sims = mat @ q
    k = max(0, int(top_k))
    if k == 0 or len(chunks) == 0:
        return []
    idxs = np.argsort(-sims)[:k]

    out: List[RetrievedChunk] = []
    for i in idxs:
        ch = chunks[int(i)]
        out.append(RetrievedChunk(chunk_id=ch.chunk_id, text=ch.text, score=float(sims[int(i)]), metadata=ch.metadata))
    return out


# ---------------------------------------------------------------------------
# Intent
# ---------------------------------------------------------------------------

Intent = Literal["data_only", "docs_only", "data_and_docs", "resources"]

_METRIC_HINTS = {
    "dv_rate",
    "sexual_assault_rate",
    "firearm_share",
    "dating_partner_share",
    "minority_victim_share",
    "native_american_victim_share",
    "reporting_proxy",
    "risk_index",
    "risk profile",
    "rank",
    "highest",
    "lowest",
    "compare",
    "trend",
    "increase",
    "decrease",
    "timeseries",
}

_DOCS_HINTS = {
    "vawa",
    "methodology",
    "definition",
    "limitations",
    "caveat",
    "nibrs",
    "reporting",
    "interpret",
    "model",
    "regression",
    "causal",
}


def classify_intent(message: str) -> Intent:
    text = (message or "").lower()
    if any(k in text for k in ["find a shelter", "shelter near", "shelters near", "resources near", "domestic violence shelter", "dv shelter", "mental health near", "counseling near", "therapy near", "hotline", "crisis line"]):
        return "resources"
    has_metric = any(h in text for h in _METRIC_HINTS)
    has_docs = any(h in text for h in _DOCS_HINTS)

    if re.search(r"\bwhat does vawa\b|\bwhat is vawa\b|\bwhat changed\b", text):
        return "docs_only"

    if has_metric and has_docs:
        return "data_and_docs"
    if has_metric:
        return "data_only"
    if has_docs:
        return "docs_only"

    return "docs_only"


def _arcgis_dashboard_url() -> str:
    u = (os.environ.get("ARCGIS_DASHBOARD_URL") or "").strip()
    return u or DEFAULT_ARCGIS_DASHBOARD_URL


def _arcgis_embed_url_for_webmap(webmap_id: str) -> str:
    # ArcGIS "Embed" app for web maps (simple iframe-friendly viewer).
    return f"https://www.arcgis.com/apps/Embed/index.html?webmap={webmap_id}"


def _arcgis_pick_tab(message: str, metric: Optional[str]) -> str:
    t = (message or "").lower()
    m = (metric or "").strip().lower()

    if "shelter" in t:
        return "Shelter Locations"
    if "rural" in t:
        return "Rural Locations"
    if "college" in t or "campus" in t:
        return "College"
    if "tribal" in t:
        return "Tribal"
    if "race" in t or "minority" in t or "native" in t:
        return "Race"
    if "violent" in t:
        return "Violent Crimes"

    if m == "sexual_assault_rate" or "sexual assault" in t:
        return "Sexual Assault"
    if m == "dv_rate" or "domestic violence" in t:
        return "Domestic Violence"
    if m == "firearm_share" or "firearm" in t or "gun" in t:
        return "Firearm"

    return "Master"


def _should_attach_arcgis_map(
    message: str,
    metric: Optional[str],
    geo_names: List[str],
    tools_used: List[Dict[str, Any]],
) -> bool:
    t = (message or "").lower()
    if re.search(r"\bmap\b|\bgis\b|\bgeographic\b|\bchoropleth\b", t):
        return True
    tool_names = {x.get("tool") for x in tools_used if isinstance(x, dict)}
    geo_tools = {
        "compare_geos",
        "rank_geos",
        "get_metric_timeseries",
        "get_risk_profile",
        "rank_states_by_policy_change",
        "get_policy_variable_summary",
    }
    if tool_names & geo_tools:
        return True
    if len(geo_names) >= 2 and metric is not None and "compare" in t:
        return True
    return False


def _build_map_embed(
    message: str,
    metric: Optional[str],
    geo_names: List[str],
    tools_used: List[Dict[str, Any]],
) -> Optional[MapEmbedPayload]:
    if not _should_attach_arcgis_map(message, metric, geo_names, tools_used):
        return None
    dashboard_url = _arcgis_dashboard_url()
    tab = _arcgis_pick_tab(message, metric)
    embed_url = dashboard_url
    states = geo_names[:8]
    m_label = _METRIC_LABELS.get(metric or "", None) or (metric.replace("_", " ") if metric else None)
    bits: List[str] = []
    bits.append(f"In the dashboard, open the “{tab}” tab to see the relevant layer.")
    if states:
        bits.append(f"Your question references: {', '.join(states)}.")
    if m_label and tab == "Master":
        bits.append(f"Suggested layer: {m_label}.")
    return MapEmbedPayload(
        show=True,
        title="Related ArcGIS dashboard",
        embed_url=embed_url,
        open_url=dashboard_url,
        caption=" ".join(bits),
        states=states,
        metric=metric,
        metric_label=m_label,
    )


# ---------------------------------------------------------------------------
# Chat entrypoint
# ---------------------------------------------------------------------------


def _doc_citation_from_chunk(chunk: RetrievedChunk) -> Dict[str, Any]:
    md = chunk.metadata
    return {
        "citation_type": "knowledge_base",
        "citation_id": md.get("citation_id", chunk.chunk_id),
        "title": md.get("title", md.get("source_file", "knowledge doc")),
        "doc_type": md.get("doc_type", ""),
        "source_file": md.get("source_file", ""),
        "chunk_id": chunk.chunk_id,
        "years_covered": md.get("years_covered", ""),
        "geo_level": md.get("geo_level", ""),
        "metric_tags": md.get("metric_tags", []),
    }


def answer_chat(req: ChatRequest) -> ChatResponse:
    message = (req.message or "").strip()
    intent = classify_intent(message)

    tools_used: List[Dict[str, Any]] = []
    docs_retrieved: List[Dict[str, Any]] = []
    citations: List[Dict[str, Any]] = []
    llm_debug: Dict[str, Any] = {"enabled": False}

    metric = detect_metric(message)
    years = extract_years(message)
    geo_names = detect_geo_names(message)

    evidence_lines: List[str] = []
    caveats: List[str] = []
    interpretation = ""
    direct_answer = ""
    skip_default_interpretation = False

    descriptive_caveat = "These are descriptive patterns from observed data and do not, by themselves, establish causation."
    policy_caveat = "Pre/post-2022 comparisons are descriptive; changes may reflect reporting/coverage shifts and do not, by themselves, establish policy effects."

    policy_var = detect_policy_variable(message)
    asks_policy_change = any(
        k in message.lower()
        for k in [
            "post_2022",
            "post-2022",
            "pre_2022",
            "pre-2022",
            "percent change",
            "% change",
            "change after 2022",
            "after 2022",
            "before 2022",
            "pre vs",
            "pre/post",
        ]
    )

    # -------------------------------------------------------------------
    # Victim resources intent (location-based)
    # -------------------------------------------------------------------
    if intent == "resources":
        loc_phrase = _extract_location_phrase(message)
        lat0, lon0 = _extract_lat_lon(message)
        if not loc_phrase and (lat0 is None or lon0 is None):
            direct_answer = "Tell me a location to search near (e.g., “near Sacramento, CA” or “32.7157, -117.1611”)."
            citations.append({"citation_type": "structured_data", "source_table": RESOURCES_FILE.name})
            return ChatResponse(
                answer={
                    "direct_answer": direct_answer,
                    "evidence": evidence_lines,
                    "interpretation": "",
                    "caveats": ["If you are in immediate danger, call 911."],
                    "citations": citations,
                    "map_embed": None,
                },
                debug={"intent": intent, "tools_used": tools_used, "docs_retrieved": docs_retrieved, "llm": llm_debug},
            )

        # Heuristics: “shelter” implies housing_shelter unless user asks broader.
        cats: List[str] = []
        tl = message.lower()
        query_hint = ""
        if "shelter" in tl:
            cats.append("housing_shelter")
            query_hint = "shelter"
        if any(x in tl for x in ["therapy", "counsel", "mental health", "psychiat", "988"]):
            cats.append("mental_health")
            query_hint = query_hint or "mental health"
        if any(x in tl for x in ["legal", "lawyer", "restraining order", "protective order"]):
            cats.append("legal_aid")
            query_hint = query_hint or "legal aid"
        if any(x in tl for x in ["hotline", "crisis line"]):
            cats.append("hotline")
            query_hint = query_hint or "hotline"
        if any(x in tl for x in ["domestic violence", "intimate partner"]):
            cats.append("domestic_violence")
            query_hint = query_hint or "domestic violence"
        if any(x in tl for x in ["sexual assault", "rape"]):
            cats.append("sexual_assault")
            query_hint = query_hint or "sexual assault"
        cats = sorted({c for c in cats if c})

        tool_out = find_victim_resources(
            query=query_hint or "",
            location=loc_phrase,
            latitude=lat0,
            longitude=lon0,
            radius_miles=25.0,
            limit=8,
            categories=cats,
        )
        tools_used.append(
            {
                "tool": "find_victim_resources",
                "args": {"query": message, "location": loc_phrase, "latitude": lat0, "longitude": lon0, "radius_miles": 25.0, "limit": 8, "categories": cats},
                "ok": tool_out.get("ok"),
            }
        )

        if not tool_out.get("ok"):
            direct_answer = f"I couldn’t search resources right now. ({tool_out.get('error')})"
            citations.append({"citation_type": "structured_data", "source_table": RESOURCES_FILE.name})
        else:
            d = tool_out["data"]
            results = d.get("results", [])
            citations.extend(tool_out.get("citations", []))
            resolved = (d.get("resolved_location") or loc_phrase or "").strip()
            if results:
                lines: List[str] = []
                lines.append(f"Here are {len(results)} resource(s) near {resolved or 'your location'}:")
                for r in results:
                    bits = [r.get("name", "")]
                    if r.get("city") or r.get("state"):
                        bits.append(", ".join([x for x in [r.get("city"), r.get("state")] if x]))
                    if r.get("distance_miles") is not None:
                        bits.append(f"{r['distance_miles']} miles")
                    if r.get("phone"):
                        bits.append(f"phone {r['phone']}")
                    if r.get("website"):
                        bits.append(f"website {r['website']}")
                    bullet = "- " + " — ".join([b for b in bits if b])
                    evidence_lines.append(bullet)
                    lines.append(bullet)
                direct_answer = "\n".join(lines).strip()
            else:
                direct_answer = f"I didn’t find any entries within {d.get('radius_miles')} miles of {resolved or 'that location'} in the current resources dataset."

        caveats.append("If you are in immediate danger, call 911. If you can’t safely call, consider texting a trusted person or using a safe device.")
        return ChatResponse(
            answer={
                "direct_answer": direct_answer,
                "evidence": evidence_lines,
                "interpretation": "",
                "caveats": caveats,
                "citations": citations,
                "map_embed": None,
            },
            debug={"intent": intent, "tools_used": tools_used, "docs_retrieved": docs_retrieved, "llm": llm_debug},
        )

    if intent in {"data_only", "data_and_docs"}:
        policy_handled = False
        if policy_var and asks_policy_change:
            if any(k in message.lower() for k in ["highest", "top", "rank", "most"]) and "state" in message.lower():
                dirn: Literal["increase", "decrease"] = "increase"
                if any(k in message.lower() for k in ["decrease", "decline", "dropped", "lowest"]):
                    dirn = "decrease"

                # Baseline filter for counts to avoid extreme percent-change blowups.
                vmeta = get_policy_variable_summary(policy_var)
                vtype = ""
                if vmeta.get("ok"):
                    vtype = str(vmeta.get("data", {}).get("variable_type", "")).strip().lower()
                min_pre = 500.0 if vtype == "count" else 0.01

                tool_out = rank_states_by_policy_change(policy_var, dirn, top_n=5, min_pre_value=min_pre)
                tools_used.append({"tool": "rank_states_by_policy_change", "args": {"variable": policy_var, "direction": dirn, "top_n": 5, "min_pre_value": min_pre}, "ok": tool_out.get("ok")})
                if tool_out.get("ok"):
                    ranked = tool_out["data"]["ranked"]
                    direct_answer = (
                        f"Top states by percent change in {policy_var} ({dirn}, pre vs post 2022). "
                        f"(Filtered to stable baselines: pre_2022 ≥ {min_pre:g}): "
                        + ", ".join([f"{r['state']} ({r['percent_change']:.2%})" for r in ranked])
                    )
                    evidence_lines.append(f"Variable: {policy_var} (pre_2022 vs post_2022_avg).")
                    for r in ranked:
                        evidence_lines.append(
                            f"- #{r['rank']} {r['state']}: percent_change={r['percent_change']:.2%}, absolute_change={r.get('absolute_change')}, pre_2022={r.get('pre_2022')}, post_2022_avg={r.get('post_2022_avg')}"
                        )
                    citations.append(tool_out["citation"])
                    caveats.append(policy_caveat)
                else:
                    direct_answer = f"I don’t have enough policy-summary data to rank that. ({tool_out.get('error')})"
                policy_handled = True
            else:
                tool_out = get_policy_variable_summary(policy_var)
                tools_used.append({"tool": "get_policy_variable_summary", "args": {"variable": policy_var}, "ok": tool_out.get("ok")})
                if tool_out.get("ok"):
                    d = tool_out["data"]
                    mpc = d.get("mean_percent_change")
                    medpc = d.get("median_percent_change")
                    inc = d.get("states_increased")
                    dec = d.get("states_decreased")
                    direct_answer = f"Across states, {policy_var} changed post-2022 vs pre-2022 (summary)."
                    if mpc is not None:
                        evidence_lines.append(f"Mean percent change: {float(mpc):.2%}")
                    if medpc is not None:
                        evidence_lines.append(f"Median percent change: {float(medpc):.2%}")
                    if inc is not None and dec is not None:
                        evidence_lines.append(f"States increased: {inc}, decreased: {dec}")
                    citations.append(tool_out["citation"])
                    caveats.append(policy_caveat)
                else:
                    direct_answer = f"I don’t have a policy-summary entry for that variable yet. ({tool_out.get('error')})"
                policy_handled = True

        if not policy_handled and ("risk profile" in message.lower() or (metric == "risk_index" and "profile" in message.lower())):
            if not geo_names or not years:
                direct_answer = "I need a geography and a year to generate a risk profile (e.g., “New Mexico in 2024”)."
            else:
                geo = resolve_geo(geo_names[0])
                yr = years[-1]
                tool_out = get_risk_profile(geo or {}, yr)
                tools_used.append({"tool": "get_risk_profile", "args": {"geo": geo, "year": yr}, "ok": tool_out.get("ok")})
                if tool_out.get("ok"):
                    data = tool_out["data"]
                    direct_answer = (
                        f"Risk profile for {data['geo_name']} ({yr}): risk_index = {data['risk_index']:.2f}."
                        + _metric_explain_sentence("risk_index")
                    )
                    evidence_lines.append(f"risk_index ({yr}): {data['risk_index']:.2f} (data_quality_flag={data['data_quality_flag']})")
                    if data.get("components"):
                        evidence_lines.append("Component breakdown:")
                        for c in data["components"]:
                            evidence_lines.append(f"- {c['component']}: {c['value']:.2f} ({c.get('note','')})".strip())
                    citations.extend(tool_out.get("citations", []))
                    caveats.append(descriptive_caveat)
                else:
                    direct_answer = f"I don’t have enough structured data to compute that risk profile. ({tool_out.get('error')})"

        elif not policy_handled and any(k in message.lower() for k in ["highest", "top", "rank"]) and ("state" in message.lower() or "states" in message.lower()):
            if metric is None:
                direct_answer = "Which metric should I rank (e.g., dv_rate, firearm_share)?"
            else:
                yr = years[-1] if years else 2024
                tool_out = rank_geos(metric=metric, year=yr, geo_level="state", top_n=5, sort_direction="desc")
                tools_used.append({"tool": "rank_geos", "args": {"metric": metric, "year": yr, "geo_level": "state", "top_n": 5, "sort_direction": "desc"}, "ok": tool_out.get("ok")})
                if tool_out.get("ok"):
                    ranked = tool_out["data"]["ranked"]
                    mname = _metric_display_name(metric)
                    direct_answer = (
                        f"Top states for {mname} in {yr}: "
                        + ", ".join([f"{r['geo_name']} ({r['value']:.2f})" for r in ranked])
                        + _metric_explain_sentence(metric)
                    )
                    evidence_lines.append(f"Ranking: {metric} in {yr} (top {len(ranked)} states).")
                    for r in ranked:
                        evidence_lines.append(f"- #{r['rank']} {r['geo_name']}: {r['value']:.2f} (flag={r['data_quality_flag']})")
                    citations.append(tool_out["citation"])
                    caveats.append(descriptive_caveat)
                else:
                    direct_answer = f"I don’t have enough structured data to rank that. ({tool_out.get('error')})"

        elif not policy_handled and "compare" in message.lower() and len(geo_names) >= 2 and metric is not None and len(years) >= 1:
            geo_a = resolve_geo(geo_names[0])
            geo_b = resolve_geo(geo_names[1])
            start_year = str(min(years))
            end_year = str(max(years))
            tool_out = compare_geos(geo_a or {}, geo_b or {}, metric, start_year, end_year)
            tools_used.append({"tool": "compare_geos", "args": {"geo_a": geo_a, "geo_b": geo_b, "metric": metric, "start_period": start_year, "end_period": end_year}, "ok": tool_out.get("ok")})
            if tool_out.get("ok"):
                d = tool_out["data"]
                a = d["geo_a"]
                b = d["geo_b"]
                a_disp = round(float(a["avg_value"]), 2)
                b_disp = round(float(b["avg_value"]), 2)
                diff_disp = round(a_disp - b_disp, 2)
                mname = _metric_display_name(metric)
                direct_answer = (
                    f"From {d['start_year']}–{d['end_year']}, {a['geo_name']} averaged {a_disp:.2f} for {mname}, "
                    f"vs {b['geo_name']} at {b_disp:.2f} (difference {diff_disp:.2f})."
                    + _metric_explain_sentence(metric)
                )
                evidence_lines.append(f"{a['geo_name']} average ({d['start_year']}–{d['end_year']}): {float(a['avg_value']):.4f}")
                evidence_lines.append(f"{b['geo_name']} average ({d['start_year']}–{d['end_year']}): {float(b['avg_value']):.4f}")
                citations.append(tool_out["citation"])
                caveats.append(descriptive_caveat)
            else:
                direct_answer = f"I don’t have enough structured data to compare those geographies. ({tool_out.get('error')})"

        elif not policy_handled and metric is not None and geo_names:
            geo = resolve_geo(geo_names[0])
            tool_out = get_metric_timeseries(geo or {}, metric, frequency="year")
            tools_used.append({"tool": "get_metric_timeseries", "args": {"geo": geo, "metric": metric, "frequency": "year"}, "ok": tool_out.get("ok")})
            if tool_out.get("ok"):
                pts = tool_out["data"]["points"]
                mname = _metric_display_name(metric)
                # Put the key numbers in the main answer (not only in evidence_lines).
                series: Dict[int, float] = {}
                for p in pts:
                    try:
                        yr = int(str(p.get("period", "")).strip())
                    except ValueError:
                        continue
                    val = p.get("value")
                    if val is None:
                        continue
                    series[yr] = float(val)

                geo_nm = tool_out["data"]["geo"]["geo_name"]
                if series:
                    yrs_sorted = sorted(series.keys())
                    parts = [f"{y}: {series[y]:.2f}" for y in yrs_sorted]
                    lo_y, hi_y = min(series, key=lambda y: series[y]), max(series, key=lambda y: series[y])
                    summary = (
                        f"{mname} over time for {geo_nm}: "
                        + "; ".join(parts)
                        + f". Lowest in {lo_y} ({series[lo_y]:.2f}); highest in {hi_y} ({series[hi_y]:.2f})."
                    )
                else:
                    summary = f"{mname} over time for {geo_nm}: no numeric points were available for the requested years."

                direct_answer = summary + _metric_explain_sentence(metric)
                evidence_lines.extend([f"- {p['period']}: {p['value']:.2f} (flag={p['data_quality_flag']})" for p in pts])
                citations.append(tool_out["citation"])
                caveats.append(descriptive_caveat)
            else:
                direct_answer = f"I don’t have enough structured data for that request. ({tool_out.get('error')})"

        else:
            if not policy_handled:
                direct_answer = "I couldn’t identify a supported metric and geography to answer with numbers. Try including a metric (e.g., dv_rate) and a place (e.g., California)."

    if intent in {"docs_only", "data_and_docs"}:
        chunks = retrieve(message, top_k=4)
        for ch in chunks:
            docs_retrieved.append(
                {
                    "chunk_id": ch.chunk_id,
                    "score": ch.score,
                    "title": ch.metadata.get("title", ""),
                    "citation_id": ch.metadata.get("citation_id", ""),
                }
            )
            citations.append(_doc_citation_from_chunk(ch))

        if chunks:
            da_stripped = (direct_answer or "").strip()
            kb_is_primary = da_stripped == "" or da_stripped.startswith("I couldn’t identify")
            if kb_is_primary:
                synthesized, kb_evidence = _synthesize_docs_answer(message, chunks)
                if synthesized:
                    direct_answer = synthesized
                    evidence_lines.extend(kb_evidence)
                    interpretation = ""
                    skip_default_interpretation = True
                else:
                    direct_answer = "I don’t have enough on-point knowledge-base excerpts to summarize that yet."
                    interpretation = ""
                    skip_default_interpretation = True
            else:
                # Already answered with structured data; keep the main bubble clean (no raw KB markdown).
                interpretation = ""
        else:
            if intent == "docs_only":
                direct_answer = "I don’t have enough knowledge-base content to answer that yet."
            caveats.append("Knowledge base retrieval returned no matching documents for this query.")

    if any(w in message.lower() for w in ["cause", "caused", "because", "led to", "impact", "effect"]):
        caveats.append("This bot can describe trends/associations in the available data, but it does not establish causal effects.")

    if not citations:
        citations.append(
            {
                "citation_type": "system",
                "citation_id": "NO-SOURCE",
                "title": "No matching structured rows or KB chunks found",
            }
        )

    if interpretation == "" and not skip_default_interpretation:
        interpretation = "Interpretation is limited. If you want, ask for caveats/definitions and I’ll pull from the methodology documents."

    # ---- Optional LLM writer step (minimal integration) ----
    # If OPENAI_API_KEY is not set, we keep deterministic behavior.
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if api_key:
        try:
            # Imported lazily so the backend still runs without the dependency
            # if the user doesn't install it.
            from openai import OpenAI  # type: ignore

            client = OpenAI(api_key=api_key)
            llm_debug = {"enabled": True, "model": "gpt-4o-mini", "used": False}

            # Keep prompt small: we provide the user message, the computed evidence,
            # and short excerpts from retrieved chunks.
            excerpt_chunks: List[Dict[str, Any]] = []
            if "chunks" in locals() and isinstance(locals().get("chunks"), list):
                for ch in locals()["chunks"][:4]:
                    txt = (ch.text or "").strip().replace("\n", " ")
                    if len(txt) > 500:
                        txt = txt[:500].rstrip() + "…"
                    excerpt_chunks.append(
                        {
                            "chunk_id": ch.chunk_id,
                            "citation_id": ch.metadata.get("citation_id", ch.chunk_id),
                            "title": ch.metadata.get("title", ch.metadata.get("source_file", "")),
                            "text_excerpt": txt,
                        }
                    )

            writer_payload = {
                "user_message": message,
                "intent": intent,
                "direct_answer_so_far": direct_answer,
                "evidence_lines": evidence_lines,
                "caveats": caveats,
                "retrieved_chunks": excerpt_chunks,
                "citations": citations,
                "constraints": [
                    "Do not invent numbers. If you include any numeric value, it must appear verbatim in evidence_lines or retrieved_chunks.",
                    "Do not claim causation. Use associational language unless the retrieved text explicitly states causal identification (rare).",
                    "If you reference a knowledge-base statement, include its citation_id in parentheses.",
                    "Keep it concise (2–6 sentences).",
                ],
            }

            system = (
                "You are a careful analyst writing a short, grounded response for a VAWA insights system. "
                "You must follow the constraints exactly."
            )

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": "Write an improved direct answer + interpretation as JSON with keys "
                        '"direct_answer" and "interpretation" only.\n\nINPUT:\n'
                        + json.dumps(writer_payload, ensure_ascii=False),
                    },
                ],
            )

            content = (resp.choices[0].message.content or "").strip()
            # Robust-ish JSON extraction (model should return pure JSON, but guard anyway)
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                content = content[start : end + 1]
            out = json.loads(content)
            if isinstance(out, dict):
                da = out.get("direct_answer")
                interp = out.get("interpretation")
                if isinstance(da, str) and da.strip():
                    direct_answer = da.strip()
                    llm_debug["used"] = True
                if isinstance(interp, str) and interp.strip():
                    interpretation = interp.strip()
                    llm_debug["used"] = True
        except Exception:
            # Fail closed: keep deterministic output if the LLM step fails.
            llm_debug = {"enabled": True, "model": "gpt-4o-mini", "used": False, "error": "llm_call_failed"}

    map_embed = _build_map_embed(message, metric, geo_names, tools_used)

    return ChatResponse(
        answer={
            "direct_answer": direct_answer,
            "evidence": evidence_lines,
            "interpretation": interpretation,
            "caveats": caveats,
            "citations": citations,
            "map_embed": map_embed,
        },
        debug={
            "intent": intent,
            "tools_used": tools_used,
            "docs_retrieved": docs_retrieved,
            "llm": llm_debug,
        },
    )
