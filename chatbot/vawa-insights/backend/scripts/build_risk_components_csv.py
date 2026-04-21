"""
Build `backend/app/data/risk_components.csv` and populate `risk_index` in `backend/app/data/metrics.csv`.

No dedicated risk-component export exists under EDA/output in this repo, so we derive components from the
project's state-year aggregated dataset:
- `<repo_root>/aggregated_crime_and_census_data.csv`

Risk components are computed as within-year z-scores across states. The `risk_index` is the mean of
available component z-scores for that state-year.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple


STATE_TO_ABBR: Dict[str, str] = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}


def _f(x: str) -> Optional[float]:
    x = (x or "").strip()
    if x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def _safe_div(n: Optional[float], d: Optional[float]) -> Optional[float]:
    if n is None or d is None or d == 0:
        return None
    return n / d


def _z(x: float, mu: float, sd: float) -> float:
    if sd == 0:
        return 0.0
    return (x - mu) / sd


def _component_specs() -> List[Tuple[str, str]]:
    """
    Returns (component_name, raw_units_hint) pairs in the order we report them.
    """
    return [
        ("dv_rate", "per 100k female population"),
        ("sexual_assault_rate", "per 100k female population"),
        ("firearm_share", "proportion"),
        ("fem_poverty_rate", "proportion"),
        ("unemployment_rate", "proportion"),
        ("gun_deaths_per_100k", "per 100k population"),
    ]


def main() -> None:
    # scripts/ -> backend/ -> vawa-insights/ -> chatbot/ -> <repo_root>
    repo_root = Path(__file__).resolve().parents[4]
    src = repo_root / "aggregated_crime_and_census_data.csv"
    if not src.exists():
        raise FileNotFoundError(f"Missing source file: {src}")

    metrics_path = repo_root / "chatbot" / "vawa-insights" / "backend" / "app" / "data" / "metrics.csv"
    risk_path = repo_root / "chatbot" / "vawa-insights" / "backend" / "app" / "data" / "risk_components.csv"
    risk_path.parent.mkdir(parents=True, exist_ok=True)

    # Load aggregated rows.
    with src.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Compute raw component values per (geo_id, year).
    raw_by_key: Dict[Tuple[str, int], Dict[str, float]] = {}
    geo_name_by_id: Dict[str, str] = {}

    for r in rows:
        state_name = (r.get("state_name") or r.get("state") or "").strip()
        if not state_name:
            continue
        abbr = STATE_TO_ABBR.get(state_name, "")
        geo_id = f"state:{abbr or state_name}"
        geo_name_by_id[geo_id] = state_name

        year = int(float(r.get("data_year") or r.get("year") or 0))
        if year <= 0:
            continue

        fem_pop = _f(r.get("fem_pop", ""))
        total_incidents = _f(r.get("total_incidents", ""))
        dv_total = _f(r.get("dv_total", ""))
        sex_assaults = _f(r.get("sex_assaults", ""))
        involving_firearm = _f(r.get("involving_firearm", ""))

        dv_rate = (dv_total / fem_pop) * 100000.0 if (dv_total is not None and fem_pop and fem_pop > 0) else None
        sa_rate = (sex_assaults / fem_pop) * 100000.0 if (sex_assaults is not None and fem_pop and fem_pop > 0) else None
        firearm_share = _safe_div(involving_firearm, total_incidents)

        fem_poverty_rate = _f(r.get("fem_poverty_rate", ""))
        unemployment_rate = _f(r.get("unemployment_rate", ""))
        gun_deaths_per_100k = _f(r.get("gun_deaths_per_100k", ""))

        vals: Dict[str, float] = {}
        if dv_rate is not None:
            vals["dv_rate"] = float(dv_rate)
        if sa_rate is not None:
            vals["sexual_assault_rate"] = float(sa_rate)
        if firearm_share is not None:
            vals["firearm_share"] = float(firearm_share)
        if fem_poverty_rate is not None:
            vals["fem_poverty_rate"] = float(fem_poverty_rate)
        if unemployment_rate is not None:
            vals["unemployment_rate"] = float(unemployment_rate)
        if gun_deaths_per_100k is not None:
            vals["gun_deaths_per_100k"] = float(gun_deaths_per_100k)

        if vals:
            raw_by_key[(geo_id, year)] = vals

    # Compute within-year stats.
    years = sorted({yr for _, yr in raw_by_key.keys()})
    components = [c for c, _ in _component_specs()]
    stats: Dict[Tuple[int, str], Tuple[float, float]] = {}

    for yr in years:
        for comp in components:
            xs = [v[comp] for (gid, y), v in raw_by_key.items() if y == yr and comp in v]
            if not xs:
                continue
            mu = mean(xs)
            sd = pstdev(xs)  # population stdev (stable, deterministic)
            if not math.isfinite(mu) or not math.isfinite(sd):
                continue
            stats[(yr, comp)] = (mu, sd)

    # Write risk components as z-scores + notes with raw values/units.
    risk_fieldnames = ["geo_id", "geo_name", "year", "component", "value", "note"]
    out_rows: List[Dict[str, str]] = []
    risk_index_by_key: Dict[Tuple[str, int], float] = {}

    units = {c: u for c, u in _component_specs()}

    for (geo_id, yr), vals in raw_by_key.items():
        z_scores: List[float] = []
        for comp in components:
            if comp not in vals:
                continue
            if (yr, comp) not in stats:
                continue
            mu, sd = stats[(yr, comp)]
            zc = _z(vals[comp], mu, sd)
            z_scores.append(zc)
            out_rows.append(
                {
                    "geo_id": geo_id,
                    "geo_name": geo_name_by_id.get(geo_id, ""),
                    "year": str(yr),
                    "component": comp,
                    "value": f"{zc:.10g}",
                    "note": f"z-score within {yr}; raw={vals[comp]:.6g} ({units.get(comp,'')})",
                }
            )

        if z_scores:
            risk_index_by_key[(geo_id, yr)] = float(mean(z_scores))

    with risk_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=risk_fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    # Update metrics.csv risk_index column (join by geo_id + year).
    if metrics_path.exists():
        with metrics_path.open("r", newline="", encoding="utf-8") as f:
            mreader = csv.DictReader(f)
            mfields = list(mreader.fieldnames or [])
            mrows = list(mreader)

        if "risk_index" not in mfields:
            mfields.append("risk_index")

        for r in mrows:
            geo_id = (r.get("geo_id") or "").strip()
            try:
                yr = int(r.get("year") or 0)
            except ValueError:
                yr = 0
            ri = risk_index_by_key.get((geo_id, yr))
            r["risk_index"] = "" if ri is None else f"{ri:.10g}"

        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            mw = csv.DictWriter(f, fieldnames=mfields)
            mw.writeheader()
            mw.writerows(mrows)

    print(f"Wrote {risk_path}")
    print(f"Updated risk_index in {metrics_path}")


if __name__ == "__main__":
    main()

