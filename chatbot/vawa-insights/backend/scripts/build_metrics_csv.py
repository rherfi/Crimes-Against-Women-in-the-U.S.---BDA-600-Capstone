"""
Build `backend/app/data/metrics.csv` from project EDA outputs.

Source of truth (preferred):
- `<repo_root>/aggregated_crime_and_census_data.csv`

Output schema matches what the chatbot expects in `app/logic.py`.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional


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


def main() -> None:
    # scripts/ -> backend/ -> vawa-insights/ -> chatbot/ -> <repo_root>
    repo_root = Path(__file__).resolve().parents[4]
    src = repo_root / "aggregated_crime_and_census_data.csv"
    if not src.exists():
        raise FileNotFoundError(f"Missing source file: {src}")

    out = repo_root / "chatbot" / "vawa-insights" / "backend" / "app" / "data" / "metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    with src.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    fieldnames = [
        "geo_id",
        "geo_name",
        "geo_type",
        "state",
        "county",
        "tribal_area",
        "year",
        "quarter",
        "month",
        "dv_rate",
        "sexual_assault_rate",
        "firearm_share",
        "dating_partner_share",
        "minority_victim_share",
        "native_american_victim_share",
        "reporting_proxy",
        "risk_index",
        "data_quality_flag",
    ]

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for r in rows:
            state_name = (r.get("state_name") or r.get("state") or "").strip()
            if not state_name:
                continue
            abbr = STATE_TO_ABBR.get(state_name, "")

            year = int(float(r.get("data_year") or r.get("year") or 0))
            fem_pop = _f(r.get("fem_pop", ""))

            total_incidents = _f(r.get("total_incidents", ""))
            dv_total = _f(r.get("dv_total", ""))
            sex_assaults = _f(r.get("sex_assaults", ""))
            involving_firearm = _f(r.get("involving_firearm", ""))
            dating_partner = _f(r.get("victim_offender_nonmarried_partner", ""))

            white_v = _f(r.get("white_victims", ""))
            black_v = _f(r.get("black_victims", ""))
            nat_amer_v = _f(r.get("nat_amer_victims", ""))
            asian_v = _f(r.get("asian_victims", ""))
            hispanic_v = _f(r.get("hispanic_victims", ""))
            other_v = _f(r.get("other_race_victims", ""))

            total_victims = None
            if None not in (white_v, black_v, nat_amer_v, asian_v, hispanic_v, other_v):
                total_victims = (white_v or 0) + (black_v or 0) + (nat_amer_v or 0) + (asian_v or 0) + (hispanic_v or 0) + (other_v or 0)
            minority_victims = None
            if total_victims is not None:
                minority_victims = (black_v or 0) + (nat_amer_v or 0) + (asian_v or 0) + (hispanic_v or 0) + (other_v or 0)

            dv_rate = None
            sa_rate = None
            reporting_proxy = None
            if fem_pop and fem_pop > 0:
                if dv_total is not None:
                    dv_rate = (dv_total / fem_pop) * 100000.0
                if sex_assaults is not None:
                    sa_rate = (sex_assaults / fem_pop) * 100000.0
                if total_incidents is not None:
                    reporting_proxy = (total_incidents / fem_pop) * 100000.0

            out_row = {
                "geo_id": f"state:{abbr or state_name}",
                "geo_name": state_name,
                "geo_type": "state",
                "state": abbr,
                "county": "",
                "tribal_area": "",
                "year": year,
                "quarter": "",
                "month": "",
                "dv_rate": dv_rate,
                "sexual_assault_rate": sa_rate,
                "firearm_share": _safe_div(involving_firearm, total_incidents),
                "dating_partner_share": _safe_div(dating_partner, total_incidents),
                "minority_victim_share": _safe_div(minority_victims, total_victims),
                "native_american_victim_share": _safe_div(nat_amer_v, total_victims),
                "reporting_proxy": reporting_proxy,
                "risk_index": "",
                "data_quality_flag": "ok",
            }

            # Normalize to strings for CSV writer (empty string for None).
            for k, v in list(out_row.items()):
                if v is None:
                    out_row[k] = ""
                elif isinstance(v, float):
                    out_row[k] = f"{v:.10g}"
                else:
                    out_row[k] = str(v)

            w.writerow(out_row)

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

