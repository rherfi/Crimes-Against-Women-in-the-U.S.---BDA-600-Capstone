"""
Structured data loader for V1.

V1 approach:
- Use small CSV files stored locally under app/data/
- Load into memory on first use (tiny data)
- Expose simple query helpers

Later we can swap to:
- Parquet + DuckDB
- PostgreSQL
- A proper metrics service layer
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


METRICS_FILE = DATA_DIR / "metrics_sample.csv"
RISK_COMPONENTS_FILE = DATA_DIR / "risk_components_sample.csv"


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
    geo_type: str  # state | county | tribal
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


def find_geo_ids_by_name(name: str) -> List[Tuple[str, str, str]]:
    """
    Very small V1 helper to map a user geo string to known geos.
    Returns (geo_id, geo_name, geo_type) candidates.
    """
    name_norm = (name or "").strip().lower()
    if not name_norm:
        return []

    rows = load_metrics_rows()
    seen = {}
    for r in rows:
        key = (r.geo_id, r.geo_name, r.geo_type)
        if key in seen:
            continue
        # crude match: exact or substring
        gn = r.geo_name.lower()
        if name_norm == gn or name_norm in gn:
            seen[key] = True

    return list(seen.keys())

