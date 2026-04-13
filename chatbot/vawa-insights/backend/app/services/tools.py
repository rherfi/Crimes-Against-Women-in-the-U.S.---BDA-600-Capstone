"""
Structured data tool functions requested by the user.

These are *the only* place numeric answers come from in V1.
If a tool cannot find data, it returns an explicit error message so the
chat orchestrator can respond "not enough data" (instead of guessing).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Literal, Optional, Tuple

from app.services.data_loader import METRIC_COLUMNS, load_metrics_rows, load_risk_components_rows


Frequency = Literal["year", "quarter", "month"]
SortDirection = Literal["asc", "desc"]
GeoLevel = Literal["state", "county", "tribal"]


def _metric_exists(metric: str) -> bool:
    return metric in METRIC_COLUMNS


def _filter_rows_by_geo(rows, geo: Dict[str, str]):
    """
    geo: {"geo_id": "..."} OR {"geo_name": "..."} OR {"state": "..."} etc.
    For V1 we keep this simple and prefer geo_id when present.
    """
    geo_id = (geo.get("geo_id") or "").strip()
    if geo_id:
        return [r for r in rows if r.geo_id == geo_id]

    geo_name = (geo.get("geo_name") or "").strip().lower()
    if geo_name:
        return [r for r in rows if r.geo_name.lower() == geo_name]

    # fallback: state name match if provided (used for "California" queries)
    state = (geo.get("state") or "").strip().lower()
    if state:
        return [r for r in rows if r.geo_type == "state" and r.geo_name.lower() == state]

    return []


def get_metric_timeseries(geo: Dict[str, str], metric: str, frequency: Frequency) -> Dict[str, Any]:
    """
    Returns timeseries points with citations.
    """
    if not _metric_exists(metric):
        return {"ok": False, "error": f"Unsupported metric '{metric}'.", "data": None}

    rows = _filter_rows_by_geo(load_metrics_rows(), geo)
    if not rows:
        return {"ok": False, "error": "No matching geography found in the dataset.", "data": None}

    # Only years 2021–2024 exist in sample data, but we don't hardcode that.
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

    # Sort for stable output
    points.sort(key=lambda x: x["period"])

    first = rows[0]
    citation = {
        "citation_type": "structured_data",
        "source_table": "metrics_sample.csv",
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
    """
    V1 supports year-to-year comparisons. start_period/end_period are years (e.g., "2021").
    """
    if not _metric_exists(metric):
        return {"ok": False, "error": f"Unsupported metric '{metric}'.", "data": None}

    try:
        start_year = int(start_period)
        end_year = int(end_period)
    except ValueError:
        return {"ok": False, "error": "V1 compare_geos supports year periods like '2021'..'2024'.", "data": None}

    rows = load_metrics_rows()
    a_rows = [r for r in _filter_rows_by_geo(rows, geo_a) if start_year <= r.year <= end_year and r.metrics.get(metric) is not None]
    b_rows = [r for r in _filter_rows_by_geo(rows, geo_b) if start_year <= r.year <= end_year and r.metrics.get(metric) is not None]

    if not a_rows or not b_rows:
        return {"ok": False, "error": "Not enough data to compare both geographies for the requested years.", "data": None}

    def yearly_avg(rs):
        # sample data has one row per year per geo; we average anyway for future-proofing
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
        "source_table": "metrics_sample.csv",
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
    """
    Returns top/bottom rankings for a metric in a given year for a geo_level.
    """
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
        "source_table": "metrics_sample.csv",
        "metric": metric,
        "year": year,
        "geo_level": geo_level,
        "sort_direction": sort_direction,
    }

    return {"ok": True, "data": {"metric": metric, "year": year, "geo_level": geo_level, "ranked": ranked}, "citation": citation}


def get_risk_profile(geo: Dict[str, str], year: int) -> Dict[str, Any]:
    """
    Returns risk index + component breakdown (from a separate small table).
    """
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
            "source_table": "metrics_sample.csv",
            "geo_id": r0.geo_id,
            "geo_name": r0.geo_name,
            "geo_type": r0.geo_type,
            "metric": "risk_index",
            "year": year,
        },
        {
            "citation_type": "structured_data",
            "source_table": "risk_components_sample.csv",
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

