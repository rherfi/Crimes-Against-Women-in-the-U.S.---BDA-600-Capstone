"""
Very small parsing helpers for V1.

We do NOT use an LLM yet. Instead:
- detect metrics by keywords
- detect years
- detect geographies by a tiny alias map (from sample data)

Later: replace with an LLM + tool calling, but keep tool functions unchanged.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from app.services.data_loader import METRIC_COLUMNS, find_geo_ids_by_name, load_metrics_rows


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

    # exact metric column names
    for m in METRIC_COLUMNS:
        if m in t:
            return m

    # alias phrases
    for phrase, metric in METRIC_ALIASES.items():
        if phrase in t:
            return metric

    return None


def detect_geo_names(text: str) -> List[str]:
    """
    For V1 we detect any known geo_name substring in the question.
    This is crude, but works with the tiny sample dataset.
    """
    t = (text or "").lower()
    geo_names = sorted({r.geo_name for r in load_metrics_rows()}, key=len, reverse=True)
    found: List[str] = []
    for name in geo_names:
        if name.lower() in t:
            found.append(name)
    # de-dup preserving order
    out = []
    for x in found:
        if x not in out:
            out.append(x)
    return out


def resolve_geo(geo_name: str) -> Optional[Dict[str, str]]:
    """
    Convert a geo_name into a geo selector dict for tool functions.
    """
    candidates = find_geo_ids_by_name(geo_name)
    if not candidates:
        # allow state matching fallback for "California" etc.
        return {"state": geo_name}
    geo_id, name, geo_type = candidates[0]
    return {"geo_id": geo_id, "geo_name": name, "geo_type": geo_type}

