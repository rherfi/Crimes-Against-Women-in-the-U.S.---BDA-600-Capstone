"""
Resource lookup for Victim Tool Bot (V1).

Safety / reliability constraints:
- We ONLY return resources that exist in our dataset file.
- We do NOT invent addresses, phone numbers, or availability.

V1 location matching is intentionally simple:
- ZIP code (5 digits)
- city/state
- county/state

Later upgrades can add real geocoding and verified datasets.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "resources_sample.json"


_ZIP_RE = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
_STATE_RE = re.compile(r"\b([A-Z]{2})\b")


@lru_cache(maxsize=1)
def _load_resources() -> List[Dict[str, Any]]:
    if not DATA_PATH.exists():
        return []
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _extract_zip(text: str) -> str:
    m = _ZIP_RE.search(text or "")
    return m.group(1) if m else ""


def _extract_state_abbrev(text: str) -> str:
    # We accept two-letter uppercase tokens if user types them (e.g., "NM", "AZ").
    m = _STATE_RE.search(text or "")
    return m.group(1) if m else ""


def _location_match_score(resource: Dict[str, Any], location_text: str) -> float:
    """
    Very simple matching score.

    - Exact ZIP match is strongest
    - city/state or county/state substring match is next
    - state-only match is weakest
    """
    loc = _norm(location_text)
    if not loc:
        return 0.0

    r_zip = _norm(str(resource.get("zip", "")))
    r_city = _norm(str(resource.get("city", "")))
    r_county = _norm(str(resource.get("county", "")))
    r_state = str(resource.get("state", "")).strip().upper()

    user_zip = _extract_zip(location_text)
    if user_zip and user_zip == r_zip:
        return 5.0

    # Substring matches (e.g., "Albuquerque, NM" contains "albuquerque")
    score = 0.0
    if r_city and r_city in loc:
        score += 2.0
    if r_county and r_county in loc:
        score += 1.5

    user_state = _extract_state_abbrev(location_text)
    if user_state and r_state and user_state == r_state:
        score += 1.0
    elif r_state and r_state.lower() in loc:
        score += 0.75

    return score


def find_resources(
    location_text: str,
    resource_type: Optional[str] = None,
    tribal_only: bool = False,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    Returns resources matching the location and optional filters.

    resource_type examples (V1):
    - emergency_hotline
    - domestic_violence_shelter
    - sexual_assault_service
    - legal_aid
    - tribal_service
    - housing_support
    - counseling
    - campus_resource
    """
    resources = _load_resources()
    loc = (location_text or "").strip()
    if not loc:
        return []

    out: List[Dict[str, Any]] = []
    for r in resources:
        if resource_type and _norm(r.get("resource_type", "")) != _norm(resource_type):
            continue
        if tribal_only and not bool(r.get("serves_tribal_communities", False)):
            continue

        score = _location_match_score(r, loc)
        if score <= 0:
            continue

        out.append({**r, "_score": score})

    out.sort(key=lambda x: float(x.get("_score", 0.0)), reverse=True)
    return [{k: v for k, v in r.items() if k != "_score"} for r in out[: max(0, limit)]]

