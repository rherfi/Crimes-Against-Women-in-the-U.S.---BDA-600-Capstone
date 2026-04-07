"""
Intent classification for V1.

We keep this heuristic and transparent:
- data_only: question asks for numbers / ranking / comparisons
- docs_only: question is conceptual about VAWA / methodology
- data_and_docs: question asks for trends + caveats/limitations/explanations
"""

from __future__ import annotations

import re
from typing import Literal


Intent = Literal["data_only", "docs_only", "data_and_docs"]


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
    has_metric = any(h in text for h in _METRIC_HINTS)
    has_docs = any(h in text for h in _DOCS_HINTS)

    # If someone asks directly "what does VAWA change", treat as docs.
    if re.search(r"\bwhat does vawa\b|\bwhat is vawa\b|\bwhat changed\b", text):
        return "docs_only"

    if has_metric and has_docs:
        return "data_and_docs"
    if has_metric:
        return "data_only"
    if has_docs:
        return "docs_only"

    # Default: retrieve docs (safer than making up numbers).
    return "docs_only"

