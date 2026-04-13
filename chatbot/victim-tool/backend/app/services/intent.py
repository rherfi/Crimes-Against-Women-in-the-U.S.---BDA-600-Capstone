"""
Intent classification for Victim Tool Bot (V1).

We intentionally keep this heuristic and transparent (not "smart"):
- More reliable
- Easier to audit for safety

Intent outcomes:
- crisis: user suggests immediate danger / imminent harm / emergency
- resource_lookup: user wants nearby help/resources
- vawa_info: user asks about VAWA rights/protections in plain language
- support_guidance: user wants supportive next steps without specific resource query
- unknown: everything else
"""

from __future__ import annotations

import re
from typing import Literal


Intent = Literal["crisis", "resource_lookup", "vawa_info", "support_guidance", "unknown"]


_CRISIS_PATTERNS = [
    r"\bhelp me\b",
    r"\bim in danger\b|\bi'm in danger\b",
    r"\bunsafe\b|\bnot safe\b",
    r"\bgoing to hurt\b|\bhurt me\b",
    r"\bkill\b|\bsuicide\b|\bself[- ]harm\b",
    r"\bweapon\b|\bgun\b|\bknife\b",
    r"\bright now\b|\btonight\b",
    r"\bscared\b",
]

_RESOURCE_HINTS = {
    "shelter",
    "hotline",
    "resources",
    "near me",
    "near",
    "nearby",
    "in my area",
    "find help",
    "dv shelter",
    "domestic violence shelter",
    "sexual assault",
    "legal aid",
    "counseling",
    "housing",
    "tribal services",
    "advocate",
}

_VAWA_HINTS = {
    "vawa",
    "dating partner",
    "dating partners",
    "tribal",
    "jurisdiction",
    "firearm",
    "gun",
    "prohibition",
    "protections",
    "rights",
    "campus",
    "underserved",
    "rural",
}

_SUPPORT_HINTS = {
    "i don't know what to do",
    "what should i do",
    "can you help",
    "i need guidance",
    "i need support",
    "i'm overwhelmed",
    "im overwhelmed",
    "i feel trapped",
}


def classify_user_intent(message: str) -> Intent:
    text = (message or "").strip().lower()
    if not text:
        return "unknown"

    # Crisis intent should win if present.
    for pat in _CRISIS_PATTERNS:
        if re.search(pat, text):
            return "crisis"

    has_resource = any(h in text for h in _RESOURCE_HINTS)
    has_vawa = any(h in text for h in _VAWA_HINTS)
    has_support = any(h in text for h in _SUPPORT_HINTS)

    if has_resource:
        return "resource_lookup"
    if has_vawa:
        return "vawa_info"
    if has_support:
        return "support_guidance"

    return "unknown"

