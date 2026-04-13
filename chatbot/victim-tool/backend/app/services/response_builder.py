"""
Safe response builder for Victim Tool Bot (V1).

This is the core "behavior" of the bot.

Safety design principles:
- Trauma-informed tone: supportive, calm, non-judgmental
- Do not ask for graphic details
- Distinguish crisis vs general info
- Do not provide legal advice; provide general information + encourage qualified help
- Never fabricate resources; only return resources present in the dataset
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.schemas import Intent, ResourceOut, ResponsePayload


def _format_location(r: Dict[str, Any]) -> str:
    parts = []
    city = (r.get("city") or "").strip()
    state = (r.get("state") or "").strip()
    county = (r.get("county") or "").strip()
    zip_code = (r.get("zip") or "").strip()
    if city:
        parts.append(city)
    if state:
        parts.append(state)
    if zip_code:
        parts.append(zip_code)
    if not parts and county and state:
        parts.append(f"{county} County, {state}")
    return ", ".join(parts).strip()


def _resource_out(r: Dict[str, Any]) -> ResourceOut:
    return ResourceOut(
        name=str(r.get("name", "")),
        resource_type=str(r.get("resource_type", "")),
        phone=str(r.get("phone", "")),
        website=str(r.get("website", "")),
        location=_format_location(r),
        notes=str(r.get("notes", "")),
        source=str(r.get("source_name", "")),
    )


def build_safe_response(
    user_message: str,
    intent: Intent,
    resources: Optional[List[Dict[str, Any]]] = None,
    docs: Optional[List[Dict[str, Any]]] = None,
    citations: Optional[List[str]] = None,
) -> ResponsePayload:
    """
    Builds a structured, trauma-informed response for the UI.
    """
    resources = resources or []
    docs = docs or []
    citations = citations or []

    support_message = ""
    practical_next_steps: List[str] = []
    caveats: List[str] = []

    # Common caveats for this sensitive-use tool.
    base_caveats = [
        "This is general information and support—not legal advice, medical advice, or emergency services.",
        "If you feel unsafe or in immediate danger, call 911 (U.S.) or your local emergency number.",
    ]

    if intent == "crisis":
        support_message = (
            "I’m really sorry you’re going through this. You deserve support, and you don’t have to handle this alone."
        )
        practical_next_steps = [
            "If you are in immediate danger, call 911 (U.S.) or your local emergency number now.",
            "If calling feels unsafe, consider going to a safer place if you can (a neighbor, a public place, or someone you trust).",
            "If you want, you can tell me your city/state, county/state, or ZIP code and what kind of help you want (shelter, hotline, legal aid).",
        ]
        caveats = [
            "I won’t ask for graphic details.",
            "If you share a location, keep it general (city/state or ZIP) rather than a full address.",
            *base_caveats,
        ]

    elif intent == "resource_lookup":
        support_message = "You deserve support. I can help you find options that may be near you."
        practical_next_steps = [
            "If any option feels unsafe to contact, trust that feeling—your safety matters most.",
            "When you reach out, you can start with: “I’m looking for confidential support and resources.”",
        ]
        caveats = [
            "Availability can change quickly (hours, bed space, waitlists). Please verify by phone/official website.",
            "If you are in immediate danger, call 911 (U.S.) or your local emergency number.",
            *base_caveats[:1],
        ]

    elif intent == "vawa_info":
        support_message = "I can explain VAWA-related protections in plain language."
        practical_next_steps = [
            "If you want, tell me which topic matters most right now (dating partners, tribal jurisdiction, firearms, campus protections, victim services).",
            "If your question is about your specific situation, consider speaking with a qualified advocate or attorney in your area.",
        ]
        caveats = [
            "Laws and eligibility can be fact-specific and may vary by jurisdiction.",
            *base_caveats[:1],
        ]

    elif intent == "support_guidance":
        support_message = "I’m here with you. You deserve support and choices."
        practical_next_steps = [
            "If you are in immediate danger, call 911 (U.S.) or your local emergency number.",
            "If you’re not in immediate danger, consider contacting a confidential hotline or local advocate for safety planning.",
            "If you share your city/state, county/state, or ZIP code, I can list resource options near you.",
        ]
        caveats = [
            "You don’t need to share graphic details for me to help with next steps.",
            *base_caveats[:1],
        ]

    else:
        support_message = (
            "I can help with (1) finding resources near you, (2) plain-language VAWA information, or (3) practical support guidance."
        )
        practical_next_steps = [
            "If you want resources near you, share a city/state, county/state, or ZIP code.",
            "If you want VAWA information, tell me the topic (dating partners, tribal jurisdiction, firearm prohibitions, victim services).",
        ]
        caveats = [*base_caveats[:1]]

    # Attach docs (as short citations + do not overquote).
    if docs and intent in {"vawa_info", "support_guidance", "unknown"}:
        caveats.append("I’m using a small local knowledge base in this prototype; it may be incomplete.")

    return ResponsePayload(
        support_message=support_message,
        resources=[_resource_out(r) for r in resources],
        practical_next_steps=practical_next_steps,
        caveats=caveats,
        citations=citations,
    )

