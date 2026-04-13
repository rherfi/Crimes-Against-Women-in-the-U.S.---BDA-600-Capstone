"""
Chat orchestrator for Victim Tool Bot (V1).

Flow:
1) classify intent
2) retrieve resources (if needed)
3) retrieve KB chunks (if needed)
4) build safe structured response + citations
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from app.schemas import ChatRequest, ChatResponse, DebugPayload
from app.services.intent import classify_user_intent
from app.services.kb_retriever import RetrievedChunk, get_vawa_info
from app.services.resources import find_resources
from app.services.response_builder import build_safe_response


_ZIP_RE = re.compile(r"\b(\d{5})(?:-\d{4})?\b")


def _detect_location_text(message: str) -> str:
    """
    Very small helper to extract a likely location phrase.

    V1 behavior:
    - If a ZIP exists, return the ZIP
    - Else if phrase contains "near <...>" return trailing phrase
    - Else if "in <...>" return trailing phrase
    - Else empty string
    """
    text = (message or "").strip()
    if not text:
        return ""

    m = _ZIP_RE.search(text)
    if m:
        return m.group(1)

    lower = text.lower()
    for key in ["near ", "in "]:
        idx = lower.find(key)
        if idx != -1 and len(text) > idx + len(key):
            candidate = text[idx + len(key) :].strip(" .")
            # Avoid returning huge slices.
            if 2 <= len(candidate) <= 60:
                return candidate

    return ""


def _detect_resource_type(message: str) -> Tuple[Optional[str], List[str]]:
    """
    Detect a single resource type filter if user is specific.
    Returns (resource_type, debug_filters)
    """
    text = (message or "").lower()
    debug_filters: List[str] = []

    mapping = [
        ("emergency_hotline", ["hotline", "crisis line", "crisis hotline"]),
        ("domestic_violence_shelter", ["shelter", "dv shelter", "domestic violence shelter"]),
        ("sexual_assault_service", ["sexual assault", "rape", "rainn"]),
        ("legal_aid", ["legal aid", "lawyer", "protective order", "restraining order"]),
        ("tribal_service", ["tribal", "reservation", "native", "indian country"]),
        ("housing_support", ["housing", "safe housing"]),
        ("counseling", ["counseling", "therapy", "support group"]),
        ("campus_resource", ["campus", "university", "title ix", "student"]),
    ]

    for rtype, hints in mapping:
        if any(h in text for h in hints):
            debug_filters.append(f"resource_type={rtype}")
            return rtype, debug_filters

    return None, debug_filters


def _detect_tribal_only(message: str) -> bool:
    text = (message or "").lower()
    return any(k in text for k in ["tribal", "reservation", "indian country", "native"])


def _doc_debug(chunks: List[RetrievedChunk]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ch in chunks:
        out.append(
            {
                "chunk_id": ch.chunk_id,
                "score": ch.score,
                "title": ch.metadata.get("title", ""),
                "citation_id": ch.metadata.get("citation_id", ""),
                "source_file": ch.metadata.get("source_file", ""),
            }
        )
    return out


def _doc_citations(chunks: List[RetrievedChunk]) -> List[str]:
    """
    V1 citations are human-readable strings the UI can show.
    """
    citations: List[str] = []
    seen = set()
    for ch in chunks:
        cid = ch.metadata.get("citation_id", ch.chunk_id)
        title = ch.metadata.get("title", "knowledge base doc")
        c = f"{cid} — {title}"
        if c not in seen:
            citations.append(c)
            seen.add(c)
    return citations


def answer_chat(req: ChatRequest) -> ChatResponse:
    message = (req.message or "").strip()
    intent = classify_user_intent(message)

    location_text = _detect_location_text(message)
    resource_type, resource_filters = _detect_resource_type(message)
    tribal_only = _detect_tribal_only(message)
    if tribal_only:
        resource_filters.append("tribal_only=true")

    resources: List[Dict[str, Any]] = []
    docs: List[RetrievedChunk] = []
    citations: List[str] = []

    # CRISIS: show crisis resources even without a location (national options).
    if intent == "crisis":
        # We pull from the dataset using a special location fallback:
        # if user did not share a location, we use "US" so national resources match.
        lookup_loc = location_text or "US"
        resources = find_resources(lookup_loc, resource_type="emergency_hotline", tribal_only=False, limit=5)
        docs = get_vawa_info("limits disclaimer not legal advice crisis", top_k=2)
        citations.extend(_doc_citations(docs))

    elif intent == "resource_lookup":
        if not location_text:
            # No location: we return an empty resource list and let the response builder
            # ask for city/state, county/state, or ZIP in a supportive way.
            resources = []
        else:
            resources = find_resources(
                location_text,
                resource_type=resource_type,
                tribal_only=tribal_only,
                limit=5,
            )
        docs = get_vawa_info("victim services hotline shelter safety planning", top_k=2)
        citations.extend(_doc_citations(docs))

    elif intent == "vawa_info":
        docs = get_vawa_info(message, top_k=4)
        citations.extend(_doc_citations(docs))

    elif intent == "support_guidance":
        docs = get_vawa_info("safety planning victim services limits disclaimer", top_k=3)
        citations.extend(_doc_citations(docs))

    else:
        docs = get_vawa_info(message, top_k=2)
        citations.extend(_doc_citations(docs))

    # If we didn't retrieve any docs, still add a minimal citation marker.
    if not citations:
        citations = ["V1-NO-SOURCE — No matching KB chunk found (prototype)"]

    # If the user asked for resources but didn't provide a location, we make the prompt explicit.
    if intent == "resource_lookup" and not location_text:
        response = build_safe_response(
            user_message=message,
            intent=intent,
            resources=[],
            docs=_doc_debug(docs),
            citations=citations,
        )
        response.support_message = (
            "I can help find resources near you. What city/state, county/state, or ZIP code should I search?"
        )
    else:
        response = build_safe_response(
            user_message=message,
            intent=intent,
            resources=resources,
            docs=_doc_debug(docs),
            citations=citations,
        )

    debug = DebugPayload(
        intent=intent,
        location_detected=location_text,
        resource_filters=resource_filters,
        docs_retrieved=_doc_debug(docs),
    )

    return ChatResponse(response=response, debug=debug)

