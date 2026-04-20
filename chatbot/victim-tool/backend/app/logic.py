"""
Victim Tool Bot — non-API logic in one module (V1).

Split into multiple files later only when this file grows hard to navigate.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.schemas import (
    ChatRequest,
    ChatResponse,
    DebugPayload,
    Intent,
    ResourceOut,
    ResponsePayload,
)

# ---------------------------------------------------------------------------
# Knowledge base (markdown under app/kb/)
# ---------------------------------------------------------------------------

KB_DIR = Path(__file__).resolve().parent / "kb"
_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def _parse_frontmatter(md_text: str) -> Tuple[Dict[str, Any], str]:
    lines = (md_text or "").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, md_text

    meta: Dict[str, Any] = {}
    i = 1
    while i < len(lines):
        if lines[i].strip() == "---":
            body = "\n".join(lines[i + 1 :])
            return meta, body
        line = lines[i].strip()
        if line and ":" in line:
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip()
            if "," in v:
                meta[k] = [x.strip() for x in v.split(",") if x.strip()]
            else:
                meta[k] = v
        i += 1

    return {}, md_text


def _chunk_markdown(body: str) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n+", body or "") if p.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0
    for p in paras:
        if buf_len + len(p) > 850 and buf:
            chunks.append("\n\n".join(buf).strip())
            buf = []
            buf_len = 0
        buf.append(p)
        buf_len += len(p)
    if buf:
        chunks.append("\n\n".join(buf).strip())
    return chunks


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]


def retrieve(query: str, top_k: int = 4) -> List[RetrievedChunk]:
    q_tokens = set(_tokenize(query))
    if not q_tokens:
        return []

    results: List[RetrievedChunk] = []
    for path in sorted(KB_DIR.glob("*.md")):
        md_text = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(md_text)
        chunks = _chunk_markdown(body)

        for idx, chunk_text in enumerate(chunks):
            c_tokens = set(_tokenize(chunk_text))
            overlap = q_tokens.intersection(c_tokens)
            if not overlap:
                continue

            bonus = 0.0
            tags = meta.get("topic_tags", [])
            if isinstance(tags, list):
                for t in tags:
                    if str(t).lower() in q_tokens:
                        bonus += 0.25

            score = float(len(overlap)) + bonus
            results.append(
                RetrievedChunk(
                    chunk_id=f"{path.stem}::p{idx+1}",
                    text=chunk_text,
                    score=score,
                    metadata={
                        **meta,
                        "source_file": path.name,
                    },
                )
            )

    results.sort(key=lambda r: r.score, reverse=True)
    return results[: max(0, top_k)]


def get_vawa_info(topic: str, top_k: int = 4) -> List[RetrievedChunk]:
    return retrieve(topic, top_k=top_k)


# ---------------------------------------------------------------------------
# Resources (JSON under app/data/)
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).resolve().parent / "data" / "resources_sample.json"
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
    m = _STATE_RE.search(text or "")
    return m.group(1) if m else ""


def _location_match_score(resource: Dict[str, Any], location_text: str) -> float:
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


# ---------------------------------------------------------------------------
# Intent
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Response templates
# ---------------------------------------------------------------------------


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
    resources = resources or []
    docs = docs or []
    citations = citations or []

    support_message = ""
    practical_next_steps: List[str] = []
    caveats: List[str] = []

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

    if docs and intent in {"vawa_info", "support_guidance", "unknown"}:
        caveats.append("I’m using a small local knowledge base in this prototype; it may be incomplete.")

    return ResponsePayload(
        support_message=support_message,
        resources=[_resource_out(r) for r in resources],
        practical_next_steps=practical_next_steps,
        caveats=caveats,
        citations=citations,
    )


# ---------------------------------------------------------------------------
# Chat entrypoint
# ---------------------------------------------------------------------------


def _detect_location_text(message: str) -> str:
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
            if 2 <= len(candidate) <= 60:
                return candidate

    return ""


def _detect_resource_type(message: str) -> Tuple[Optional[str], List[str]]:
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

    if intent == "crisis":
        lookup_loc = location_text or "US"
        resources = find_resources(lookup_loc, resource_type="emergency_hotline", tribal_only=False, limit=5)
        docs = get_vawa_info("limits disclaimer not legal advice crisis", top_k=2)
        citations.extend(_doc_citations(docs))

    elif intent == "resource_lookup":
        if not location_text:
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

    if not citations:
        citations = ["V1-NO-SOURCE — No matching KB chunk found (prototype)"]

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
