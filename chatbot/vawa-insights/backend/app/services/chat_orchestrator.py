"""
Chat orchestrator for V1.

Key guarantees:
- Numeric values come only from tool outputs (structured CSV)
- Always include citations (structured data citations and/or doc citations)
- Clearly label descriptive vs (non-)causal statements
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.schemas import ChatRequest, ChatResponse
from app.services.intent import classify_intent
from app.services.parsing import detect_geo_names, detect_metric, extract_years, resolve_geo
from app.services.retriever import retrieve
from app.services.tools import compare_geos, get_metric_timeseries, get_risk_profile, rank_geos


def _doc_citation_from_chunk(chunk) -> Dict[str, Any]:
    md = chunk.metadata
    return {
        "citation_type": "knowledge_base",
        "citation_id": md.get("citation_id", chunk.chunk_id),
        "title": md.get("title", md.get("source_file", "knowledge doc")),
        "doc_type": md.get("doc_type", ""),
        "source_file": md.get("source_file", ""),
        "chunk_id": chunk.chunk_id,
        "years_covered": md.get("years_covered", ""),
        "geo_level": md.get("geo_level", ""),
        "metric_tags": md.get("metric_tags", []),
    }


def answer_chat(req: ChatRequest) -> ChatResponse:
    message = (req.message or "").strip()
    intent = classify_intent(message)

    tools_used: List[Dict[str, Any]] = []
    docs_retrieved: List[Dict[str, Any]] = []
    citations: List[Dict[str, Any]] = []

    # ---- Data side (tools) ----
    metric = detect_metric(message)
    years = extract_years(message)
    geo_names = detect_geo_names(message)

    evidence_lines: List[str] = []
    caveats: List[str] = []
    interpretation = ""
    direct_answer = ""

    # Always add a core caveat when we present descriptive time patterns.
    descriptive_caveat = "These are descriptive patterns from observed data and do not, by themselves, establish causation."

    if intent in {"data_only", "data_and_docs"}:
        # Handle: "risk profile for X in 2024"
        if "risk profile" in message.lower() or (metric == "risk_index" and "profile" in message.lower()):
            if not geo_names or not years:
                direct_answer = "I need a geography and a year to generate a risk profile (e.g., “New Mexico in 2024”)."
            else:
                geo = resolve_geo(geo_names[0])
                yr = years[-1]
                tool_out = get_risk_profile(geo or {}, yr)
                tools_used.append({"tool": "get_risk_profile", "args": {"geo": geo, "year": yr}, "ok": tool_out.get("ok")})
                if tool_out.get("ok"):
                    data = tool_out["data"]
                    direct_answer = f"Risk profile for {data['geo_name']} ({yr}): risk_index = {data['risk_index']:.2f}."
                    evidence_lines.append(f"risk_index ({yr}): {data['risk_index']:.2f} (data_quality_flag={data['data_quality_flag']})")
                    if data.get("components"):
                        evidence_lines.append("Component breakdown:")
                        for c in data["components"]:
                            evidence_lines.append(f"- {c['component']}: {c['value']:.2f} ({c.get('note','')})".strip())
                    citations.extend(tool_out.get("citations", []))
                    caveats.append(descriptive_caveat)
                else:
                    direct_answer = f"I don’t have enough structured data to compute that risk profile. ({tool_out.get('error')})"

        # Handle: "Which states had the highest DV rate in 2024?"
        elif any(k in message.lower() for k in ["highest", "top", "rank"]) and ("state" in message.lower() or "states" in message.lower()):
            if metric is None:
                direct_answer = "Which metric should I rank (e.g., dv_rate, firearm_share)?"
            else:
                yr = years[-1] if years else 2024
                tool_out = rank_geos(metric=metric, year=yr, geo_level="state", top_n=5, sort_direction="desc")
                tools_used.append({"tool": "rank_geos", "args": {"metric": metric, "year": yr, "geo_level": "state", "top_n": 5, "sort_direction": "desc"}, "ok": tool_out.get("ok")})
                if tool_out.get("ok"):
                    ranked = tool_out["data"]["ranked"]
                    direct_answer = f"Top states for {metric} in {yr} (sample data): " + ", ".join([f"{r['geo_name']} ({r['value']:.2f})" for r in ranked])
                    evidence_lines.append(f"Ranking: {metric} in {yr} (top {len(ranked)} states).")
                    for r in ranked:
                        evidence_lines.append(f"- #{r['rank']} {r['geo_name']}: {r['value']:.2f} (flag={r['data_quality_flag']})")
                    citations.append(tool_out["citation"])
                    caveats.append(descriptive_caveat)
                else:
                    direct_answer = f"I don’t have enough structured data to rank that. ({tool_out.get('error')})"

        # Handle: "Compare California and Texas in firearm involvement from 2021 to 2024."
        elif "compare" in message.lower() and len(geo_names) >= 2 and metric is not None and len(years) >= 1:
            geo_a = resolve_geo(geo_names[0])
            geo_b = resolve_geo(geo_names[1])
            start_year = str(min(years))
            end_year = str(max(years))
            tool_out = compare_geos(geo_a or {}, geo_b or {}, metric, start_year, end_year)
            tools_used.append({"tool": "compare_geos", "args": {"geo_a": geo_a, "geo_b": geo_b, "metric": metric, "start_period": start_year, "end_period": end_year}, "ok": tool_out.get("ok")})
            if tool_out.get("ok"):
                d = tool_out["data"]
                a = d["geo_a"]
                b = d["geo_b"]
                direct_answer = (
                    f"From {d['start_year']}–{d['end_year']}, {a['geo_name']} averaged {a['avg_value']:.2f} for {metric}, "
                    f"vs {b['geo_name']} at {b['avg_value']:.2f} (difference {d['difference']:.2f})."
                )
                evidence_lines.append(f"{a['geo_name']} average ({d['start_year']}–{d['end_year']}): {a['avg_value']:.2f}")
                evidence_lines.append(f"{b['geo_name']} average ({d['start_year']}–{d['end_year']}): {b['avg_value']:.2f}")
                citations.append(tool_out["citation"])
                caveats.append(descriptive_caveat)
            else:
                direct_answer = f"I don’t have enough structured data to compare those geographies. ({tool_out.get('error')})"

        # Fallback: timeseries for a single geo + metric
        elif metric is not None and geo_names:
            geo = resolve_geo(geo_names[0])
            tool_out = get_metric_timeseries(geo or {}, metric, frequency="year")
            tools_used.append({"tool": "get_metric_timeseries", "args": {"geo": geo, "metric": metric, "frequency": "year"}, "ok": tool_out.get("ok")})
            if tool_out.get("ok"):
                pts = tool_out["data"]["points"]
                direct_answer = f"{metric} over time for {tool_out['data']['geo']['geo_name']} (sample data)."
                evidence_lines.extend([f"- {p['period']}: {p['value']:.2f} (flag={p['data_quality_flag']})" for p in pts])
                citations.append(tool_out["citation"])
                caveats.append(descriptive_caveat)
            else:
                direct_answer = f"I don’t have enough structured data for that request. ({tool_out.get('error')})"

        else:
            direct_answer = "I couldn’t identify a supported metric and geography to answer with numbers. Try including a metric (e.g., dv_rate) and a place (e.g., California)."

    # ---- Docs side (retrieval) ----
    if intent in {"docs_only", "data_and_docs"}:
        chunks = retrieve(message, top_k=4)
        for ch in chunks:
            docs_retrieved.append(
                {
                    "chunk_id": ch.chunk_id,
                    "score": ch.score,
                    "title": ch.metadata.get("title", ""),
                    "citation_id": ch.metadata.get("citation_id", ""),
                }
            )
            citations.append(_doc_citation_from_chunk(ch))

        if chunks:
            # For V1 we "ground" the explanation by quoting/summarizing retrieved text,
            # but we keep it short.
            if direct_answer == "":
                direct_answer = "Here’s what the knowledge base says (V1 prototype):"
            interpretation_bits = []
            for ch in chunks[:2]:
                snippet = ch.text.strip().replace("\n", " ")
                if len(snippet) > 280:
                    snippet = snippet[:280].rstrip() + "…"
                interpretation_bits.append(f"- {snippet}")
            interpretation = "\n".join(interpretation_bits)
        else:
            # If docs-only and we retrieved nothing, we should be explicit.
            if intent == "docs_only":
                direct_answer = "I don’t have enough knowledge-base content to answer that yet (V1)."
            caveats.append("Knowledge base retrieval returned no matching documents for this query in V1.")

    # Global caveat: if user implies causation, be explicit.
    if any(w in message.lower() for w in ["cause", "caused", "because", "led to", "impact", "effect"]):
        caveats.append("This bot can describe trends/associations in the available data, but V1 does not establish causal effects.")

    # Ensure we always return at least one citation.
    if not citations:
        citations.append(
            {
                "citation_type": "system",
                "citation_id": "V1-NO-SOURCE",
                "title": "No matching structured rows or KB chunks found",
            }
        )

    if interpretation == "":
        interpretation = "Interpretation is limited in V1. If you want, ask for caveats/definitions and I’ll pull from the methodology documents."

    return ChatResponse(
        answer={
            "direct_answer": direct_answer,
            "evidence": evidence_lines,
            "interpretation": interpretation,
            "caveats": caveats,
            "citations": citations,
        },
        debug={
            "intent": intent,
            "tools_used": tools_used,
            "docs_retrieved": docs_retrieved,
        },
    )

