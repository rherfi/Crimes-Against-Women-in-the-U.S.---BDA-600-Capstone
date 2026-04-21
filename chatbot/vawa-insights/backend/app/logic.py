"""
VAWA Insights Bot — non-API logic in one module.

Split into multiple files later only when this file grows hard to navigate.
"""

from __future__ import annotations

import csv
import json
import os
import ast
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from app.schemas import ChatRequest, ChatResponse, MapEmbedPayload

# ---------------------------------------------------------------------------
# Data loader (CSV under app/data/)
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"

# Public ArcGIS Dashboard (override with ARCGIS_DASHBOARD_URL for staging/production).
DEFAULT_ARCGIS_DASHBOARD_URL = (
    "https://sdsugeo.maps.arcgis.com/apps/dashboards/7fa98d321eb94e9baaffc43fe65b973e"
)

_METRIC_LABELS: Dict[str, str] = {
    "dv_rate": "Domestic violence rate",
    "sexual_assault_rate": "Sexual assault rate",
    "firearm_share": "Firearm involvement share",
    "dating_partner_share": "Dating partner share",
    "minority_victim_share": "Minority victim share",
    "native_american_victim_share": "Native American victim share",
    "reporting_proxy": "Reporting proxy",
    "risk_index": "Risk index",
}

METRICS_FILE = DATA_DIR / "metrics.csv"
RISK_COMPONENTS_FILE = DATA_DIR / "risk_components.csv"

# EDA outputs (kept outside chatbot to keep backend small).
REPO_ROOT = Path(__file__).resolve().parents[4]
EDA_OUTPUT_DIR = REPO_ROOT / "EDA" / "output"
POLICY_VARIABLE_LEVEL_FILE = EDA_OUTPUT_DIR / "policy_variable_level_summary.csv"
POLICY_STATE_LEVEL_CHANGES_FILE = EDA_OUTPUT_DIR / "policy_state_level_changes.csv"

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
    geo_type: str
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


# ---------------------------------------------------------------------------
# Policy stats tools (pre vs post 2022) from EDA/output
# ---------------------------------------------------------------------------

POLICY_VARIABLE_ALIASES: Dict[str, str] = {
    "total incidents": "total_incidents",
    "incidents": "total_incidents",
    "domestic violence": "dv_total",
    "dv total": "dv_total",
    "sexual assaults": "sex_assaults",
    "sexual assault": "sex_assaults",
    "firearm involvement": "involving_firearm",
    "gun involvement": "involving_firearm",
    "nonmarried partner": "victim_offender_nonmarried_partner",
    "unmarried partner": "victim_offender_nonmarried_partner",
    "dating partner": "victim_offender_nonmarried_partner",
    "spouse": "victim_offender_rel_spouse",
    "near schools": "near_school",
    "near school": "near_school",
    "tribal lands": "on_tribal_lands",
    "on tribal lands": "on_tribal_lands",
}


def _load_policy_rows(path: Path, cache_key: str) -> List[Dict[str, str]]:
    if cache_key in _CACHE:
        return _CACHE[cache_key]
    if not path.exists():
        _CACHE[cache_key] = []
        return []
    rows = _read_csv(path)
    _CACHE[cache_key] = rows
    return rows


def list_policy_variables() -> List[str]:
    rows = _load_policy_rows(POLICY_VARIABLE_LEVEL_FILE, "policy_variable_level_rows")
    vars_ = sorted({(r.get("variable") or "").strip() for r in rows if (r.get("variable") or "").strip()})
    return vars_


def detect_policy_variable(text: str) -> Optional[str]:
    t = (text or "").lower()
    for phrase, var in POLICY_VARIABLE_ALIASES.items():
        if phrase in t:
            return var
    for v in list_policy_variables():
        if v.lower() in t:
            return v
    return None


def _as_float(x: str) -> Optional[float]:
    try:
        return float((x or "").strip())
    except Exception:
        return None


def _as_int(x: str) -> Optional[int]:
    try:
        return int(float((x or "").strip()))
    except Exception:
        return None


def get_policy_variable_summary(variable: str) -> Dict[str, Any]:
    v = (variable or "").strip()
    if not v:
        return {"ok": False, "error": "Missing variable.", "data": None}

    rows = _load_policy_rows(POLICY_VARIABLE_LEVEL_FILE, "policy_variable_level_rows")
    match = next((r for r in rows if (r.get("variable") or "").strip() == v), None)
    if not match:
        return {"ok": False, "error": f"No policy summary found for variable '{v}'.", "data": None}

    def parse_top_list(s: str) -> List[Dict[str, Any]]:
        s = (s or "").strip()
        if not s:
            return []
        try:
            val = ast.literal_eval(s)
            return val if isinstance(val, list) else []
        except Exception:
            return []

    data = {
        "variable": v,
        "variable_type": (match.get("variable_type") or "").strip(),
        "states_with_valid_percent_change": _as_int(match.get("states_with_valid_percent_change", "")),
        "mean_absolute_change": _as_float(match.get("mean_absolute_change", "")),
        "median_absolute_change": _as_float(match.get("median_absolute_change", "")),
        "mean_percent_change": _as_float(match.get("mean_percent_change", "")),
        "median_percent_change": _as_float(match.get("median_percent_change", "")),
        "states_increased": _as_int(match.get("states_increased", "")),
        "states_decreased": _as_int(match.get("states_decreased", "")),
        "states_no_change": _as_int(match.get("states_no_change", "")),
        "top_5_largest_decreases": parse_top_list(match.get("top_5_largest_decreases", "")),
        "top_5_largest_increases": parse_top_list(match.get("top_5_largest_increases", "")),
        "source_file": POLICY_VARIABLE_LEVEL_FILE.name,
    }

    citation = {
        "citation_type": "structured_data",
        "source_table": POLICY_VARIABLE_LEVEL_FILE.name,
        "variable": v,
        "periods": ["pre_2022", "post_2022_avg"],
    }

    return {"ok": True, "data": data, "citation": citation}


def rank_states_by_policy_change(variable: str, direction: Literal["increase", "decrease"], top_n: int = 5, *, min_pre_value: float) -> Dict[str, Any]:
    v = (variable or "").strip()
    if not v:
        return {"ok": False, "error": "Missing variable.", "data": None}
    top_n = max(1, min(int(top_n or 5), 20))

    rows = _load_policy_rows(POLICY_STATE_LEVEL_CHANGES_FILE, "policy_state_level_changes_rows")
    filtered = [r for r in rows if (r.get("variable") or "").strip() == v]
    if not filtered:
        return {"ok": False, "error": f"No rows found for variable '{v}'.", "data": None}

    out = []
    for r in filtered:
        pc = _as_float(r.get("percent_change", ""))
        if pc is None:
            continue
        pre = _as_float(r.get("pre_2022", ""))
        if pre is not None and pre < float(min_pre_value):
            continue
        out.append(
            {
                "state": (r.get("state") or "").strip(),
                "percent_change": pc,
                "absolute_change": _as_float(r.get("absolute_change", "")),
                "pre_2022": pre,
                "post_2022_avg": _as_float(r.get("post_2022_avg", "")),
                "direction": (r.get("direction") or "").strip(),
            }
        )

    if not out:
        return {"ok": False, "error": f"No numeric percent_change values found for variable '{v}'.", "data": None}

    reverse = direction == "increase"
    out.sort(key=lambda x: x["percent_change"], reverse=reverse)
    ranked = out[:top_n]
    for i, r in enumerate(ranked, start=1):
        r["rank"] = i

    citation = {
        "citation_type": "structured_data",
        "source_table": POLICY_STATE_LEVEL_CHANGES_FILE.name,
        "variable": v,
        "sort": f"percent_change_{direction}",
        "top_n": top_n,
        "filters": {"min_pre_value": min_pre_value},
        "periods": ["pre_2022", "post_2022_avg"],
    }

    return {"ok": True, "data": {"variable": v, "direction": direction, "ranked": ranked}, "citation": citation}

def find_geo_ids_by_name(name: str) -> List[Tuple[str, str, str]]:
    name_norm = (name or "").strip().lower()
    if not name_norm:
        return []

    rows = load_metrics_rows()
    seen = {}
    for r in rows:
        key = (r.geo_id, r.geo_name, r.geo_type)
        if key in seen:
            continue
        gn = r.geo_name.lower()
        if name_norm == gn or name_norm in gn:
            seen[key] = True

    return list(seen.keys())


# ---------------------------------------------------------------------------
# Query parsing
# ---------------------------------------------------------------------------

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

    for m in METRIC_COLUMNS:
        if m in t:
            return m

    for phrase, metric in METRIC_ALIASES.items():
        if phrase in t:
            return metric

    return None


def detect_geo_names(text: str) -> List[str]:
    t = (text or "").lower()
    geo_names = sorted({r.geo_name for r in load_metrics_rows()}, key=len, reverse=True)
    found: List[str] = []
    for name in geo_names:
        if name.lower() in t:
            found.append(name)
    out = []
    for x in found:
        if x not in out:
            out.append(x)
    return out


def resolve_geo(geo_name: str) -> Optional[Dict[str, str]]:
    candidates = find_geo_ids_by_name(geo_name)
    if not candidates:
        return {"state": geo_name}
    geo_id, name, geo_type = candidates[0]
    return {"geo_id": geo_id, "geo_name": name, "geo_type": geo_type}


# ---------------------------------------------------------------------------
# Structured data tools
# ---------------------------------------------------------------------------

Frequency = Literal["year", "quarter", "month"]
SortDirection = Literal["asc", "desc"]
GeoLevel = Literal["state", "county", "tribal"]


def _metric_exists(metric: str) -> bool:
    return metric in METRIC_COLUMNS


def _filter_rows_by_geo(rows, geo: Dict[str, str]):
    geo_id = (geo.get("geo_id") or "").strip()
    if geo_id:
        return [r for r in rows if r.geo_id == geo_id]

    geo_name = (geo.get("geo_name") or "").strip().lower()
    if geo_name:
        return [r for r in rows if r.geo_name.lower() == geo_name]

    state = (geo.get("state") or "").strip().lower()
    if state:
        return [r for r in rows if r.geo_type == "state" and r.geo_name.lower() == state]

    return []


def get_metric_timeseries(geo: Dict[str, str], metric: str, frequency: Frequency) -> Dict[str, Any]:
    if not _metric_exists(metric):
        return {"ok": False, "error": f"Unsupported metric '{metric}'.", "data": None}

    rows = _filter_rows_by_geo(load_metrics_rows(), geo)
    if not rows:
        return {"ok": False, "error": "No matching geography found in the dataset.", "data": None}

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

    points.sort(key=lambda x: x["period"])

    first = rows[0]
    citation = {
        "citation_type": "structured_data",
        "source_table": "metrics.csv",
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
    if not _metric_exists(metric):
        return {"ok": False, "error": f"Unsupported metric '{metric}'.", "data": None}

    try:
        start_year = int(start_period)
        end_year = int(end_period)
    except ValueError:
        return {"ok": False, "error": "compare_geos supports year periods like '2021'..'2024'.", "data": None}

    rows = load_metrics_rows()
    a_rows = [r for r in _filter_rows_by_geo(rows, geo_a) if start_year <= r.year <= end_year and r.metrics.get(metric) is not None]
    b_rows = [r for r in _filter_rows_by_geo(rows, geo_b) if start_year <= r.year <= end_year and r.metrics.get(metric) is not None]

    if not a_rows or not b_rows:
        return {"ok": False, "error": "Not enough data to compare both geographies for the requested years.", "data": None}

    def yearly_avg(rs):
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
        "source_table": "metrics.csv",
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
        "source_table": "metrics.csv",
        "metric": metric,
        "year": year,
        "geo_level": geo_level,
        "sort_direction": sort_direction,
    }

    return {"ok": True, "data": {"metric": metric, "year": year, "geo_level": geo_level, "ranked": ranked}, "citation": citation}


def get_risk_profile(geo: Dict[str, str], year: int) -> Dict[str, Any]:
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
            "source_table": "metrics.csv",
            "geo_id": r0.geo_id,
            "geo_name": r0.geo_name,
            "geo_type": r0.geo_type,
            "metric": "risk_index",
            "year": year,
        },
        {
            "citation_type": "structured_data",
            "source_table": "risk_components.csv",
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


# ---------------------------------------------------------------------------
# Knowledge base retriever
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


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]


def _chunk_markdown(body: str) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n+", body or "") if p.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0
    for p in paras:
        if buf_len + len(p) > 900 and buf:
            chunks.append("\n\n".join(buf).strip())
            buf = []
            buf_len = 0
        buf.append(p)
        buf_len += len(p)
    if buf:
        chunks.append("\n\n".join(buf).strip())
    return chunks


def retrieve(query: str, top_k: int = 4) -> List[RetrievedChunk]:
    """
    Retrieve KB chunks for a query.

    Default: keyword-overlap ranking.
    If OPENAI_API_KEY is set: semantic ranking using embeddings (more conversational).
    """
    query = (query or "").strip()
    if not query:
        return []

    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if api_key:
        try:
            return _retrieve_semantic(query, top_k=top_k, api_key=api_key)
        except Exception:
            # Fail closed to keyword retrieval
            pass
    return _retrieve_keyword(query, top_k=top_k)


def _kb_all_chunks() -> List[RetrievedChunk]:
    chunks_out: List[RetrievedChunk] = []
    for path in sorted(KB_DIR.glob("*.md")):
        md_text = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(md_text)
        chunks = _chunk_markdown(body)
        for idx, chunk_text in enumerate(chunks):
            chunks_out.append(
                RetrievedChunk(
                    chunk_id=f"{path.stem}::p{idx+1}",
                    text=chunk_text,
                    score=0.0,
                    metadata={**meta, "source_file": path.name},
                )
            )
    return chunks_out


def _retrieve_keyword(query: str, top_k: int = 4) -> List[RetrievedChunk]:
    q_tokens = set(_tokenize(query))
    if not q_tokens:
        return []

    results: List[RetrievedChunk] = []
    for ch in _kb_all_chunks():
        c_tokens = set(_tokenize(ch.text))
        overlap = q_tokens.intersection(c_tokens)
        if not overlap:
            continue
        bonus = 0.0
        tags = ch.metadata.get("metric_tags", [])
        if isinstance(tags, list):
            for t in tags:
                if str(t).lower() in q_tokens:
                    bonus += 0.25
        score = float(len(overlap)) + bonus
        results.append(RetrievedChunk(chunk_id=ch.chunk_id, text=ch.text, score=score, metadata=ch.metadata))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[: max(0, top_k)]


def _embed_texts_openai(texts: List[str], *, api_key: str, model: str = "text-embedding-3-small") -> List[List[float]]:
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def _retrieve_semantic(query: str, top_k: int, api_key: str) -> List[RetrievedChunk]:
    import numpy as np

    cache_key = "kb_semantic_index_v1"
    index = _CACHE.get(cache_key)
    if not index:
        chunks = _kb_all_chunks()
        embs = _embed_texts_openai([c.text for c in chunks], api_key=api_key)
        mat = np.array(embs, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms
        index = {"chunks": chunks, "mat": mat}
        _CACHE[cache_key] = index

    chunks = index["chunks"]
    mat = index["mat"]

    q_emb = _embed_texts_openai([query], api_key=api_key)[0]
    q = np.array(q_emb, dtype=np.float32)
    qn = np.linalg.norm(q)
    if not np.isfinite(qn) or qn == 0:
        return _retrieve_keyword(query, top_k=top_k)
    q = q / qn

    sims = mat @ q
    k = max(0, int(top_k))
    if k == 0 or len(chunks) == 0:
        return []
    idxs = np.argsort(-sims)[:k]

    out: List[RetrievedChunk] = []
    for i in idxs:
        ch = chunks[int(i)]
        out.append(RetrievedChunk(chunk_id=ch.chunk_id, text=ch.text, score=float(sims[int(i)]), metadata=ch.metadata))
    return out


# ---------------------------------------------------------------------------
# Intent
# ---------------------------------------------------------------------------

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

    if re.search(r"\bwhat does vawa\b|\bwhat is vawa\b|\bwhat changed\b", text):
        return "docs_only"

    if has_metric and has_docs:
        return "data_and_docs"
    if has_metric:
        return "data_only"
    if has_docs:
        return "docs_only"

    return "docs_only"


def _arcgis_dashboard_url() -> str:
    u = (os.environ.get("ARCGIS_DASHBOARD_URL") or "").strip()
    return u or DEFAULT_ARCGIS_DASHBOARD_URL


def _should_attach_arcgis_map(
    message: str,
    metric: Optional[str],
    geo_names: List[str],
    tools_used: List[Dict[str, Any]],
) -> bool:
    t = (message or "").lower()
    if re.search(r"\bmap\b|\bgis\b|\bgeographic\b|\bchoropleth\b", t):
        return True
    tool_names = {x.get("tool") for x in tools_used if isinstance(x, dict)}
    geo_tools = {
        "compare_geos",
        "rank_geos",
        "get_metric_timeseries",
        "get_risk_profile",
        "rank_states_by_policy_change",
        "get_policy_variable_summary",
    }
    if tool_names & geo_tools:
        return True
    if len(geo_names) >= 2 and metric is not None and "compare" in t:
        return True
    return False


def _build_map_embed(
    message: str,
    metric: Optional[str],
    geo_names: List[str],
    tools_used: List[Dict[str, Any]],
) -> Optional[MapEmbedPayload]:
    if not _should_attach_arcgis_map(message, metric, geo_names, tools_used):
        return None
    url = _arcgis_dashboard_url()
    states = geo_names[:8]
    m_label = _METRIC_LABELS.get(metric or "", None) or (
        metric.replace("_", " ") if metric else None
    )
    bits: List[str] = []
    if m_label:
        bits.append(f"In the dashboard, select a layer that matches: {m_label}.")
    if states:
        bits.append(f"Your question references: {', '.join(states)}.")
    bits.append(
        "The embed shows the full dashboard; use its map and layer list to focus areas of interest. "
        "Auto-zoom to specific states needs URL parameters enabled on the dashboard (ArcGIS Dashboards)."
    )
    return MapEmbedPayload(
        show=True,
        title="Related ArcGIS dashboard",
        embed_url=url,
        open_url=url,
        caption=" ".join(bits),
        states=states,
        metric=metric,
        metric_label=m_label,
    )


# ---------------------------------------------------------------------------
# Chat entrypoint
# ---------------------------------------------------------------------------


def _doc_citation_from_chunk(chunk: RetrievedChunk) -> Dict[str, Any]:
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
    llm_debug: Dict[str, Any] = {"enabled": False}

    metric = detect_metric(message)
    years = extract_years(message)
    geo_names = detect_geo_names(message)

    evidence_lines: List[str] = []
    caveats: List[str] = []
    interpretation = ""
    direct_answer = ""

    descriptive_caveat = "These are descriptive patterns from observed data and do not, by themselves, establish causation."
    policy_caveat = "Pre/post-2022 comparisons are descriptive; changes may reflect reporting/coverage shifts and do not, by themselves, establish policy effects."

    policy_var = detect_policy_variable(message)
    asks_policy_change = any(
        k in message.lower()
        for k in [
            "post_2022",
            "post-2022",
            "pre_2022",
            "pre-2022",
            "percent change",
            "% change",
            "change after 2022",
            "after 2022",
            "before 2022",
            "pre vs",
            "pre/post",
        ]
    )

    if intent in {"data_only", "data_and_docs"}:
        policy_handled = False
        if policy_var and asks_policy_change:
            if any(k in message.lower() for k in ["highest", "top", "rank", "most"]) and "state" in message.lower():
                dirn: Literal["increase", "decrease"] = "increase"
                if any(k in message.lower() for k in ["decrease", "decline", "dropped", "lowest"]):
                    dirn = "decrease"

                # Baseline filter for counts to avoid extreme percent-change blowups.
                vmeta = get_policy_variable_summary(policy_var)
                vtype = ""
                if vmeta.get("ok"):
                    vtype = str(vmeta.get("data", {}).get("variable_type", "")).strip().lower()
                min_pre = 500.0 if vtype == "count" else 0.01

                tool_out = rank_states_by_policy_change(policy_var, dirn, top_n=5, min_pre_value=min_pre)
                tools_used.append({"tool": "rank_states_by_policy_change", "args": {"variable": policy_var, "direction": dirn, "top_n": 5, "min_pre_value": min_pre}, "ok": tool_out.get("ok")})
                if tool_out.get("ok"):
                    ranked = tool_out["data"]["ranked"]
                    direct_answer = (
                        f"Top states by percent change in {policy_var} ({dirn}, pre vs post 2022). "
                        f"(Filtered to stable baselines: pre_2022 ≥ {min_pre:g}): "
                        + ", ".join([f"{r['state']} ({r['percent_change']:.2%})" for r in ranked])
                    )
                    evidence_lines.append(f"Variable: {policy_var} (pre_2022 vs post_2022_avg).")
                    for r in ranked:
                        evidence_lines.append(
                            f"- #{r['rank']} {r['state']}: percent_change={r['percent_change']:.2%}, absolute_change={r.get('absolute_change')}, pre_2022={r.get('pre_2022')}, post_2022_avg={r.get('post_2022_avg')}"
                        )
                    citations.append(tool_out["citation"])
                    caveats.append(policy_caveat)
                else:
                    direct_answer = f"I don’t have enough policy-summary data to rank that. ({tool_out.get('error')})"
                policy_handled = True
            else:
                tool_out = get_policy_variable_summary(policy_var)
                tools_used.append({"tool": "get_policy_variable_summary", "args": {"variable": policy_var}, "ok": tool_out.get("ok")})
                if tool_out.get("ok"):
                    d = tool_out["data"]
                    mpc = d.get("mean_percent_change")
                    medpc = d.get("median_percent_change")
                    inc = d.get("states_increased")
                    dec = d.get("states_decreased")
                    direct_answer = f"Across states, {policy_var} changed post-2022 vs pre-2022 (summary)."
                    if mpc is not None:
                        evidence_lines.append(f"Mean percent change: {float(mpc):.2%}")
                    if medpc is not None:
                        evidence_lines.append(f"Median percent change: {float(medpc):.2%}")
                    if inc is not None and dec is not None:
                        evidence_lines.append(f"States increased: {inc}, decreased: {dec}")
                    citations.append(tool_out["citation"])
                    caveats.append(policy_caveat)
                else:
                    direct_answer = f"I don’t have a policy-summary entry for that variable yet. ({tool_out.get('error')})"
                policy_handled = True

        if not policy_handled and ("risk profile" in message.lower() or (metric == "risk_index" and "profile" in message.lower())):
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

        elif not policy_handled and any(k in message.lower() for k in ["highest", "top", "rank"]) and ("state" in message.lower() or "states" in message.lower()):
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

        elif not policy_handled and "compare" in message.lower() and len(geo_names) >= 2 and metric is not None and len(years) >= 1:
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

        elif not policy_handled and metric is not None and geo_names:
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
            if not policy_handled:
                direct_answer = "I couldn’t identify a supported metric and geography to answer with numbers. Try including a metric (e.g., dv_rate) and a place (e.g., California)."

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
            if direct_answer == "":
                direct_answer = "Here’s what the knowledge base says:"
            interpretation_bits = []
            for ch in chunks[:2]:
                snippet = ch.text.strip().replace("\n", " ")
                if len(snippet) > 280:
                    snippet = snippet[:280].rstrip() + "…"
                interpretation_bits.append(f"- {snippet}")
            interpretation = "\n".join(interpretation_bits)
        else:
            if intent == "docs_only":
                direct_answer = "I don’t have enough knowledge-base content to answer that yet."
            caveats.append("Knowledge base retrieval returned no matching documents for this query.")

    if any(w in message.lower() for w in ["cause", "caused", "because", "led to", "impact", "effect"]):
        caveats.append("This bot can describe trends/associations in the available data, but it does not establish causal effects.")

    if not citations:
        citations.append(
            {
                "citation_type": "system",
                "citation_id": "NO-SOURCE",
                "title": "No matching structured rows or KB chunks found",
            }
        )

    if interpretation == "":
        interpretation = "Interpretation is limited. If you want, ask for caveats/definitions and I’ll pull from the methodology documents."

    # ---- Optional LLM writer step (minimal integration) ----
    # If OPENAI_API_KEY is not set, we keep deterministic behavior.
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if api_key:
        try:
            # Imported lazily so the backend still runs without the dependency
            # if the user doesn't install it.
            from openai import OpenAI  # type: ignore

            client = OpenAI(api_key=api_key)
            llm_debug = {"enabled": True, "model": "gpt-4o-mini", "used": False}

            # Keep prompt small: we provide the user message, the computed evidence,
            # and short excerpts from retrieved chunks.
            excerpt_chunks: List[Dict[str, Any]] = []
            if "chunks" in locals() and isinstance(locals().get("chunks"), list):
                for ch in locals()["chunks"][:4]:
                    txt = (ch.text or "").strip().replace("\n", " ")
                    if len(txt) > 500:
                        txt = txt[:500].rstrip() + "…"
                    excerpt_chunks.append(
                        {
                            "chunk_id": ch.chunk_id,
                            "citation_id": ch.metadata.get("citation_id", ch.chunk_id),
                            "title": ch.metadata.get("title", ch.metadata.get("source_file", "")),
                            "text_excerpt": txt,
                        }
                    )

            writer_payload = {
                "user_message": message,
                "intent": intent,
                "direct_answer_so_far": direct_answer,
                "evidence_lines": evidence_lines,
                "caveats": caveats,
                "retrieved_chunks": excerpt_chunks,
                "citations": citations,
                "constraints": [
                    "Do not invent numbers. If you include any numeric value, it must appear verbatim in evidence_lines or retrieved_chunks.",
                    "Do not claim causation. Use associational language unless the retrieved text explicitly states causal identification (rare).",
                    "If you reference a knowledge-base statement, include its citation_id in parentheses.",
                    "Keep it concise (2–6 sentences).",
                ],
            }

            system = (
                "You are a careful analyst writing a short, grounded response for a VAWA insights system. "
                "You must follow the constraints exactly."
            )

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": "Write an improved direct answer + interpretation as JSON with keys "
                        '"direct_answer" and "interpretation" only.\n\nINPUT:\n'
                        + json.dumps(writer_payload, ensure_ascii=False),
                    },
                ],
            )

            content = (resp.choices[0].message.content or "").strip()
            # Robust-ish JSON extraction (model should return pure JSON, but guard anyway)
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                content = content[start : end + 1]
            out = json.loads(content)
            if isinstance(out, dict):
                da = out.get("direct_answer")
                interp = out.get("interpretation")
                if isinstance(da, str) and da.strip():
                    direct_answer = da.strip()
                    llm_debug["used"] = True
                if isinstance(interp, str) and interp.strip():
                    interpretation = interp.strip()
                    llm_debug["used"] = True
        except Exception:
            # Fail closed: keep deterministic output if the LLM step fails.
            llm_debug = {"enabled": True, "model": "gpt-4o-mini", "used": False, "error": "llm_call_failed"}

    map_embed = _build_map_embed(message, metric, geo_names, tools_used)

    return ChatResponse(
        answer={
            "direct_answer": direct_answer,
            "evidence": evidence_lines,
            "interpretation": interpretation,
            "caveats": caveats,
            "citations": citations,
            "map_embed": map_embed,
        },
        debug={
            "intent": intent,
            "tools_used": tools_used,
            "docs_retrieved": docs_retrieved,
            "llm": llm_debug,
        },
    )
