"""
Lightweight document retriever for V1 (no embeddings, no vector DB).

How it works:
- Loads markdown docs from app/kb/
- Parses simple frontmatter metadata blocks
- Chunks by paragraphs
- Scores chunks by keyword overlap with the query

Later upgrades:
- Replace scoring with embeddings + FAISS/Chroma
- Add better chunking (sentence split, windowing)
- Add query expansion / reranking
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


KB_DIR = Path(__file__).resolve().parent.parent / "kb"


_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def _parse_frontmatter(md_text: str) -> Tuple[Dict[str, Any], str]:
    """
    Parses a very small YAML-like frontmatter:

    ---
    key: value
    key2: a,b,c
    ---
    body...
    """
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
            # Basic list support: "a, b, c"
            if "," in v:
                meta[k] = [x.strip() for x in v.split(",") if x.strip()]
            else:
                meta[k] = v
        i += 1

    # If malformed, treat whole thing as body.
    return {}, md_text


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]


def _chunk_markdown(body: str) -> List[str]:
    """
    Simple chunking: split into paragraphs, keep medium-sized chunks.
    """
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
    Returns top_k chunks with metadata and a simple overlap score.
    """
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
            # Small bonus for explicit metric tags in metadata.
            bonus = 0.0
            tags = meta.get("metric_tags", [])
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

