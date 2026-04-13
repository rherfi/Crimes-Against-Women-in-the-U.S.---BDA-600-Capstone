"""
Lightweight knowledge base retriever for Victim Tool Bot (V1).

We intentionally keep this simple and local:
- Markdown docs in app/kb/
- Tiny frontmatter metadata (citation_id, title, doc_type, topic_tags)
- Chunking by paragraphs
- Keyword-overlap scoring

This is easy to swap later for embeddings + a vector index.
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
    Parses a small YAML-ish frontmatter:

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

            # Small tag bonus.
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
    """
    Convenience wrapper matching your requested function name.
    """
    return retrieve(topic, top_k=top_k)

