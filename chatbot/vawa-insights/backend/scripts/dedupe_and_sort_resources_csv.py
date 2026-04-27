"""
Dedupe + alphabetize resources.csv (in place).

This script is intentionally dependency-free and robust to common CSV issues:
- blank spacer lines
- repeated header rows in the middle of the file
- rows with more/fewer columns than expected (extras are merged into the last column)

Usage:
  python scripts/dedupe_and_sort_resources_csv.py
  python scripts/dedupe_and_sort_resources_csv.py --path app/data/resources.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from io import StringIO
from pathlib import Path
from typing import Dict, List, Tuple


HEADER: List[str] = [
    "resource_id",
    "name",
    "category",
    "subcategory",
    "services",
    "address",
    "city",
    "state",
    "postal_code",
    "country",
    "phone",
    "website",
    "latitude",
    "longitude",
    "notes",
    "source",
]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _signature(row: Dict[str, str]) -> Tuple[str, str, str, str, str, str]:
    """
    A conservative “same org” signature (to catch duplicates even with different IDs).
    """

    return (
        _norm(row.get("name", "")),
        _norm(row.get("phone", "")),
        _norm(row.get("website", "")),
        _norm(row.get("address", "")),
        _norm(row.get("city", "")),
        _norm(row.get("state", "")),
    )


def _read_rows_robust(path: Path) -> List[Dict[str, str]]:
    raw_text = path.read_text(encoding="utf-8")
    header_line = ",".join(HEADER)

    lines = [ln for ln in raw_text.splitlines() if ln.strip()]
    lines = [ln for ln in lines if ln.strip() != header_line]  # drop repeated headers

    rows: List[Dict[str, str]] = []
    for ln in lines:
        vals = next(csv.reader([ln]))
        if len(vals) < len(HEADER):
            vals = vals + [""] * (len(HEADER) - len(vals))
        elif len(vals) > len(HEADER):
            # Merge extras into last column so we never “lose” text.
            vals = vals[: len(HEADER) - 1] + [",".join(vals[len(HEADER) - 1 :])]
        rows.append({k: (v or "") for k, v in zip(HEADER, vals)})
    return rows


def dedupe_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen_ids = set()
    seen_sigs = set()
    kept: List[Dict[str, str]] = []

    for row in rows:
        rid = (row.get("resource_id") or "").strip()
        if not rid or rid.lower() == "resource_id":
            continue

        # Drop ALT variants when base exists.
        base = rid[:-4] if rid.endswith("-ALT") else rid
        if rid.endswith("-ALT") and base in seen_ids:
            continue

        sig = _signature(row)
        if rid in seen_ids:
            continue
        if sig in seen_sigs:
            continue

        seen_ids.add(rid)
        seen_sigs.add(sig)
        kept.append(row)

    return kept


def sort_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    # “Alphabetize all entries” – primary sort by name, then state, then city.
    return sorted(
        rows,
        key=lambda r: (
            _norm(r.get("name", "")),
            _norm(r.get("state", "")),
            _norm(r.get("city", "")),
            _norm(r.get("category", "")),
        ),
    )


def write_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    out = StringIO()
    w = csv.DictWriter(out, fieldnames=HEADER, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    w.writeheader()
    for r in rows:
        w.writerow({k: (r.get(k, "") or "") for k in HEADER})

    path.write_text(out.getvalue(), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="app/data/resources.csv", help="Path to resources.csv (relative to backend/)")
    args = parser.parse_args()

    p = Path(args.path)
    if not p.exists():
        raise SystemExit(f"File not found: {p}")

    rows = _read_rows_robust(p)
    deduped = dedupe_rows(rows)
    sorted_rows = sort_rows(deduped)
    write_rows(p, sorted_rows)

    print(f"wrote {p} (in={len(rows)} kept={len(sorted_rows)} dropped={len(rows) - len(sorted_rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

