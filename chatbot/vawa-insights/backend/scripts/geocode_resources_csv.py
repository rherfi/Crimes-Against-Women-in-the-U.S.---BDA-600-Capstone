"""
Geocode missing latitude/longitude in resources.csv (in place).

Uses OpenStreetMap Nominatim (free) with:
- disk cache (so reruns don't re-query)
- conservative rate limiting
- resume support via --max (run in batches)

IMPORTANT:
Nominatim has usage policies. Keep requests low and cache results.

Usage:
  python scripts/geocode_resources_csv.py
  python scripts/geocode_resources_csv.py --max 100
  python scripts/geocode_resources_csv.py --sleep-seconds 1.2
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests


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


def _is_missing(x: str) -> bool:
    return (x or "").strip() == ""


def _build_query(row: Dict[str, str]) -> str:
    parts = []
    for k in ["address", "city", "state", "postal_code", "country"]:
        v = (row.get(k) or "").strip()
        if v:
            parts.append(v)
    if not parts:
        return ""
    # Add org name as a hint (helps with PO boxes / shared addresses)
    nm = (row.get("name") or "").strip()
    if nm:
        return f"{nm}, " + ", ".join(parts)
    return ", ".join(parts)


def _build_candidate_queries(row: Dict[str, str]) -> List[str]:
    """
    Build multiple candidate geocoding queries (most specific first).
    This increases hit rate for rows like PO Boxes or shared org names.
    """
    addr = (row.get("address") or "").strip()
    city = (row.get("city") or "").strip()
    state = (row.get("state") or "").strip()
    postal = (row.get("postal_code") or "").strip()
    country = (row.get("country") or "").strip()
    name = (row.get("name") or "").strip()

    parts = [p for p in [addr, city, state, postal, country] if p]
    place = ", ".join(parts)
    place_no_name = ", ".join([p for p in [addr, city, state, postal] if p])
    city_state = ", ".join([p for p in [city, state, postal, country] if p])

    out: List[str] = []
    if name and place:
        out.append(f"{name}, {place}")
    if place_no_name:
        out.append(place_no_name)
    if city_state:
        out.append(city_state)
    # de-dupe while preserving order
    seen = set()
    uniq: List[str] = []
    for q in out:
        k = q.lower().strip()
        if not k or k in seen:
            continue
        seen.add(k)
        uniq.append(q)
    return uniq


def _load_cache(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(path: Path, cache: Dict[str, Dict[str, str]]) -> None:
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def geocode_nominatim(q: str, *, user_agent: str, timeout_s: int = 15) -> Tuple[str, str, str]:
    """
    Returns (lat, lon, display_name) as strings.
    Raises on errors / no hits.
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": q, "format": "json", "limit": 1, "addressdetails": 0}
    headers = {"User-Agent": user_agent}
    r = requests.get(url, params=params, headers=headers, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list) or not data:
        raise ValueError("no_results")
    hit = data[0]
    lat = str(hit.get("lat") or "").strip()
    lon = str(hit.get("lon") or "").strip()
    disp = str(hit.get("display_name") or "").strip()
    if not lat or not lon:
        raise ValueError("missing_latlon")
    return lat, lon, disp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="app/data/resources.csv")
    ap.add_argument("--cache", default="app/data/resources_geocode_cache.json")
    ap.add_argument("--max", type=int, default=150, help="Max rows to geocode this run (batching).")
    ap.add_argument("--sleep-seconds", type=float, default=1.05, help="Sleep between requests.")
    ap.add_argument("--user-agent", default="vawa-insights-bot/0.1 (educational project)")
    ap.add_argument("--state", default="", help="Only geocode rows matching this state (e.g., CA).")
    ap.add_argument("--city", default="", help="Only geocode rows matching this city (case-insensitive).")
    ap.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry cached error entries (otherwise only cached successes are reused).",
    )
    args = ap.parse_args()

    csv_path = Path(args.path)
    cache_path = Path(args.cache)

    if not csv_path.exists():
        raise SystemExit(f"Missing file: {csv_path}")

    cache = _load_cache(cache_path)

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    did = 0
    updated = 0
    for row in rows:
        if did >= max(0, int(args.max)):
            break
        rid = (row.get("resource_id") or "").strip()
        if not rid:
            continue

        if args.state and (row.get("state") or "").strip().lower() != args.state.strip().lower():
            continue
        if args.city and (row.get("city") or "").strip().lower() != args.city.strip().lower():
            continue

        if not (_is_missing(row.get("latitude", "")) or _is_missing(row.get("longitude", ""))):
            continue

        candidates = _build_candidate_queries(row)
        if not candidates:
            continue

        # Try candidates in order; consult cache per-candidate.
        row_updated = False
        for q in candidates:
            ck = q.lower()
            cached = cache.get(ck) or {}
            if cached.get("lat") and cached.get("lon"):
                row["latitude"] = cached["lat"]
                row["longitude"] = cached["lon"]
                updated += 1
                row_updated = True
                break
            if cached.get("error") and not args.retry_errors:
                continue

            try:
                lat, lon, disp = geocode_nominatim(q, user_agent=args.user_agent)
                cache[ck] = {"lat": lat, "lon": lon, "display_name": disp, "ts": str(int(time.time()))}
                row["latitude"] = lat
                row["longitude"] = lon
                updated += 1
                row_updated = True
                did += 1
                time.sleep(max(0.0, float(args.sleep_seconds)))
                break
            except Exception as e:
                cache[ck] = {"error": str(e), "ts": str(int(time.time()))}
                did += 1
                time.sleep(max(0.0, float(args.sleep_seconds)))

        if row_updated:
            continue

    # Write back CSV (in place)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (r.get(k, "") or "") for k in HEADER})

    _save_cache(cache_path, cache)
    print(f"updated_latlon={updated} requested={did} cache_entries={len(cache)} wrote={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

