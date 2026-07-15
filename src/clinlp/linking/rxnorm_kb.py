"""Workstream A: build `data/RxNorm/rxnorm_processed.csv` — RxNorm had no local
data at all before this module (see `Plan.md`'s "Lấp gap RxNorm").

Two acquisition paths were planned in `Plan.md`, run "song song" (in parallel):

- **Path 1 (fuller, not built here):** NLM UTS account (free, manual signup) ->
  RxNorm Full Monthly Release -> `RXNCONSO.RRF`, filtered to English + the
  relevant term types. ~100K+ concepts, includes full historical/remapped
  RxCUIs. Left as documented follow-up work — it needs a human to register a
  UTS account, so it can't be scripted end-to-end here.
- **Path 2 (implemented here):** the public RxNav REST `allconcepts` bulk
  endpoint (`GET /REST/allconcepts.json?tty=<TTY>`), no auth required, fetched
  per term-type (`TERM_TYPES` below) rather than per drug name — confirmed
  live against the real API while building this module (~52K concepts total
  across the 7 term types; see the network-call log kept in
  `data/RxNorm/raw_rest_cache/*.json`).

**Known gap in Path 2, found empirically while building this module:**
`allconcepts` only returns *currently active* concepts. Two RxCUIs from
`Topic.md`'s own worked example — 360047 ("Chlorpheniramine 0.4 MG/ML...")
and 1660761 ("Capsaicin 0.38 MG/ML...") — are historical (`Remapped` /
`Obsolete` per `/rxcui/{rxcui}/historystatus.json`, verified live) and
therefore ABSENT from the bulk endpoint no matter how it's queried. Path 1's
full RRF release would include them (it carries history); Path 2 structurally
cannot, since you would already need to know the RxCUI to ask about it — bulk
discovery by term-type only surfaces the current set.

To keep the organizer's own two worked-example codes resolvable without
fabricating data, `build_rxnorm_kb()` also fetches those two *specific,
known* RxCUIs via the targeted `historystatus.json` endpoint (still one-time,
still offline/build-only network access — never called from `link.py`) and
tags them `source=historical_supplement` in the output CSV so their
provenance is explicit and auditable. This does not generalize to arbitrary
drug names (there's no bulk "give me all historical concepts" endpoint on
this API) — it is a two-row patch for the organizer's own reference example.

Every other row in the CSV comes straight from the Path-2 bulk fetch and is
tagged `source=allconcepts_bulk`.

Only this module (and a future thin `scripts/build_rxnorm_index.py` wrapper,
if one is added) may perform network I/O. `linking/link.py` must never import
`requests`/`urllib` — see CLAUDE.md's "Critical constraints" (self-hosted
only, no external API calls, at inference time).
"""
from __future__ import annotations

import csv
import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

RXNAV_BASE = "https://rxnav.nlm.nih.gov/REST"
TERM_TYPES: tuple[str, ...] = ("IN", "PIN", "SCD", "SBD", "BN", "GPCK", "BPCK")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_CACHE_DIR = REPO_ROOT / "data" / "RxNorm" / "raw_rest_cache"
DEFAULT_OUT_CSV = REPO_ROOT / "data" / "RxNorm" / "rxnorm_processed.csv"

# See module docstring: these two organizer-referenced RxCUIs are historical
# and absent from the current-only bulk `allconcepts` endpoint, so they are
# fetched individually as a documented, auditable exception.
HISTORICAL_SUPPLEMENT_RXCUIS: tuple[str, ...] = ("360047", "1660761")

_USER_AGENT = "clinlp-kb-builder/0.1 (Viettel clinical NLP competition, offline KB build)"


def _http_get_json(url: str, timeout: float = 30.0, retries: int = 3, retry_wait: float = 2.0) -> dict:
    """Shared retry wrapper for the (small number of) network calls this
    module makes. Raises after exhausting retries rather than returning an
    empty/fake result — a failed fetch must be visible, never silently
    treated as "zero concepts of this type"."""
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            logger.warning("GET %s attempt %d/%d failed: %s", url, attempt, retries, e)
            if attempt < retries:
                time.sleep(retry_wait)
    raise RuntimeError(f"failed to GET {url} after {retries} attempts") from last_err


def fetch_tty(tty: str, **kwargs) -> dict:
    """Fetch the raw `allconcepts.json` response for one term-type."""
    return _http_get_json(f"{RXNAV_BASE}/allconcepts.json?tty={tty}", **kwargs)


def fetch_all_ttys(
    term_types: Iterable[str] = TERM_TYPES,
    cache_dir: str | Path = DEFAULT_RAW_CACHE_DIR,
    use_cache: bool = True,
) -> dict[str, dict]:
    """Fetch (or load from cache) the raw per-term-type JSON. Always writes
    (or refreshes) the cache file so a rebuild is reproducible from
    `data/RxNorm/raw_rest_cache/` without re-hitting the network — this is
    the artifact that lets `build_rxnorm_kb()` be re-run offline."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for tty in term_types:
        cache_path = cache_dir / f"{tty}.json"
        if use_cache and cache_path.exists():
            with open(cache_path, encoding="utf-8") as fh:
                results[tty] = json.load(fh)
            continue
        data = fetch_tty(tty)
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        results[tty] = data
    return results


def fetch_historical_supplement(
    rxcuis: Iterable[str] = HISTORICAL_SUPPLEMENT_RXCUIS,
    cache_dir: str | Path = DEFAULT_RAW_CACHE_DIR,
    use_cache: bool = True,
) -> list[dict]:
    """Targeted lookup for the small, hardcoded list of historical RxCUIs
    documented in the module docstring. Returns rows with an extra
    `status` field (e.g. "Remapped"/"Obsolete") preserved for transparency."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for rxcui in rxcuis:
        cache_path = cache_dir / f"historical_{rxcui}.json"
        if use_cache and cache_path.exists():
            with open(cache_path, encoding="utf-8") as fh:
                data = json.load(fh)
        else:
            data = _http_get_json(f"{RXNAV_BASE}/rxcui/{rxcui}/historystatus.json")
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
        attrs = data.get("rxcuiStatusHistory", {}).get("attributes", {})
        status = data.get("rxcuiStatusHistory", {}).get("metaData", {}).get("status", "")
        if not attrs.get("rxcui"):
            logger.warning("historystatus lookup for rxcui=%s returned no attributes", rxcui)
            continue
        rows.append(
            {
                "rxcui": attrs.get("rxcui", rxcui),
                "name": attrs.get("name", ""),
                "tty": attrs.get("tty", ""),
                "source": "historical_supplement",
                "status": status,
            }
        )
    return rows


def _extract_bulk_rows(raw: dict, tty: str) -> list[dict]:
    concepts = raw.get("minConceptGroup", {}).get("minConcept", []) or []
    return [
        {
            "rxcui": c.get("rxcui", ""),
            "name": c.get("name", ""),
            "tty": c.get("tty", tty),
            "source": "allconcepts_bulk",
            "status": "",
        }
        for c in concepts
    ]


def build_rxnorm_kb(
    out_path: str | Path = DEFAULT_OUT_CSV,
    cache_dir: str | Path = DEFAULT_RAW_CACHE_DIR,
    term_types: Iterable[str] = TERM_TYPES,
    use_cache: bool = True,
    include_historical_supplement: bool = True,
) -> int:
    """Build `rxnorm_processed.csv` from the Path-2 bulk fetch, plus the
    documented historical-supplement patch. Returns the row count written."""
    raw_by_tty = fetch_all_ttys(term_types, cache_dir=cache_dir, use_cache=use_cache)

    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for tty, raw in raw_by_tty.items():
        for row in _extract_bulk_rows(raw, tty):
            key = (row["rxcui"], row["tty"])
            if key in seen or not row["rxcui"]:
                continue
            seen.add(key)
            rows.append(row)

    if include_historical_supplement:
        for row in fetch_historical_supplement(cache_dir=cache_dir, use_cache=use_cache):
            key = (row["rxcui"], row["tty"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["rxcui", "name", "tty", "source", "status"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def load_rxnorm_kb(path: str | Path = DEFAULT_OUT_CSV) -> list[dict]:
    """Load the already-built `rxnorm_processed.csv`. Returns `[]` (not an
    error) if it hasn't been built yet, so `linking/link.py` degrades
    gracefully rather than crashing at import time."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


if __name__ == "__main__":
    n = build_rxnorm_kb()
    print(f"wrote {n} RxNorm rows to {DEFAULT_OUT_CSV}")
