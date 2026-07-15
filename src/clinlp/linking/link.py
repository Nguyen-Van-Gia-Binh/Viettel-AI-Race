"""Bước 5 (Workstream A): entity linking entrypoint.

Interface locked by `pipeline.py`'s `LinkFn` type alias — do not change the
signature here without updating `pipeline.py`:

    link(entity_text, concept_type, source_text) -> list[str]

`pipeline.py` only calls this for `CHẨN_ĐOÁN`/`THUỐC` concepts (schema.py's
`CANDIDATE_ELIGIBLE_TYPES`); other types never reach this function through the
pipeline.

Baseline strategy (`Plan.md` tuần 1): exact/fuzzy string match via `rapidfuzz`
against the locally pre-built KB CSVs (`icd10_kb.build_icd10_kb()` /
`rxnorm_kb.build_rxnorm_kb()`), ranked by match score descending. No
BM25/dense retrieval/reranker yet — that's `linking/retrieval.py` +
`linking/rerank.py`, planned for tuần 2-3. `source_text` is accepted (to match
the pipeline interface) but unused for now; a future version could use
surrounding context (e.g. a nearby lab panel name) to disambiguate ties.

HARD INVARIANT (CLAUDE.md "Critical constraints", competition rule): this
module must never import `requests`/`urllib`, directly or transitively at
call time — the model/pipeline at inference time must never make an external
network call. All network access for entity linking happens once, offline, in
`icd10_kb.py`/`rxnorm_kb.py`'s `build_*_kb()` functions. This is meant to be
grep-checkable: `grep -E "requests|urllib" link.py` must find nothing.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from rapidfuzz import fuzz, process

from clinlp.linking.icd10_kb import DEFAULT_OUT_CSV as ICD10_CSV
from clinlp.linking.icd10_kb import load_icd10_kb
from clinlp.linking.rxnorm_kb import DEFAULT_OUT_CSV as RXNORM_CSV
from clinlp.linking.rxnorm_kb import load_rxnorm_kb
from clinlp.schema import ConceptType

DEFAULT_TOP_K = 5
# rapidfuzz scores are 0-100. These cutoffs are deliberately loose (Vietnamese
# CHẨN_ĐOÁN text vs an English/vi_aliases KB, or a THUỐC mention with a
# dose/route the KB entry doesn't share verbatim, both suppress the raw
# score) — a baseline fuzzy matcher trades some precision for not returning
# nothing. Revisit once retrieval.py/rerank.py exist.
MIN_SCORE_ICD10 = 40.0
MIN_SCORE_RXNORM = 45.0


@lru_cache(maxsize=1)
def _icd10_choices(csv_path: str = str(ICD10_CSV)) -> dict[str, str]:
    """code -> single searchable string (short/long description + any seed
    Vietnamese aliases, joined). One dict entry per code so rapidfuzz's
    dict-mode `process.extract` can return codes directly as match keys."""
    rows = load_icd10_kb(csv_path)
    choices: dict[str, str] = {}
    for row in rows:
        parts = [row.get("short_description", ""), row.get("long_description", "")]
        aliases = row.get("vi_aliases", "")
        if aliases:
            parts.extend(a for a in aliases.split(";") if a)
        text = " | ".join(p for p in parts if p)
        if text:
            choices[row["code"]] = text
    return choices


@lru_cache(maxsize=1)
def _rxnorm_choices(csv_path: str = str(RXNORM_CSV)) -> dict[str, str]:
    """rxcui -> RxNorm name string."""
    rows = load_rxnorm_kb(csv_path)
    choices: dict[str, str] = {}
    for row in rows:
        name = row.get("name", "")
        rxcui = row.get("rxcui", "")
        if name and rxcui:
            choices[rxcui] = name
    return choices


def _rank_matches(entity_text: str, choices: dict[str, str], top_k: int, min_score: float) -> list[str]:
    if not entity_text or not choices:
        return []
    # partial_ratio (best matching substring), not WRatio: KB entries are
    # long dosage-form strings (e.g. "capsaicin 0.38 MG/ML / menthol 40
    # MG/ML / methyl salicylate 350 MG/ML Topical Cream") while entity_text
    # is typically just the drug/diagnosis's own short name-and-dose prefix
    # of that string. Empirically verified against Topic.md's worked
    # examples: WRatio scored dozens of unrelated same-length KB rows in a
    # tie with the correct one (e.g. RxCUI 1660761 for "Capsaicin 0.38
    # MG/ML" fell out of the top 40 entirely), while partial_ratio correctly
    # ranks the true match #1 in every case tested (both RxNorm and ICD-10).
    # process.extract on a dict returns (matched_string, score, key) tuples,
    # already sorted by score descending.
    results = process.extract(entity_text, choices, scorer=fuzz.partial_ratio, limit=top_k)
    return [key for _matched_text, score, key in results if score >= min_score]


def link(
    entity_text: str,
    concept_type: ConceptType,
    source_text: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[str]:
    """Return candidate codes for `entity_text`, ranked most-likely-first.

    Empty list means "no confident match" (e.g. score below cutoff, or the KB
    CSV hasn't been built yet) — never a guessed/placeholder code.
    """
    entity_text = (entity_text or "").strip()
    if not entity_text:
        return []

    if concept_type == ConceptType.CHAN_DOAN:
        return _rank_matches(entity_text, _icd10_choices(), top_k, MIN_SCORE_ICD10)
    if concept_type == ConceptType.THUOC:
        return _rank_matches(entity_text, _rxnorm_choices(), top_k, MIN_SCORE_RXNORM)
    return []
