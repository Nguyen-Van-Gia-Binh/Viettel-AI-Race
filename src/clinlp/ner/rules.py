"""Bước 3 (Workstream B, Week 1): rule-based NER — regex + dictionary
matching over the raw source text.

INVARIANT (same as position.py / CLAUDE.md): match against the *raw* source
text, never a normalized copy (`preprocessing.normalize_for_llm`) — offsets
found here feed straight into assertion-window calculations in
`ner/assertions.py`, and must line up with the same raw file `position.py`
resolves against later in the pipeline.

This module only finds spans + types. Assertion detection is
`ner/assertions.py`'s job; wiring both together into something that matches
`pipeline.NerFn` is `ner/pipeline_adapter.py`'s job.
"""
from __future__ import annotations

import re
from typing import Iterator

from clinlp.ner.dictionaries import (
    DIAGNOSIS_TERMS,
    DRUG_TERMS,
    LAB_TEST_TERMS,
    SYMPTOM_TERMS,
)
from clinlp.schema import ConceptType

# start, end, matched_text (exact substring of the source, case as written —
# dictionary matching is case-insensitive but we always yield what's
# actually in the source, never the canonical dictionary form)
Match = tuple[int, int, str]

# (start, end, matched_text, type)
RawTypedMatch = tuple[int, int, str, ConceptType]


def _build_alternation(terms: list[str]) -> re.Pattern:
    """Longest-first alternation so overlapping dictionary entries prefer the
    longer span at a given start position (regex alternation tries branches
    left-to-right and stops at the first that matches, so sorting
    longest-first gives "maximal munch" behaviour without a separate
    longest-match pass)."""
    ordered = sorted(set(terms), key=len, reverse=True)
    escaped = [re.escape(t) for t in ordered]
    pattern = r"(?<!\w)(?:" + "|".join(escaped) + r")(?!\w)"
    return re.compile(pattern, re.IGNORECASE)


_SYMPTOM_RE = _build_alternation(SYMPTOM_TERMS)
_DIAGNOSIS_RE = _build_alternation(DIAGNOSIS_TERMS)
_DRUG_RE = _build_alternation(DRUG_TERMS)
_LAB_TEST_RE = _build_alternation(LAB_TEST_TERMS)

# Numeric lab-result token: a number (Vietnamese comma or dot as the decimal
# separator) optionally followed immediately by a unit. Deliberately never
# scanned over the whole document (dosage numbers like "25mg" or ages would
# swamp it with false positives) — only tried in a short window right after
# a TÊN_XÉT_NGHIỆM match, e.g. "kali 6.3", "troponin 0.01".
# The unit suffix (with its leading whitespace) is one optional group, not
# two independent `\s*`/unit pieces — otherwise a bare number with no unit
# (e.g. "creatinine 5.7\n") would still swallow trailing whitespace/newline
# into the matched span since `\s*` alone always matches (possibly empty).
_LAB_RESULT_NUM_RE = re.compile(r"\d+([.,]\d+)?(?:\s*(?:%|mg|ml|mmol/L|mmHg|lần/phút|°C|g/dL))?")
_LAB_RESULT_SEP_RE = re.compile(r"[\s:(\)]*")
_LAB_RESULT_WINDOW = 15


def find_dict_matches(source_text: str, pattern: re.Pattern) -> Iterator[Match]:
    for m in pattern.finditer(source_text):
        yield m.start(), m.end(), source_text[m.start() : m.end()]


def _lab_result_matches(source_text: str, lab_test_ends: list[int]) -> list[Match]:
    results: list[Match] = []
    for end in lab_test_ends:
        window = source_text[end : end + _LAB_RESULT_WINDOW]
        sep = _LAB_RESULT_SEP_RE.match(window)
        offset = sep.end() if sep else 0
        num = _LAB_RESULT_NUM_RE.match(window, offset)
        if num and num.end() > offset:
            start = end + offset
            text = window[offset : num.end()]
            results.append((start, start + len(text), text))
    return results


def extract_raw_matches(source_text: str) -> list[RawTypedMatch]:
    """Find every dictionary/regex match in `source_text`, resolve overlaps
    so the same span is never emitted under two different types, and return
    them left-to-right.

    Overlap resolution matters here specifically because a wrong `type` is a
    *double* penalty under the scoring formula (CLAUDE.md) — better to drop
    an ambiguous span than to emit it twice under conflicting types. Ties go
    to CHẨN_ĐOÁN / THUỐC first since `candidates_score` (linking) is the
    highest-weighted metric.
    """
    candidates: list[RawTypedMatch] = []
    for pattern, ctype in (
        (_DIAGNOSIS_RE, ConceptType.CHAN_DOAN),
        (_DRUG_RE, ConceptType.THUOC),
        (_SYMPTOM_RE, ConceptType.TRIEU_CHUNG),
        (_LAB_TEST_RE, ConceptType.TEN_XET_NGHIEM),
    ):
        for start, end, text in find_dict_matches(source_text, pattern):
            candidates.append((start, end, text, ctype))

    lab_test_ends = [c[1] for c in candidates if c[3] == ConceptType.TEN_XET_NGHIEM]
    for start, end, text in _lab_result_matches(source_text, lab_test_ends):
        candidates.append((start, end, text, ConceptType.KET_QUA_XET_NGHIEM))

    # Greedy longest-match-first, left-to-right, no overlaps.
    candidates.sort(key=lambda c: (c[0], -(c[1] - c[0])))
    accepted: list[RawTypedMatch] = []
    occupied: list[tuple[int, int]] = []
    for start, end, text, ctype in candidates:
        if any(start < o_end and end > o_start for o_start, o_end in occupied):
            continue
        accepted.append((start, end, text, ctype))
        occupied.append((start, end))

    accepted.sort(key=lambda c: c[0])
    return accepted
