"""Bước 3: exact character-offset position calculation.

INVARIANT (CLAUDE.md): never let a model emit position numbers directly. This
module is the only place spans get turned into `[start, end]` offsets, and it
always matches against the untouched raw source text — never a normalized
working copy (see `preprocessing.py`) — so offsets are never shifted.

Three tiers, tried in order, so a span is always found via *some* exact,
traceable strategy rather than guessed:
1. exact substring match
2. whitespace-flexible match (handles missing/extra spaces, e.g. a line-wrap
   join like "chuyển dạBệnh sử" in `input/input/31.txt`)
3. Unicode-normalization-flexible match (handles composed vs decomposed
   Vietnamese diacritics between the model's output and the raw file)

All three tiers return offsets into the original raw string — there is no
separate normalized copy to remap offsets from.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

Span = tuple[int, int]


def find_all_exact(source: str, entity_text: str) -> list[Span]:
    """All exact occurrences of entity_text in source, left to right."""
    if not entity_text:
        return []
    spans = []
    start = 0
    while True:
        idx = source.find(entity_text, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(entity_text)))
        start = idx + 1  # step by 1 so overlapping occurrences are still found
    return spans


def _whitespace_flexible_pattern(entity_text: str) -> re.Pattern:
    parts = re.split(r"(\s+)", entity_text)
    pattern = "".join(r"\s+" if p and p.isspace() else re.escape(p) for p in parts)
    return re.compile(pattern)


def find_all_whitespace_flexible(source: str, entity_text: str) -> list[Span]:
    if not entity_text:
        return []
    pattern = _whitespace_flexible_pattern(entity_text)
    return [(m.start(), m.end()) for m in pattern.finditer(source)]


def find_all_unicode_flexible(source: str, entity_text: str) -> list[Span]:
    """Sliding-window search comparing NFC-normalized forms. Still returns
    offsets into the raw (un-normalized) source. Bounded-width window, fine for
    corpus files of a few KB each."""
    if not entity_text:
        return []
    target = unicodedata.normalize("NFC", entity_text)
    target_len = len(entity_text)
    spans: set[Span] = set()
    # NFC/NFD length differences are small (roughly one extra codepoint per
    # composed Vietnamese character with a combining mark) — +/-6 comfortably covers it.
    for window in range(max(1, target_len - 6), target_len + 7):
        if window > len(source):
            continue
        for i in range(0, len(source) - window + 1):
            candidate = source[i : i + window]
            if unicodedata.normalize("NFC", candidate) == target:
                spans.add((i, i + window))
    return sorted(spans)


@dataclass
class PositionResolver:
    """Resolves a sequence of extracted entity texts to non-overlapping spans in
    one source document, so duplicate entities (e.g. "đánh trống ngực"
    appearing repeatedly in `input/input/1.txt`) each get their own correct,
    distinct occurrence instead of collapsing onto the first match."""

    source: str
    used_spans: set[Span] = field(default_factory=set)

    def resolve(self, entity_text: str) -> Span | None:
        if not entity_text:
            logger.warning("empty entity_text passed to PositionResolver.resolve")
            return None

        for finder, tier in (
            (find_all_exact, "exact"),
            (find_all_whitespace_flexible, "whitespace-flexible"),
            (find_all_unicode_flexible, "unicode-flexible"),
        ):
            candidates = [s for s in finder(self.source, entity_text) if s not in self.used_spans]
            if candidates:
                span = candidates[0]
                if tier != "exact":
                    logger.warning(
                        "position fallback tier=%s used for entity_text=%r -> span=%s",
                        tier,
                        entity_text,
                        span,
                    )
                self.used_spans.add(span)
                return span

        logger.error("could not resolve any position for entity_text=%r", entity_text)
        return None
