"""Bước 0: preprocessing for the LLM-facing working copy only.

CRITICAL: the output of `normalize_for_llm` is ONLY for prompting or heuristics
that tolerate minor text changes (e.g. section-header matching). Never compute
a `position` against normalized text — `clinlp.position` always matches back
against the untouched raw file, per CLAUDE.md: "never run text through a
normalization step that shifts character offsets before matching back against
the original".
"""
from __future__ import annotations

import unicodedata


def normalize_for_llm(text: str) -> str:
    """NFC-normalize text for feeding to the LLM. Do not use the result for
    position calculation — pass the untouched raw text to `clinlp.position`."""
    return unicodedata.normalize("NFC", text)
