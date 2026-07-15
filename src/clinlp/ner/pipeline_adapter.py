"""Final assembly point for the Week-1/2/3 pipelines: wires
either `ner/rules.py` (rule-based baseline) or `ner/llm_extract.py` (LLM-based)
together with assertions into a single callable matching `pipeline.NerFn`.
Supports configuration-driven swapping and robust rule-based fallback.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

from clinlp.ner.assertions import detect_assertions
from clinlp.ner.rules import extract_raw_matches
from clinlp.schema import ASSERTION_ELIGIBLE_TYPES, Assertion, ConceptType
from clinlp.sections import build_section_index

# Matches pipeline.RawMention / pipeline.NerFn without importing pipeline.py
RawMention = tuple[str, ConceptType, list[Assertion]]

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "model.json"


def rules_ner_fn(source_text: str) -> Iterator[RawMention]:
    """Pure rule-based NER & Assertion detection."""
    section_index = build_section_index(source_text)

    for start, end, text, concept_type in extract_raw_matches(source_text):
        assertions: list[Assertion] = []
        if concept_type in ASSERTION_ELIGIBLE_TYPES:
            assertions = detect_assertions(source_text, start, end, section_index)
        yield text, concept_type, assertions


def ner_fn(source_text: str) -> Iterator[RawMention]:
    """Adaptable `ner_fn` routing. Reads `configs/model.json` to decide whether
    to use the LLM extractor or the rules baseline. If the LLM extraction fails,
    it automatically falls back to the rules baseline to guarantee a valid output.
    """
    from clinlp.ner.llm_extract import llm_ner_fn

    mode = "rules"
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
                mode = config.get("pipeline", {}).get("mode", "rules")
    except Exception as e:
        logger.warning("Failed to load pipeline mode config, using rules baseline: %s", e)

    if mode == "llm":
        try:
            # Try running the LLM NER and Assertion extractor.
            # We convert to a list to trigger any generation/parsing errors
            # immediately so that the fallback can catch them.
            results = list(llm_ner_fn(source_text))
            yield from results
            return
        except Exception as e:
            logger.error("LLM extraction failed. Falling back to rules baseline. Error: %s", e)

    # Default/Fallback to rules
    yield from rules_ner_fn(source_text)

