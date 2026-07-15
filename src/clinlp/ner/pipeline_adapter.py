"""Final assembly point for the Week-1 rule-based baseline: wires
`ner/rules.py` (span + type detection) together with `ner/assertions.py`
(Layer 1 NegEx + Layer 2 section propagation via `sections.py`) into a single
callable matching `pipeline.NerFn` — this is what gets passed as `ner_fn` to
`pipeline.run_pipeline()` (e.g. from a future `scripts/run_baseline.py`).

Deliberately a separate module rather than bolted onto `ner/rules.py`, so
`rules.py` stays a pure span/type finder that `ner/llm_extract.py` or a
future hybrid NER stage could reuse without pulling in assertion logic.
"""
from __future__ import annotations

from typing import Iterator

from clinlp.ner.assertions import detect_assertions
from clinlp.ner.rules import extract_raw_matches
from clinlp.schema import ASSERTION_ELIGIBLE_TYPES, Assertion, ConceptType
from clinlp.sections import build_section_index

# Matches pipeline.RawMention / pipeline.NerFn without importing pipeline.py
# itself, to avoid a needless import-time dependency in the other direction
# (pipeline.py must stay import-clean of every concrete ner_fn implementation).
RawMention = tuple[str, ConceptType, list[Assertion]]


def ner_fn(source_text: str) -> Iterator[RawMention]:
    """Rule-based baseline `ner_fn`: dictionary/regex span detection +
    3-layer assertion detection, matching `pipeline.NerFn`'s contract
    exactly (entity text + type + assertions; no position, no candidates —
    `pipeline.run_pipeline` fills those in via `PositionResolver`/`link_fn`).
    """
    section_index = build_section_index(source_text)

    for start, end, text, concept_type in extract_raw_matches(source_text):
        assertions: list[Assertion] = []
        if concept_type in ASSERTION_ELIGIBLE_TYPES:
            assertions = detect_assertions(source_text, start, end, section_index)
        yield text, concept_type, assertions
