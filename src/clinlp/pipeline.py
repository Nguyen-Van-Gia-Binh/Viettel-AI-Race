"""Orchestrator: preprocessing -> sections -> ner -> position -> linking (only
CHẨN_ĐOÁN/THUỐC) -> validate -> schema objects.

The NER and linking steps are injected as callables rather than imported as
concrete implementations, so this module is usable and testable from day one —
before `ner.rules`/`ner.llm_extract` or `linking.link` exist for real — and so
swapping the Week-1 rule baseline for the later hybrid LLM+retrieval pipeline
is a one-line change at the call site (`scripts/run_baseline.py` vs
`scripts/run_inference.py`), not a rewrite of this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from clinlp.position import PositionResolver
from clinlp.schema import CANDIDATE_ELIGIBLE_TYPES, Assertion, Concept, ConceptType
from clinlp.validate import ValidationIssue, validate_concepts

# A raw extracted mention before position/candidates are resolved.
RawMention = tuple[str, ConceptType, list[Assertion]]

# ner_fn(source_text) -> mentions with NO position and NO candidates filled in yet.
NerFn = Callable[[str], Iterable[RawMention]]
# link_fn(entity_text, type, source_text) -> candidate codes; only called for
# CHẨN_ĐOÁN/THUỐC.
LinkFn = Callable[[str, ConceptType, str], list[str]]


@dataclass
class PipelineResult:
    concepts: list[Concept] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)


def run_pipeline(source_text: str, ner_fn: NerFn, link_fn: LinkFn | None = None) -> PipelineResult:
    """Run one document through the pipeline.

    If `link_fn` is omitted, CHẨN_ĐOÁN/THUỐC concepts get an empty candidates
    list — acceptable for an early baseline; `validate.py` flags it as a
    warning, not an error, so it never silently looks like a bug.
    """
    resolver = PositionResolver(source_text)
    concepts: list[Concept] = []

    for entity_text, concept_type, assertions in ner_fn(source_text):
        span = resolver.resolve(entity_text)
        if span is None:
            continue  # already logged inside PositionResolver — never guess a position

        start, end = span
        matched_text = source_text[start:end]

        candidates = None
        if concept_type in CANDIDATE_ELIGIBLE_TYPES:
            candidates = link_fn(matched_text, concept_type, source_text) if link_fn else []

        concepts.append(
            Concept(
                text=matched_text,
                type=concept_type,
                position=span,
                assertions=assertions,
                candidates=candidates,
            )
        )

    issues = validate_concepts(source_text, concepts)
    return PipelineResult(concepts=concepts, issues=issues)
