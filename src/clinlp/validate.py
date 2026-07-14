"""Bước 6: schema/position/consistency validation over a full file's concept list.

Returns a list of issues rather than raising on the first problem, so
`scripts/make_submission.py` can report everything wrong with a file at once
and refuse to zip it if any ERROR-severity issue is present.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from clinlp.schema import (
    ASSERTION_ELIGIBLE_TYPES,
    CANDIDATE_ELIGIBLE_TYPES,
    MAX_ASSERTIONS,
    Concept,
)


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass
class ValidationIssue:
    severity: Severity
    concept_index: int
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.value}] concept#{self.concept_index}: {self.message}"


def validate_concepts(source_text: str, concepts: list[Concept]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    seen_type_position: set[tuple[str, tuple[int, int]]] = set()
    n = len(source_text)

    for i, c in enumerate(concepts):
        start, end = c.position
        if not (0 <= start < end <= n):
            issues.append(
                ValidationIssue(
                    Severity.ERROR,
                    i,
                    f"position {list(c.position)} out of bounds for source of length {n}",
                )
            )
        else:
            actual = source_text[start:end]
            if actual != c.text:
                issues.append(
                    ValidationIssue(
                        Severity.ERROR,
                        i,
                        f"text mismatch at position {list(c.position)}: "
                        f"concept.text={c.text!r} but source[start:end]={actual!r}",
                    )
                )

        if c.assertions and c.type not in ASSERTION_ELIGIBLE_TYPES:
            issues.append(
                ValidationIssue(
                    Severity.ERROR,
                    i,
                    f"type {c.type.value} must not carry assertions, got {[a.value for a in c.assertions]}",
                )
            )
        if len(c.assertions) > MAX_ASSERTIONS:
            issues.append(
                ValidationIssue(Severity.ERROR, i, f"assertions exceeds max of {MAX_ASSERTIONS}: {c.assertions}")
            )
        if len(set(c.assertions)) != len(c.assertions):
            issues.append(ValidationIssue(Severity.ERROR, i, f"duplicate assertion values: {c.assertions}"))

        if c.candidates is not None and c.type not in CANDIDATE_ELIGIBLE_TYPES:
            issues.append(
                ValidationIssue(
                    Severity.ERROR, i, f"type {c.type.value} must not carry candidates, got {c.candidates}"
                )
            )
        if c.type in CANDIDATE_ELIGIBLE_TYPES and not c.candidates:
            issues.append(
                ValidationIssue(Severity.WARNING, i, f"type {c.type.value} has no candidates (empty linking result)")
            )

        key = (c.type.value, c.position)
        if key in seen_type_position:
            issues.append(ValidationIssue(Severity.ERROR, i, f"duplicate concept: same type+position {key} already seen"))
        seen_type_position.add(key)

    return issues


def has_errors(issues: list[ValidationIssue]) -> bool:
    return any(issue.severity == Severity.ERROR for issue in issues)
