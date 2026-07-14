"""Shared output contract — every workstream codes against this.

Mirrors the exact submission schema from `Topic.md` §3.2 / `CLAUDE.md`
"Output contract". This module only defines *shape*; semantic checks (position
bounds, type/assertion consistency, duplicates) live in `validate.py` so they
can be reported as a list of issues instead of raising on the first problem.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ConceptType(str, Enum):
    TRIEU_CHUNG = "TRIỆU_CHỨNG"
    TEN_XET_NGHIEM = "TÊN_XÉT_NGHIỆM"
    KET_QUA_XET_NGHIEM = "KẾT_QUẢ_XÉT_NGHIỆM"
    CHAN_DOAN = "CHẨN_ĐOÁN"
    THUOC = "THUỐC"


class Assertion(str, Enum):
    NEGATED = "isNegated"
    FAMILY = "isFamily"
    HISTORICAL = "isHistorical"


# Per Topic.md §3.2: assertions only apply to these 3 types, candidates
# (ICD-10/RxNorm) only to these 2 (the linking targets).
ASSERTION_ELIGIBLE_TYPES = frozenset(
    {ConceptType.TRIEU_CHUNG, ConceptType.CHAN_DOAN, ConceptType.THUOC}
)
CANDIDATE_ELIGIBLE_TYPES = frozenset({ConceptType.CHAN_DOAN, ConceptType.THUOC})
MAX_ASSERTIONS = 3


class Concept(BaseModel):
    """One extracted concept. Stays permissive on cross-field rules (e.g. an
    assertion on an ineligible type) so a concept can exist mid-pipeline before
    it's fully resolved — `validate.py` is what enforces the submission-time
    invariants and reports every violation at once.
    """

    text: str
    type: ConceptType
    position: tuple[int, int]
    assertions: list[Assertion] = Field(default_factory=list)
    candidates: Optional[list[str]] = None

    def to_json_dict(self) -> dict:
        """Serialize to the exact submission schema.

        `candidates` is omitted for types that don't carry it (CLAUDE.md:
        "omitted for other types"). `assertions` is always emitted as a list —
        empty for ineligible types — matching the worked example in `Topic.md`
        where every TRIỆU_CHỨNG/THUỐC entry shows an `"assertions"` array even
        when empty.
        """
        d: dict = {
            "text": self.text,
            "type": self.type.value,
            "position": [self.position[0], self.position[1]],
            "assertions": [a.value for a in self.assertions],
        }
        if self.type in CANDIDATE_ELIGIBLE_TYPES:
            d["candidates"] = list(self.candidates or [])
        return d

    @classmethod
    def from_json_dict(cls, d: dict) -> "Concept":
        pos = d["position"]
        return cls(
            text=d["text"],
            type=ConceptType(d["type"]),
            position=(pos[0], pos[1]),
            assertions=[Assertion(a) for a in d.get("assertions", [])],
            candidates=d.get("candidates"),
        )
