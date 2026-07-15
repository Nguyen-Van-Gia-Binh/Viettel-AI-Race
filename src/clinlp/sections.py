"""Bước 1 (Workstream B): Vietnamese clinical section-header parser.

Extends the regex heuristic prototyped in `scripts/survey.py`
(`header_pattern` there matches short standalone lines that look like a
section title — starts with an uppercase Vietnamese letter, 3-45 chars,
no leading "-", no "label: value" punctuation). This module turns that
survey heuristic into a real span+lookup API used by `ner/assertions.py`'s
Layer 2 (section propagation of `isHistorical` / `isFamily`).

INVARIANT (same as position.py): section spans are character offsets on the
*raw, untouched* source text — never run this over a normalized copy
(`preprocessing.normalize_for_llm`), or offsets fed back into `position.py`
would be shifted.

Header-line detection is intentionally a *full-line* match (the whole
stripped line must be letters+spaces, optionally preceded by "N." / "N)"
numbering): this is what makes it cheap to tell a real section header
("Thuốc trước khi nhập viện" on its own line, `input/input/1.txt`) apart
from a "Label: value" line ("Thuốc trước khi nhập viện: aspirin 325mg hằng
ngày", `input/input/22.txt`) — the latter never matches because it contains
a colon. Both cases still end up historical: the standalone-header case via
its own section, the inline-label case via the enclosing numbered section
("1. Tiền sử bệnh") that it sits inside.
"""
from __future__ import annotations

import re
from typing import NamedTuple

from clinlp.ner.dictionaries import NEGATION_CUES

_VN_UPPER = "A-ZĐÁÀẢÃẠÂẤẦẨẪẬĂẮẰẲẴẶÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ"
_VN_LOWER = "a-zàảãáạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ"
_VN_LETTER = _VN_UPPER + _VN_LOWER

# Whole-line match: optional "N." / "N)" numbering, then 3-45 chars of
# letters+spaces only (mirrors survey.py's length/character-class heuristic,
# but written as a single anchored pattern so exclusions like "starts with
# '-'" or "has 'label:' punctuation" fall out for free instead of needing
# separate `if` checks).
HEADER_LINE_RE = re.compile(
    rf"^\s*\d*[.)]?\s*([{_VN_UPPER}][{_VN_LETTER} ]{{2,44}})\s*$"
)

# "Tiền sử bệnh hiện tại" / "Bệnh sử hiện tại" is History-of-PRESENT-illness
# (current symptoms), not patient history, even though the former literally
# contains the substring "tiền sử". Without this exclusion, every current
# symptom listed under that heading in input/input/1.txt and 15.txt would be
# wrongly propagated as isHistorical.
_PRESENT_MARKER = "hiện tại"


class Section(NamedTuple):
    """One parsed section span. Deliberately a NamedTuple so callers can use
    it either as `(start, end, label)` (per the "trả về (start, end,
    section_label)" requirement) or via the named `.is_historical` /
    `.is_family` fields that Layer 2 propagation needs.
    """

    start: int
    end: int
    label: str
    is_historical: bool
    is_family: bool


def _iter_lines_with_offsets(text: str):
    """Yield (line_start, line_end_excl_terminator, full_line_incl_terminator)
    for each line, so header spans and section boundaries land on exact
    offsets into `text` regardless of LF vs CRLF (CLAUDE.md risk #2)."""
    pos = 0
    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        yield pos, pos + len(content), line
        pos += len(line)


def _is_historical_header(header_text: str) -> bool:
    low = header_text.lower()
    if _PRESENT_MARKER in low:
        return False
    return any(cue.lower() in low for cue in NEGATION_CUES["section_historical"])


def _is_family_header(header_text: str) -> bool:
    # "Tiền sử gia đình" (CLAUDE.md) — family propagation is keyed off the
    # literal "gia đình" substring, not the full family_cues list (that list
    # also has single-syllable tokens like "bố"/"mẹ" that are for Layer-1
    # in-clause cue matching, not header classification).
    return "gia đình" in header_text.lower()


def parse_sections(source_text: str) -> list[Section]:
    """Parse all section headers in `source_text` into a flat, non-nested
    list of spans. Each header's section runs from right after its own line
    to right before the next header line (of any kind) or end of text —
    a flat model rather than a nested one, which is enough to satisfy the
    two propagation rules in CLAUDE.md (a numbered top-level header like
    "1. Tiền sử bệnh" and a standalone sub-header like "Thuốc trước khi nhập
    viện" both work the same way) without the added complexity of tracking
    header hierarchy that nothing here currently needs.
    """
    # Each entry: (line_start, content_start, header_text). content_start is
    # where this header's own section body begins (right after its line);
    # line_start is where the *next* header's line begins is what bounds the
    # previous section's end, so both offsets are kept.
    headers: list[tuple[int, int, str]] = []
    for line_start, line_end, full_line in _iter_lines_with_offsets(source_text):
        content = source_text[line_start:line_end]
        if not content.strip():
            continue
        m = HEADER_LINE_RE.match(content)
        if m:
            content_start = line_start + len(full_line)
            headers.append((line_start, content_start, m.group(1).strip()))

    sections: list[Section] = []
    for i, (_line_start, content_start, label) in enumerate(headers):
        end = headers[i + 1][0] if i + 1 < len(headers) else len(source_text)
        sections.append(
            Section(
                start=content_start,
                end=end,
                label=label,
                is_historical=_is_historical_header(label),
                is_family=_is_family_header(label),
            )
        )
    return sections


class SectionIndex:
    """Lookup: which section does character offset X belong to."""

    def __init__(self, sections: list[Section]):
        self.sections = sections

    def section_at(self, offset: int) -> Section | None:
        for s in self.sections:
            if s.start <= offset < s.end:
                return s
        return None

    def is_historical_at(self, offset: int) -> bool:
        s = self.section_at(offset)
        return bool(s and s.is_historical)

    def is_family_at(self, offset: int) -> bool:
        s = self.section_at(offset)
        return bool(s and s.is_family)


def build_section_index(source_text: str) -> SectionIndex:
    return SectionIndex(parse_sections(source_text))
