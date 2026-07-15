"""Bước 4 (Workstream B, Week 1): 3-layer assertion detection.

- Layer 1 (this module, rule-based): Vietnamese NegEx-style cue matching —
  pre/post negation, family, historical cues — scoped to the entity's own
  clause (`Sota_research_report.md` §4.2/§4.3).
- Layer 2 (this module, via `sections.py`): section-header propagation —
  `isHistorical`/`isFamily` for every entity inside a "Tiền sử"/"Thuốc trước
  khi nhập viện"/"Bệnh nền"/"Bệnh kèm"/"Tiền sử gia đình" section.
- Layer 3 (LLM refinement for ambiguous cases) — tuần 3.

Edge cases handled per §4.3:
- Long-range negation: a pre-negation cue's scope extends forward to the
  next hard sentence boundary (`.`, `;`, newline), not just the next word —
  e.g. "Không buồn nôn, hay nôn, đổ mồ hôi" negates all three items even
  though only the first is adjacent to "Không".
- Combined assertions: family + negation can both fire for the same entity
  ("Gia đình không có tiền sử bệnh tim").
- Implicit symbols: `(-)`/`(+)`-style shorthand, e.g. `input/input/31.txt`'s
  "(FM +)" (positive — no negation) vs "(VB -)" / "(LOF -)" (negative).
"""
from __future__ import annotations

import re

from clinlp.ner.dictionaries import NEGATION_CUES
from clinlp.schema import Assertion
from clinlp.sections import SectionIndex

# Clause boundaries for Layer 1 scoping. Deliberately narrow — only hard
# sentence-final punctuation and line breaks end a clause; a comma does NOT
# (that's exactly what lets "Không buồn nôn, hay nôn, đổ mồ hôi" negate all
# three comma-joined items in one clause).
_HARD_STOP_RE = re.compile(r"[.\n;]")

# Structural scope-limiters for Layer 1's forward negation window — a
# contrastive conjunction after the cue ends its scope early (e.g. "không
# sốt nhưng có ho" must not negate "ho"). This is a clause-boundary
# mechanism, not a semantic cue, so it deliberately lives here rather than
# in NEGATION_CUES (CLAUDE.md: don't add cues beyond the seeded dict).
_SCOPE_BREAK_CONJUNCTIONS = ["nhưng", "tuy nhiên", "song"]

# Implicit negative/positive shorthand, e.g. "(VB -)", "(LOF -)", "(-)" vs
# "(FM +)" (input/input/31.txt). Only "-" signals a negation; "+" never does.
_IMPLICIT_NEG_SYMBOL_RE = re.compile(r"\([A-Za-zÀ-ỹ]{0,8}\s*-\s*\)")

# "không cản quang" / "không thuốc cản quang" / "không tiêm thuốc cản quang"
# is a fixed Vietnamese radiology idiom for "without [IV] contrast [agent]"
# (input/input/22.txt: "chụp ct sọ não không thuốc cản quang Hình ảnh khối
# máu tụ dưới màng cứng..."). The "không" there modifies the imaging
# technique, not the diagnosis that follows in the same clause — masking it
# out before the pre-negation scan keeps a genuine "không X" earlier in the
# same clause (if any) working normally. This is a scope refinement on the
# existing "không" cue, not a new cue (CLAUDE.md: don't extend NEGATION_CUES).
_NEGATION_IDIOM_EXCEPTION_RE = re.compile(
    r"không\s+(?:tiêm\s+)?(?:thuốc\s+)?cản\s+quang", re.IGNORECASE
)

# Post-negation cues (and the implicit "(-)" symbol) are checked immediately
# after the entity, not over a wide window — unlike pre-negation, a
# post-negation marker like "(-)"/"âm tính" is meant to sit directly next to
# the entity it describes ("XN ABC (-)" / "cúm âm tính"). A loose window
# would false-positive on unrelated text nearby, e.g. "...metoprolol 25mg po
# bid, không có cải thiện" (input/input/1.txt line 35) — "không" there
# negates "cải thiện" (no improvement), not the drug "metoprolol" 22 chars
# earlier, so the match must be anchored right after the entity (allowing
# only whitespace/punctuation separators, not other words, in between).
_POST_SEP_RE = re.compile(r"[\s:,\-]*")
_POST_NEGATION_WINDOW = 20


def _cue_pattern(cue: str) -> re.Pattern:
    """Word-boundary-wrap pure letter/space cues (so "em" doesn't match
    inside "kém"); symbol-bearing cues like "(-)" are matched literally
    since \\b isn't meaningful around punctuation."""
    if re.fullmatch(r"[^\W\d_](?:[^\W\d_ ])*(?: [^\W\d_]+)*", cue):
        return re.compile(r"(?<!\w)" + re.escape(cue) + r"(?!\w)", re.IGNORECASE)
    return re.compile(re.escape(cue), re.IGNORECASE)


_PRE_NEGATION_PATTERNS = [_cue_pattern(c) for c in NEGATION_CUES["pre_negation"]]
_POST_NEGATION_PATTERNS = [_cue_pattern(c) for c in NEGATION_CUES["post_negation"]]
_FAMILY_PATTERNS = [_cue_pattern(c) for c in NEGATION_CUES["family_cues"]]
_HISTORICAL_PATTERNS = [_cue_pattern(c) for c in NEGATION_CUES["historical_cues"]]
_SCOPE_BREAK_PATTERNS = [_cue_pattern(c) for c in _SCOPE_BREAK_CONJUNCTIONS]


def _clause_bounds(source: str, pos: int) -> tuple[int, int]:
    """[start, end) of the clause containing offset `pos`, bounded by the
    nearest hard-stop punctuation on either side (or string edges)."""
    start = 0
    for m in _HARD_STOP_RE.finditer(source, 0, pos):
        start = m.end()
    end_match = _HARD_STOP_RE.search(source, pos)
    end = end_match.start() if end_match else len(source)
    return start, end


def detect_negation(source: str, start: int, end: int) -> bool:
    """Layer 1 pre/post negation, scoped to the entity's own clause (long-
    range for pre-negation, narrow window for post-negation/symbols)."""
    clause_start, clause_end = _clause_bounds(source, start)
    raw_pre_context = source[clause_start:start]
    pre_context = _NEGATION_IDIOM_EXCEPTION_RE.sub(
        lambda m: " " * len(m.group(0)), raw_pre_context
    )

    last_cue_end = -1
    for pattern in _PRE_NEGATION_PATTERNS:
        for m in pattern.finditer(pre_context):
            last_cue_end = max(last_cue_end, m.end())
    if last_cue_end != -1:
        # A contrastive conjunction between the cue and the entity ends the
        # negation scope early (e.g. "không sốt nhưng có ho").
        between = pre_context[last_cue_end:]
        if not any(p.search(between) for p in _SCOPE_BREAK_PATTERNS):
            return True

    post_context = source[end : min(clause_end, end + _POST_NEGATION_WINDOW)]

    # Word/symbol cues ("âm tính", "bình thường", "(-)") must sit right after
    # the entity — only whitespace/",:-" separators allowed in between, no
    # other words — so a match anywhere later in the window never counts.
    sep = _POST_SEP_RE.match(post_context)
    probe = post_context[sep.end() if sep else 0 :]
    if any(p.match(probe) for p in _POST_NEGATION_PATTERNS):
        return True

    # The implicit "(-)" symbol (e.g. "(VB -)") is checked separately since
    # its own leading "(" must not be eaten by _POST_SEP_RE first — only
    # plain whitespace may precede it.
    if _IMPLICIT_NEG_SYMBOL_RE.match(post_context.lstrip(" \t")):
        return True
    return False


def detect_family_cue(source: str, start: int, end: int) -> bool:
    """Layer 1 family cue — anywhere in the entity's own clause."""
    clause_start, clause_end = _clause_bounds(source, start)
    clause = source[clause_start:clause_end]
    return any(p.search(clause) for p in _FAMILY_PATTERNS)


def detect_historical_cue(source: str, start: int, end: int) -> bool:
    """Layer 1 historical cue — text before the entity within its own
    clause (e.g. "có tiền sử hen suyễn")."""
    clause_start, _clause_end = _clause_bounds(source, start)
    pre_context = source[clause_start:start]
    return any(p.search(pre_context) for p in _HISTORICAL_PATTERNS)


def detect_assertions(
    source_text: str,
    start: int,
    end: int,
    section_index: SectionIndex | None = None,
) -> list[Assertion]:
    """Run Layer 1 (cue-based) + Layer 2 (section propagation, if
    `section_index` is given) for one entity span and return the combined
    assertion list, in `Assertion` enum declaration order."""
    assertions: list[Assertion] = []

    if detect_negation(source_text, start, end):
        assertions.append(Assertion.NEGATED)

    is_family = detect_family_cue(source_text, start, end)
    is_historical = detect_historical_cue(source_text, start, end)
    if section_index is not None:
        is_historical = is_historical or section_index.is_historical_at(start)
        is_family = is_family or section_index.is_family_at(start)

    if is_family:
        assertions.append(Assertion.FAMILY)
    if is_historical:
        assertions.append(Assertion.HISTORICAL)

    return assertions
