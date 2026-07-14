"""Reimplementation of the competition scoring formula (CLAUDE.md "Scoring"):

    final_score = 0.3 * text_score (1 - WER on `text` spans)
                + 0.3 * assertions_score (Jaccard)
                + 0.4 * candidates_score (Jaccard, weighted by len(ground_truth)+1)

`assertions_score` is only aggregated over concepts whose type carries
assertions (TRIỆU_CHỨNG/CHẨN_ĐOÁN/THUỐC); `candidates_score` only over concepts
whose type carries candidates (CHẨN_ĐOÁN/THUỐC) — matching CLAUDE.md's "only
applies to" / "omitted for other types" wording. `text_score` applies to every
concept regardless of type.

"Wrong type = double penalty" (Topic_Analysis.md §3.3): a predicted concept
whose position overlaps a gold concept but whose `type` differs is NOT counted
as a match — the gold concept scores as a miss (0 on every applicable metric)
*and* the predicted concept scores as a spurious prediction (0 on every
applicable metric) — two zero-scored items instead of one.

ASSUMPTION — no official reference scorer has been published by the
organizers, only the formula description in Topic_Analysis.md/CLAUDE.md.
Matching between predicted and gold concepts here is done by (a) identical
`type` and (b) overlapping `position` spans, greedily picking the
highest-overlap pairs first. If the organizers reveal a different matching
rule (e.g. an IoU threshold, or exact-position-only matching), this is the one
module to update — every other module (eval harness, error analysis) treats
this as ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from clinlp.schema import ASSERTION_ELIGIBLE_TYPES, CANDIDATE_ELIGIBLE_TYPES, Concept, ConceptType

Span = tuple[int, int]


def _word_error_rate(gold_text: str, pred_text: str) -> float:
    """Word-level WER = edit distance over word sequences / len(gold words)."""
    gold_words = gold_text.split()
    pred_words = pred_text.split()
    if not gold_words:
        return 0.0 if not pred_words else 1.0

    n, m = len(gold_words), len(pred_words)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            tmp = dp[j]
            dp[j] = prev if gold_words[i - 1] == pred_words[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[m] / n


def text_score(gold_text: str, pred_text: str) -> float:
    return max(0.0, 1.0 - _word_error_rate(gold_text, pred_text))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def assertions_score(gold: Concept, pred: Concept) -> float:
    return jaccard({x.value for x in gold.assertions}, {x.value for x in pred.assertions})


def candidates_score(gold: Concept, pred: Concept) -> float:
    return jaccard(set(gold.candidates or []), set(pred.candidates or []))


def candidates_weight(gold: Concept | None) -> int:
    """`len(ground_truth)+1` weighting from the formula. A spurious prediction
    with no gold concept has no ground-truth candidate list to weight by —
    defaults to weight 1 (neutral)."""
    if gold is None or gold.candidates is None:
        return 1
    return len(gold.candidates) + 1


@dataclass
class ScoredItem:
    type: ConceptType
    kind: str  # "matched" | "missed_gold" | "spurious_pred"
    text_score: float
    assertions_score: float
    candidates_score: float
    candidates_weight: int


def _spans_overlap(a: Span, b: Span) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _overlap_len(a: Span, b: Span) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def match_concepts(gold: list[Concept], pred: list[Concept]) -> list[ScoredItem]:
    """Greedy same-type, max-overlap matching. See module docstring for the
    matching assumption this encodes."""
    candidate_pairs = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            if g.type == p.type and _spans_overlap(g.position, p.position):
                candidate_pairs.append((_overlap_len(g.position, p.position), gi, pi))
    candidate_pairs.sort(reverse=True)

    used_gold: set[int] = set()
    used_pred: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, gi, pi in candidate_pairs:
        if gi in used_gold or pi in used_pred:
            continue
        pairs.append((gi, pi))
        used_gold.add(gi)
        used_pred.add(pi)

    items: list[ScoredItem] = []
    for gi, pi in pairs:
        g, p = gold[gi], pred[pi]
        items.append(
            ScoredItem(
                type=g.type,
                kind="matched",
                text_score=text_score(g.text, p.text),
                assertions_score=assertions_score(g, p),
                candidates_score=candidates_score(g, p),
                candidates_weight=candidates_weight(g),
            )
        )
    for gi, g in enumerate(gold):
        if gi not in used_gold:
            items.append(
                ScoredItem(
                    type=g.type,
                    kind="missed_gold",
                    text_score=0.0,
                    assertions_score=0.0,
                    candidates_score=0.0,
                    candidates_weight=candidates_weight(g),
                )
            )
    for pi, p in enumerate(pred):
        if pi not in used_pred:
            items.append(
                ScoredItem(
                    type=p.type,
                    kind="spurious_pred",
                    text_score=0.0,
                    assertions_score=0.0,
                    candidates_score=0.0,
                    candidates_weight=candidates_weight(None),
                )
            )
    return items


@dataclass
class FileScore:
    text_score: float
    assertions_score: float
    candidates_score: float
    final_score: float
    items: list[ScoredItem] = field(default_factory=list)

    @property
    def n_matched(self) -> int:
        return sum(1 for it in self.items if it.kind == "matched")

    @property
    def n_missed_gold(self) -> int:
        return sum(1 for it in self.items if it.kind == "missed_gold")

    @property
    def n_spurious_pred(self) -> int:
        return sum(1 for it in self.items if it.kind == "spurious_pred")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 1.0


def score_file(gold: list[Concept], pred: list[Concept]) -> FileScore:
    items = match_concepts(gold, pred)
    if not items:
        return FileScore(1.0, 1.0, 1.0, 1.0, items)

    text = _mean([it.text_score for it in items])

    assertion_items = [it for it in items if it.type in ASSERTION_ELIGIBLE_TYPES]
    assertions = _mean([it.assertions_score for it in assertion_items])

    candidate_items = [it for it in items if it.type in CANDIDATE_ELIGIBLE_TYPES]
    total_weight = sum(it.candidates_weight for it in candidate_items)
    candidates = (
        sum(it.candidates_score * it.candidates_weight for it in candidate_items) / total_weight
        if total_weight
        else 1.0
    )

    final = 0.3 * text + 0.3 * assertions + 0.4 * candidates
    return FileScore(
        text_score=text,
        assertions_score=assertions,
        candidates_score=candidates,
        final_score=final,
        items=items,
    )


def corpus_score(file_scores: list[FileScore]) -> float:
    """Unweighted mean of final_score across files — an assumption (the
    organizers may score the concatenation of all files rather than a mean of
    per-file scores); revisit if their aggregation differs."""
    return _mean([fs.final_score for fs in file_scores])
