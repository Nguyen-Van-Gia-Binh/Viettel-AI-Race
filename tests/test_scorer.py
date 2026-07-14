from clinlp.eval.scorer import FileScore, corpus_score, match_concepts, score_file, text_score
from clinlp.schema import Assertion, Concept, ConceptType


def concept(text, type_, position, assertions=None, candidates=None):
    return Concept(text=text, type=type_, position=position, assertions=assertions or [], candidates=candidates)


def test_perfect_match_scores_one():
    gold = [
        concept("ho khan", ConceptType.TRIEU_CHUNG, (0, 7)),
        concept("viêm phổi", ConceptType.CHAN_DOAN, (20, 29), candidates=["J18.9"]),
    ]
    pred = [
        concept("ho khan", ConceptType.TRIEU_CHUNG, (0, 7)),
        concept("viêm phổi", ConceptType.CHAN_DOAN, (20, 29), candidates=["J18.9"]),
    ]
    fs = score_file(gold, pred)
    assert fs.text_score == 1.0
    assert fs.assertions_score == 1.0
    assert fs.candidates_score == 1.0
    assert fs.final_score == 1.0
    assert fs.n_matched == 2


def test_wrong_type_is_double_penalty_not_a_match():
    """Same span, but predicted type differs from gold type -> must NOT match;
    counted as one missed_gold + one spurious_pred (Topic_Analysis.md §3.3),
    zeroing every metric for both instead of just one."""
    gold = [concept("đau bụng", ConceptType.TRIEU_CHUNG, (0, 8))]
    pred = [concept("đau bụng", ConceptType.CHAN_DOAN, (0, 8), candidates=["R10.9"])]

    items = match_concepts(gold, pred)
    assert len(items) == 2
    assert sorted(it.kind for it in items) == ["missed_gold", "spurious_pred"]
    assert all(it.text_score == 0.0 for it in items)

    fs = score_file(gold, pred)
    assert fs.final_score == 0.0


def test_missed_gold_entity_reduces_score():
    gold = [
        concept("ho khan", ConceptType.TRIEU_CHUNG, (0, 7)),
        concept("sốt cao", ConceptType.TRIEU_CHUNG, (20, 27)),
    ]
    pred = [concept("ho khan", ConceptType.TRIEU_CHUNG, (0, 7))]
    fs = score_file(gold, pred)
    assert fs.n_matched == 1
    assert fs.n_missed_gold == 1
    assert fs.text_score == 0.5  # one perfect match + one zero, averaged


def test_spurious_prediction_reduces_score():
    gold = [concept("ho khan", ConceptType.TRIEU_CHUNG, (0, 7))]
    pred = [
        concept("ho khan", ConceptType.TRIEU_CHUNG, (0, 7)),
        concept("đau ngực", ConceptType.TRIEU_CHUNG, (20, 28)),
    ]
    fs = score_file(gold, pred)
    assert fs.n_matched == 1
    assert fs.n_spurious_pred == 1
    assert fs.text_score == 0.5


def test_candidates_score_weighted_by_ground_truth_length_plus_one():
    gold = [
        concept("viêm phổi", ConceptType.CHAN_DOAN, (0, 9), candidates=["J18.9"]),  # weight 2, pred wrong -> 0
        concept("trào ngược dạ dày", ConceptType.CHAN_DOAN, (20, 37), candidates=["K21.0", "K21.9"]),  # weight 3, exact
    ]
    pred = [
        concept("viêm phổi", ConceptType.CHAN_DOAN, (0, 9), candidates=["J15.9"]),
        concept("trào ngược dạ dày", ConceptType.CHAN_DOAN, (20, 37), candidates=["K21.0", "K21.9"]),
    ]
    fs = score_file(gold, pred)
    # weighted mean: (0*2 + 1*3) / (2+3) = 0.6
    assert fs.candidates_score == 0.6


def test_assertions_score_ignores_ineligible_types():
    gold = [
        concept("công thức máu", ConceptType.TEN_XET_NGHIEM, (0, 13)),  # never carries assertions
        concept(
            "hen suyễn",
            ConceptType.CHAN_DOAN,
            (20, 29),
            assertions=[Assertion.HISTORICAL],
            candidates=["J45.9"],
        ),
    ]
    pred = [
        concept("công thức máu", ConceptType.TEN_XET_NGHIEM, (0, 13)),
        concept("hen suyễn", ConceptType.CHAN_DOAN, (20, 29), assertions=[], candidates=["J45.9"]),
    ]
    fs = score_file(gold, pred)
    # only the CHẨN_ĐOÁN item counts toward assertions_score: jaccard({isHistorical}, {}) == 0.
    # if the TÊN_XÉT_NGHIỆM item were wrongly included it would dilute this toward 0.5.
    assert fs.assertions_score == 0.0


def test_word_error_rate_basic():
    assert text_score("ho khan", "ho khan") == 1.0
    assert text_score("ho khan kéo dài", "ho khan") < 1.0


def test_corpus_score_is_mean_of_file_scores():
    scores = [FileScore(1.0, 1.0, 1.0, 1.0), FileScore(0.0, 0.0, 0.0, 0.0)]
    assert corpus_score(scores) == 0.5
