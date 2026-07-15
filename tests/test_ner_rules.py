"""Rule-based NER (dictionary/regex) tests grounded in real cases from
input/input/*.txt, mirroring tests/test_position.py's style."""
import os

from clinlp.ner.rules import extract_raw_matches
from clinlp.schema import ConceptType

INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "input", "input")


def read_input(n: int) -> str:
    with open(os.path.join(INPUT_DIR, f"{n}.txt"), encoding="utf-8", newline="") as fh:
        return fh.read()


def test_repeated_symptom_matched_at_every_occurrence_1txt():
    """input/input/1.txt repeats "đánh trống ngực" many times (the same
    C9-in-Topic_Analysis.md case tests/test_position.py exercises for
    position resolution) — the dictionary matcher must find every
    occurrence, not just the first."""
    source = read_input(1)
    matches = extract_raw_matches(source)

    dtn = [m for m in matches if m[2] == "đánh trống ngực" and m[3] == ConceptType.TRIEU_CHUNG]
    assert len(dtn) > 3
    for start, end, text, _ctype in dtn:
        assert source[start:end] == text == "đánh trống ngực"


def test_matches_are_left_to_right_and_non_overlapping_1txt():
    source = read_input(1)
    matches = extract_raw_matches(source)

    starts = [m[0] for m in matches]
    assert starts == sorted(starts)

    for (s1, e1, _, _), (s2, e2, _, _) in zip(matches, matches[1:]):
        assert e1 <= s2, f"overlapping matches: ({s1},{e1}) and ({s2},{e2})"


def test_historical_drug_list_found_as_thuoc_1txt():
    """input/input/1.txt lines 3-5 list metoprolol/doxycycline/atenolol
    under "Thuốc trước khi nhập viện" — all three must be found as THUỐC."""
    source = read_input(1)
    matches = extract_raw_matches(source)

    drug_texts = {m[2].lower() for m in matches if m[3] == ConceptType.THUOC}
    assert {"metoprolol", "doxycycline", "atenolol"} <= drug_texts


def test_diagnosis_found_22txt():
    """input/input/22.txt: "khối máu tụ dưới màng cứng bán cấp hai bên" —
    the dictionary term "khối máu tụ dưới màng cứng" must be found as
    CHẨN_ĐOÁN, not TRIỆU_CHỨNG (a wrong `type` is a double penalty under the
    scoring formula, CLAUDE.md)."""
    source = read_input(22)
    matches = extract_raw_matches(source)

    diagnoses = [m for m in matches if m[3] == ConceptType.CHAN_DOAN]
    assert any(m[2] == "khối máu tụ dưới màng cứng" for m in diagnoses)


def test_ob_symptom_terms_found_31txt():
    """input/input/31.txt: "thai máy tốt (FM +)", "chưa ra huyết âm đạo (VB
    -)", "chưa vỡ ối/rỉ ối (LOF -)" — the obstetric findings must be
    detected as TRIỆU_CHỨNG spans exactly matching the source substrings."""
    source = read_input(31)
    matches = extract_raw_matches(source)

    symptom_texts = {m[2] for m in matches if m[3] == ConceptType.TRIEU_CHUNG}
    assert "thai máy tốt" in symptom_texts
    assert "ra huyết âm đạo" in symptom_texts
    assert "vỡ ối" in symptom_texts
    assert "rỉ ối" in symptom_texts


def test_lab_result_number_after_lab_test_37txt():
    """input/input/37.txt line 34: "creatinine 5.7" — a numeric result must
    be captured as KẾT_QUẢ_XÉT_NGHIỆM directly after the TÊN_XÉT_NGHIỆM
    match, without swallowing unrelated numbers elsewhere in the file."""
    source = read_input(37)
    matches = extract_raw_matches(source)

    results = [m for m in matches if m[3] == ConceptType.KET_QUA_XET_NGHIEM]
    result_texts = {m[2] for m in results}
    assert "5.7" in result_texts
