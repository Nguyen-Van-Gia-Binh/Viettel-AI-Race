"""Assertion detection (Layer 1 NegEx + Layer 2 section propagation) tests.
Three cases are grounded in real input/input/*.txt lines per CLAUDE.md
(negation in 1.txt, historical-section propagation in 22.txt, implicit
symbols in 31.txt); the combined-assertion case uses the synthetic example
given verbatim in Sota_research_report.md §4.3 since no real corpus line
surfaced that exact combination."""
import os

from clinlp.ner.assertions import detect_assertions, detect_negation
from clinlp.sections import build_section_index
from clinlp.schema import Assertion

INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "input", "input")


def read_input(n: int) -> str:
    with open(os.path.join(INPUT_DIR, f"{n}.txt"), encoding="utf-8", newline="") as fh:
        return fh.read()


def test_long_range_negation_1txt():
    """input/input/1.txt line 30: "Không  buồn nôn, hay nôn, đổ mồ hôi" —
    the single "Không" cue at the start of the line must negate all three
    comma-joined symptoms, not just the first (Sota_research_report.md §4.3
    "long-range negation")."""
    source = read_input(1)

    for entity in ("buồn nôn", "nôn", "đổ mồ hôi"):
        start = source.index(entity, source.index("Không  buồn nôn"))
        end = start + len(entity)
        assert detect_negation(source, start, end) is True, f"{entity!r} should be negated"


def test_historical_section_propagation_22txt():
    """input/input/22.txt line 2, "Bệnh lý mãn tính: bệnh mạch máu", sits
    inside the numbered "1. Tiền sử bệnh" section (lines 2-3, everything up
    to the next header "2. Bệnh sử hiện tại") — "bệnh mạch máu" must get
    isHistorical via Layer 2 section propagation even though no Layer-1
    historical cue word appears anywhere in its own clause ("Bệnh lý mãn
    tính:" contains none of the NEGATION_CUES["historical_cues"] phrases).

    ("aspirin" on line 3, "Thuốc trước khi nhập viện: aspirin 325mg hằng
    ngày", is NOT used for the negative half of this check: that line's own
    clause literally contains the historical cue phrase "trước khi nhập
    viện" immediately before "aspirin", so Layer 1 alone already tags it
    historical regardless of section propagation — sections.py's docstring
    notes both mechanisms agree there. "bệnh mạch máu" isolates Layer 2's
    contribution cleanly since it has no such inline cue.)"""
    source = read_input(22)
    section_index = build_section_index(source)

    start = source.index("bệnh mạch máu")
    end = start + len("bệnh mạch máu")

    assertions = detect_assertions(source, start, end, section_index)
    assert Assertion.HISTORICAL in assertions

    # Without section context, Layer 1 alone should NOT find isHistorical
    # here — this is exactly what Layer 2 propagation is for.
    assertions_no_section = detect_assertions(source, start, end, section_index=None)
    assert Assertion.HISTORICAL not in assertions_no_section


def test_implicit_symbols_31txt():
    """input/input/31.txt:
    - "thai máy tốt (FM +)" — a "+" symbol must NOT trigger isNegated.
    - "chưa ra huyết âm đạo (VB -)" — negated via the pre-negation cue
      "chưa" (reinforced by the "-" symbol).
    - "chưa vỡ ối/rỉ ối (LOF -)" — "chưa" must negate both "vỡ ối" and
      "rỉ ối" even though a "/" (not a hard clause boundary) sits between
      them.
    """
    source = read_input(31)

    fm_start = source.index("thai máy tốt")
    fm_end = fm_start + len("thai máy tốt")
    assert detect_negation(source, fm_start, fm_end) is False

    vb_start = source.index("ra huyết âm đạo")
    vb_end = vb_start + len("ra huyết âm đạo")
    assert detect_negation(source, vb_start, vb_end) is True

    for entity in ("vỡ ối", "rỉ ối"):
        start = source.index(entity)
        end = start + len(entity)
        assert detect_negation(source, start, end) is True, f"{entity!r} should be negated"


def test_combined_family_and_negation_synthetic():
    """Sota_research_report.md §4.3 "combined assertions" example: "Gia đình
    không có tiền sử bệnh tim" must yield isFamily + isNegated (+
    isHistorical, since "tiền sử" also precedes the entity in the same
    clause) all at once."""
    source = "Gia đình không có tiền sử bệnh tim."
    start = source.index("bệnh tim")
    end = start + len("bệnh tim")

    assertions = detect_assertions(source, start, end)
    assert Assertion.FAMILY in assertions
    assert Assertion.NEGATED in assertions
    assert Assertion.HISTORICAL in assertions


def test_post_negation_does_not_leak_across_unrelated_words_1txt():
    """input/input/1.txt line 35: "Bắt đầu dùng metoprolol 25mg po bid,
    không có cải thiện" — "không" there negates "cải thiện" (no
    improvement), not the drug "metoprolol" ~22 characters earlier. A loose
    post-negation window previously mis-tagged this occurrence of
    metoprolol as isNegated; the fix requires the post-negation cue to sit
    immediately after the entity, not just anywhere later in the clause."""
    source = read_input(1)
    start = source.index("metoprolol", source.index("Bắt đầu dùng"))
    end = start + len("metoprolol")

    assert detect_negation(source, start, end) is False


def test_radiology_contrast_idiom_is_not_negation_22txt():
    """input/input/22.txt: "chụp ct sọ não không thuốc cản quang Hình ảnh
    khối máu tụ dưới màng cứng bán cấp hai bên" — "không thuốc cản quang"
    ("without contrast") is an imaging-technique idiom, not a negation of
    the diagnosis that follows it in the same clause."""
    source = read_input(22)
    start = source.index("khối máu tụ dưới màng cứng")
    end = start + len("khối máu tụ dưới màng cứng")

    assert detect_negation(source, start, end) is False
