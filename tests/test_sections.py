"""Section-header parser tests grounded in real cases from input/input/*.txt —
mirrors the style of tests/test_position.py (docstrings cite the real
file/line that motivates each case)."""
import os

from clinlp.sections import build_section_index, parse_sections

INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "input", "input")


def read_input(n: int) -> str:
    with open(os.path.join(INPUT_DIR, f"{n}.txt"), encoding="utf-8", newline="") as fh:
        return fh.read()


def test_numbered_historical_section_propagates_to_inline_drug_line_22txt():
    """input/input/22.txt line 3: "Thuốc trước khi nhập viện: aspirin 325mg
    hằng ngày" sits under the numbered header "1. Tiền sử bệnh" (line 1) —
    that line itself is a "Label: value" line (has a colon), so it is never
    detected as its own header; the aspirin mention must instead inherit
    isHistorical from the enclosing numbered section."""
    source = read_input(22)
    idx = build_section_index(source)

    aspirin_pos = source.index("aspirin")
    section = idx.section_at(aspirin_pos)

    assert section is not None
    assert section.label == "Tiền sử bệnh"
    assert section.is_historical is True
    assert idx.is_historical_at(aspirin_pos) is True


def test_present_illness_header_is_not_historical_1txt():
    """input/input/1.txt line 12: "2.  Tiền sử bệnh hiện tại" is History-of-
    Present-Illness (current symptoms listed below it, e.g. "đánh trống
    ngực" on line 17) even though the header text literally contains the
    substring "tiền sử" — the "hiện tại" (current) marker must suppress the
    historical classification, or every current symptom in that section
    would be wrongly tagged isHistorical."""
    source = read_input(1)
    idx = build_section_index(source)

    # "đánh trống ngực" under "Các triệu chứng hiện tại" (line 17), inside
    # section 2, not section 1's "Thuốc trước khi nhập viện".
    second_mention = source.index("đánh trống ngực", source.index("Các triệu chứng hiện tại"))
    section = idx.section_at(second_mention)

    assert section is not None
    assert "hiện tại" in section.label.lower()
    assert section.is_historical is False
    assert idx.is_historical_at(second_mention) is False


def test_standalone_subheader_thuoc_truoc_khi_nhap_vien_1txt():
    """input/input/1.txt line 2: "Thuốc trước khi nhập viện" is a standalone
    header line (no colon, own line) inside section 1, unlike the inline
    "Label: value" form seen in 22.txt. It must be detected as its own
    header and propagate isHistorical to metoprolol/doxycycline/atenolol
    listed under it (lines 3-5)."""
    source = read_input(1)
    sections = parse_sections(source)

    labels = [s.label for s in sections]
    assert "Thuốc trước khi nhập viện" in labels

    idx = build_section_index(source)
    for drug in ("metoprolol", "doxycycline", "atenolol"):
        pos = source.index(drug)
        assert idx.is_historical_at(pos) is True, f"{drug!r} should be inside the historical drug section"


def test_section_at_returns_none_before_first_header():
    source = "some preamble\n1. Tiền sử bệnh\ncontent here\n"
    idx = build_section_index(source)
    assert idx.section_at(0) is None
    assert idx.is_historical_at(0) is False


def test_offsets_are_exact_substrings_of_source():
    """Every parsed section's [start, end) must be a valid slice of the raw
    source — guards against off-by-one errors in the line-offset bookkeeping."""
    source = read_input(22)
    for s in parse_sections(source):
        assert 0 <= s.start <= s.end <= len(source)
