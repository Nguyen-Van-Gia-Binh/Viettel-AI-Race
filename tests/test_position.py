"""Position-resolver tests grounded in real edge cases from input/input/*.txt —
not synthetic examples, so a fix here is a fix for a case that actually occurs
in the competition corpus.
"""
import os

from clinlp.position import (
    PositionResolver,
    find_all_exact,
    find_all_unicode_flexible,
    find_all_whitespace_flexible,
)

INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "input", "input")


def read_input(n: int) -> str:
    with open(os.path.join(INPUT_DIR, f"{n}.txt"), encoding="utf-8", newline="") as fh:
        return fh.read()


def test_exact_match_basic():
    source = "Bệnh nhân bị ho đờm xanh."
    spans = find_all_exact(source, "ho đờm xanh")
    assert spans == [(13, 24)]
    assert source[13:24] == "ho đờm xanh"


def test_duplicate_entity_gets_distinct_occurrences_1txt():
    """input/input/1.txt repeats "đánh trống ngực" and "khó thở" many times
    (C9 in Topic_Analysis.md) — each mention must resolve to its own span, in
    order, not all collapse onto the first occurrence."""
    source = read_input(1)
    resolver = PositionResolver(source)

    occurrences = find_all_exact(source, "đánh trống ngực")
    assert len(occurrences) > 3  # confirms this really is a repeated-entity case

    resolved = [resolver.resolve("đánh trống ngực") for _ in range(len(occurrences))]
    assert all(s is not None for s in resolved)
    assert len(set(resolved)) == len(resolved)  # every occurrence is distinct
    assert resolved == occurrences  # resolved in left-to-right order

    for start, end in resolved:
        assert source[start:end] == "đánh trống ngực"

    # a further request beyond the real number of occurrences must fail loudly,
    # never silently reuse an already-assigned span
    assert resolver.resolve("đánh trống ngực") is None


def test_missing_space_line_wrap_join_31txt():
    """input/input/31.txt line 1 reads "...Khởi phát chuyển dạBệnh sử: Thai..."
    — no space between "chuyển dạ" and "Bệnh sử". Both halves must still
    resolve to their own exact, correct spans since each is itself a real
    substring of the raw file."""
    source = read_input(31)
    assert "chuyển dạBệnh sử" in source

    resolver = PositionResolver(source)
    span1 = resolver.resolve("chuyển dạ")
    span2 = resolver.resolve("Bệnh sử")

    assert span1 is not None and span2 is not None
    assert source[span1[0] : span1[1]] == "chuyển dạ"
    assert source[span2[0] : span2[1]] == "Bệnh sử"
    assert span1[1] == span2[0]  # confirms they are adjacent with no gap


def test_dense_vitals_token_1txt():
    """input/input/1.txt line 48: "VS98.3 12987 56 18 99RA" — a dense
    whitespace-separated vitals token with no delimiters between fields.
    Position calculation must not choke on the lack of internal structure."""
    source = read_input(1)
    entity = "VS98.3 12987 56 18 99RA"
    assert entity in source

    resolver = PositionResolver(source)
    span = resolver.resolve(entity)
    assert span is not None
    assert source[span[0] : span[1]] == entity


def test_missing_space_before_khong_con_1txt():
    """input/input/1.txt line 49: "...bệnh nhân không còn cảm giáckhó chịu vùng
    ngực" — no space between "cảm giác" and "khó chịu". The full downstream
    phrase "khó chịu vùng ngực" is still an exact substring (it also occurs
    once more, negated, on line 27) and must resolve without any fallback tier."""
    source = read_input(1)
    assert "cảm giáckhó chịu vùng ngực" in source

    spans = find_all_exact(source, "khó chịu vùng ngực")
    assert len(spans) == 2  # line 27 ("không có khó chịu...") and line 49 (the join)
    for start, end in spans:
        assert source[start:end] == "khó chịu vùng ngực"

    join_idx = source.index("cảm giáckhó chịu vùng ngực") + len("cảm giác")
    assert (join_idx, join_idx + len("khó chịu vùng ngực")) in spans


def test_whitespace_flexible_fallback_for_extra_spaces():
    """A model might emit a span with collapsed/extra whitespace relative to
    the raw file (e.g. input/input/1.txt line 7 has "có   Căng thẳng  nhiều")."""
    source = "bệnh nhân có   Căng thẳng  nhiều trong công việc"
    entity = "có Căng thẳng nhiều"  # single-spaced, unlike the raw multi-spaced text
    assert find_all_exact(source, entity) == []  # exact match must fail first

    spans = find_all_whitespace_flexible(source, entity)
    assert len(spans) == 1
    start, end = spans[0]
    assert source[start:end] == "có   Căng thẳng  nhiều"


def test_unicode_flexible_fallback_for_nfd_vs_nfc():
    """Composed (NFC) vs decomposed (NFD) Vietnamese diacritics must still
    resolve to the same raw offsets."""
    import unicodedata

    nfc_word = "khó"
    nfd_word = unicodedata.normalize("NFD", nfc_word)
    assert nfc_word != nfd_word  # confirms the two forms really do differ

    source = f"bệnh nhân {nfd_word} thở"
    assert find_all_exact(source, nfc_word) == []  # exact match must fail first

    spans = find_all_unicode_flexible(source, nfc_word)
    assert len(spans) == 1
    start, end = spans[0]
    assert unicodedata.normalize("NFC", source[start:end]) == nfc_word


def test_unresolvable_entity_returns_none_not_a_guess():
    resolver = PositionResolver("bệnh nhân bị ho")
    assert resolver.resolve("sốt cao 39 độ") is None
