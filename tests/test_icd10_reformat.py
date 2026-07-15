"""ICD-10 code reformatting + KB build tests (Workstream A).

`reformat_icd10_code()` cases are grounded in `Topic.md`'s own worked example
(K21.0/K21.9) plus the boundary cases called out in
`src/clinlp/linking/icd10_kb.py`'s docstring (3-char category codes with no
dot, and the longest real ICD-10-CM code shape, 7 chars before any
extension). The KB-load test reads the actual `data/ICD-10/icd10_processed.csv`
already built from the real CMS file rather than a synthetic fixture, per
CLAUDE.md's "extend this pattern rather than writing purely synthetic test
cases".
"""
from clinlp.linking.icd10_kb import DEFAULT_OUT_CSV, load_icd10_kb, reformat_icd10_code


def test_reformat_topic_md_example():
    """Topic.md's own worked example: "K219" -> "K21.9"."""
    assert reformat_icd10_code("K219") == "K21.9"
    assert reformat_icd10_code("K210") == "K21.0"


def test_reformat_short_category_code_unchanged():
    """A 3-char category/header code (e.g. "A00") has nothing to put after a
    dot — `icd10_kb.py`'s docstring documents this as intentionally left
    unchanged, not padded or errored on."""
    assert reformat_icd10_code("A00") == "A00"
    assert reformat_icd10_code("I10") == "I10"


def test_reformat_longest_code_shape():
    """Longest real ICD-10-CM code shape cited in `icd10_kb.py`'s docstring:
    up to 4 chars after the dot (e.g. laterality/episode-of-care suffixes)."""
    assert reformat_icd10_code("S52209K") == "S52.209K"


def test_reformat_strips_whitespace():
    assert reformat_icd10_code("  K219  ") == "K21.9"


def test_processed_csv_has_gerd_codes():
    """`data/ICD-10/icd10_processed.csv` (already built from the real CMS
    order file) must contain both codes from Topic.md's GERD example, dotted
    and with the seeded Vietnamese alias attached."""
    rows = load_icd10_kb(DEFAULT_OUT_CSV)
    by_code = {r["code"]: r for r in rows}

    assert "K21.0" in by_code
    assert "K21.9" in by_code
    for code in ("K21.0", "K21.9"):
        assert "GERD" in by_code[code]["vi_aliases"] or "trào ngược" in by_code[code]["vi_aliases"]


def test_processed_csv_no_undotted_codes_leak_through():
    """Every code longer than 3 chars in the processed CSV must have been
    dotted — catches a regression where `reformat_icd10_code` is skipped for
    some row."""
    rows = load_icd10_kb(DEFAULT_OUT_CSV)
    offenders = [r["code"] for r in rows if len(r["code"]) > 3 and "." not in r["code"]]
    assert offenders == [], f"found {len(offenders)} undotted long codes, e.g. {offenders[:5]}"
