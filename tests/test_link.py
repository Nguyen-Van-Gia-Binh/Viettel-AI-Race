"""`linking/link.py` entity-linking tests (Workstream A).

Uses the already-built `data/ICD-10/icd10_processed.csv` and
`data/RxNorm/rxnorm_processed.csv` directly (real KB data, not a mock/fixture)
against the worked examples given verbatim in `Topic.md`:
"bệnh trào ngược dạ dày - thực quản" -> K21.0/K21.9; "Chlorpheniramine 0.4
MG/ML" -> RxNorm 360047; "Capsaicin 0.38 MG/ML" -> RxNorm 1660761 (the latter
two are historical/remapped RxCUIs absent from the live bulk API and only
resolvable via `rxnorm_kb.py`'s documented historical-supplement patch — see
that module's docstring).
"""
from clinlp.linking.link import link
from clinlp.schema import ConceptType


def test_link_gerd_diagnosis_topic_md_example():
    """Topic.md: "bệnh trào ngược dạ dày - thực quản" -> mã ICD bao gồm
    K21.0, K21.9."""
    candidates = link("bệnh trào ngược dạ dày - thực quản", ConceptType.CHAN_DOAN, "")
    assert "K21.0" in candidates
    assert "K21.9" in candidates


def test_link_chlorpheniramine_topic_md_example():
    """Topic.md: "Chlorpheniramine 0.4 MG/ML" -> RxNorm 360047."""
    candidates = link("Chlorpheniramine 0.4 MG/ML", ConceptType.THUOC, "")
    assert "360047" in candidates


def test_link_capsaicin_topic_md_example():
    """Topic.md: "Capsaicin 0.38 MG/ML" -> RxNorm 1660761. This RxCUI is
    `Obsolete` and absent from the live RxNav `allconcepts` bulk endpoint
    (see `linking/rxnorm_kb.py` module docstring) — resolvable only because
    `build_rxnorm_kb()` patches it in via a targeted historystatus lookup."""
    candidates = link("Capsaicin 0.38 MG/ML", ConceptType.THUOC, "")
    assert "1660761" in candidates


def test_link_common_diagnoses_seen_in_corpus():
    """Additional diagnosis terms with a seeded `vi_aliases` entry in
    `icd10_kb.py`'s `VI_ALIASES` table."""
    assert "I10" in link("tăng huyết áp", ConceptType.CHAN_DOAN, "")
    assert "E11.9" in link("đái tháo đường type 2", ConceptType.CHAN_DOAN, "")
    assert "J18.9" in link("viêm phổi", ConceptType.CHAN_DOAN, "")


def test_link_ineligible_type_returns_empty():
    """`link()` is only ever called by `pipeline.py` for CHẨN_ĐOÁN/THUỐC
    (schema.py's CANDIDATE_ELIGIBLE_TYPES), but must degrade to an empty list
    rather than error if called with another type directly."""
    assert link("sốt cao", ConceptType.TRIEU_CHUNG, "") == []


def test_link_empty_entity_text_returns_empty():
    assert link("", ConceptType.CHAN_DOAN, "") == []
    assert link("   ", ConceptType.THUOC, "") == []


def test_link_returns_at_most_top_k():
    candidates = link("viêm phổi", ConceptType.CHAN_DOAN, "", top_k=3)
    assert len(candidates) <= 3
