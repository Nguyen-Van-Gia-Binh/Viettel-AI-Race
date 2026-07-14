from clinlp.schema import Assertion, Concept, ConceptType
from clinlp.validate import Severity, has_errors, validate_concepts


def test_valid_concepts_have_no_errors():
    source = "Bệnh nhân được chẩn đoán trào ngược dạ dày, dùng omeprazol 20mg."
    concepts = [
        Concept(
            text="trào ngược dạ dày",
            type=ConceptType.CHAN_DOAN,
            position=(25, 42),
            assertions=[],
            candidates=["K21.9"],
        ),
        Concept(
            text="omeprazol 20mg",
            type=ConceptType.THUOC,
            position=(49, 63),
            assertions=[Assertion.HISTORICAL],
            candidates=["7646"],
        ),
    ]
    issues = validate_concepts(source, concepts)
    assert not has_errors(issues)


def test_text_mismatch_at_position_is_an_error():
    source = "Bệnh nhân bị ho khan."
    concepts = [Concept(text="ho khan", type=ConceptType.TRIEU_CHUNG, position=(13, 20))]
    # deliberately shift the position by one so source[start:end] != concept.text
    concepts[0] = concepts[0].model_copy(update={"position": (14, 21)})
    issues = validate_concepts(source, concepts)
    assert has_errors(issues)
    assert any("text mismatch" in i.message for i in issues)


def test_position_out_of_bounds_is_an_error():
    source = "Bệnh nhân bị ho."
    concepts = [Concept(text="ho", type=ConceptType.TRIEU_CHUNG, position=(13, 100))]
    issues = validate_concepts(source, concepts)
    assert has_errors(issues)
    assert any("out of bounds" in i.message for i in issues)


def test_assertion_on_ineligible_type_is_an_error():
    source = "Kết quả xét nghiệm công thức máu"
    concepts = [
        Concept(
            text="công thức máu",
            type=ConceptType.TEN_XET_NGHIEM,
            position=(19, 32),
            assertions=[Assertion.NEGATED],
        )
    ]
    issues = validate_concepts(source, concepts)
    assert has_errors(issues)
    assert any("must not carry assertions" in i.message for i in issues)


def test_candidates_on_ineligible_type_is_an_error():
    source = "ho khan kéo dài"
    concepts = [
        Concept(text="ho khan", type=ConceptType.TRIEU_CHUNG, position=(0, 7), candidates=["R05"])
    ]
    issues = validate_concepts(source, concepts)
    assert has_errors(issues)
    assert any("must not carry candidates" in i.message for i in issues)


def test_missing_candidates_on_eligible_type_is_a_warning_not_error():
    source = "chẩn đoán viêm phổi"
    concepts = [Concept(text="viêm phổi", type=ConceptType.CHAN_DOAN, position=(10, 19), candidates=[])]
    issues = validate_concepts(source, concepts)
    assert not has_errors(issues)
    assert any(i.severity == Severity.WARNING for i in issues)


def test_duplicate_same_type_and_position_is_an_error():
    source = "ho khan ho khan"
    concepts = [
        Concept(text="ho khan", type=ConceptType.TRIEU_CHUNG, position=(0, 7)),
        Concept(text="ho khan", type=ConceptType.TRIEU_CHUNG, position=(0, 7)),
    ]
    issues = validate_concepts(source, concepts)
    assert has_errors(issues)
    assert any("duplicate concept" in i.message for i in issues)


def test_to_json_dict_omits_candidates_for_non_linking_types():
    c = Concept(text="ho khan", type=ConceptType.TRIEU_CHUNG, position=(0, 7))
    d = c.to_json_dict()
    assert "candidates" not in d
    assert d["assertions"] == []


def test_to_json_dict_includes_candidates_for_diagnosis_and_drug():
    c = Concept(text="viêm phổi", type=ConceptType.CHAN_DOAN, position=(0, 9), candidates=["J18.9"])
    d = c.to_json_dict()
    assert d["candidates"] == ["J18.9"]


def test_from_json_dict_roundtrip():
    d = {
        "text": "aspirin 81 mg po daily",
        "type": "THUỐC",
        "candidates": ["243670"],
        "assertions": ["isHistorical"],
        "position": [89, 111],
    }
    c = Concept.from_json_dict(d)
    assert c.to_json_dict() == d
