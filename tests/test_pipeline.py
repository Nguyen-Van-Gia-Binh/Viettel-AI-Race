from clinlp.pipeline import run_pipeline
from clinlp.schema import Assertion, ConceptType
from clinlp.validate import has_errors


def fake_ner(source_text: str):
    yield ("ho khan", ConceptType.TRIEU_CHUNG, [])
    yield ("viêm phổi", ConceptType.CHAN_DOAN, [Assertion.HISTORICAL])
    yield ("không tồn tại trong văn bản", ConceptType.TRIEU_CHUNG, [])  # unresolvable on purpose


def fake_link(entity_text: str, concept_type: ConceptType, source_text: str) -> list[str]:
    return ["J18.9"] if entity_text == "viêm phổi" else []


def test_pipeline_wires_ner_position_and_linking_together():
    source = "Bệnh nhân ho khan, tiền sử viêm phổi."
    result = run_pipeline(source, ner_fn=fake_ner, link_fn=fake_link)

    # the unresolvable mention must be dropped, not fabricated
    assert len(result.concepts) == 2

    by_type = {c.type: c for c in result.concepts}
    symptom = by_type[ConceptType.TRIEU_CHUNG]
    diagnosis = by_type[ConceptType.CHAN_DOAN]

    assert source[symptom.position[0] : symptom.position[1]] == "ho khan"
    assert source[diagnosis.position[0] : diagnosis.position[1]] == "viêm phổi"
    assert diagnosis.candidates == ["J18.9"]
    assert diagnosis.assertions == [Assertion.HISTORICAL]

    assert not has_errors(result.issues)


def test_pipeline_without_link_fn_gives_empty_candidates_not_an_error():
    source = "chẩn đoán viêm phổi"

    def ner(_text):
        yield ("viêm phổi", ConceptType.CHAN_DOAN, [])

    result = run_pipeline(source, ner_fn=ner, link_fn=None)
    assert result.concepts[0].candidates == []
    assert not has_errors(result.issues)  # missing candidates is a warning, not an error
