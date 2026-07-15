"""Unit tests for Workstream C (Week 2): data preparation, augmentation, and validation.
"""
from __future__ import annotations

import pytest
from clinlp.dataprep.templates import generate_single_note, generate_templated_notes
from clinlp.dataprep.synth_generate import parse_tagged_text
from clinlp.dataprep.translate_adapt import convert_english_to_tagged
from clinlp.dataprep.augment_testset import swap_entities, flip_negations
from clinlp.dataprep.quality_filter import filter_note, validate_entity_offsets, resolve_overlapping_entities


def test_xml_tag_parser():
    """Verify that parse_tagged_text correctly extracts entities and computes character offsets."""
    tagged = 'Bệnh nhân không có <entity type="TRIỆU_CHỨNG" neg="true">khó thở</entity>, tiền sử <entity type="CHẨN_ĐOÁN" hist="true" codes="I10">tăng huyết áp</entity>.'
    parsed = parse_tagged_text(tagged)
    
    # Check text is cleaned correctly
    assert parsed["text"] == "Bệnh nhân không có khó thở, tiền sử tăng huyết áp."
    
    # Check entities count
    assert len(parsed["entities"]) == 2
    
    # Entity 1: khó thở (start: 19, end: 26)
    ent1 = parsed["entities"][0]
    assert ent1["text"] == "khó thở"
    assert ent1["type"] == "TRIỆU_CHỨNG"
    assert ent1["assertions"]["isNegated"] is True
    assert ent1["assertions"]["isFamily"] is False
    assert ent1["assertions"]["isHistorical"] is False
    assert parsed["text"][ent1["start"]:ent1["end"]] == "khó thở"
    
    # Entity 2: tăng huyết áp (start: 37, end: 50)
    ent2 = parsed["entities"][1]
    assert ent2["text"] == "tăng huyết áp"
    assert ent2["type"] == "CHẨN_ĐOÁN"
    assert ent2["assertions"]["isNegated"] is False
    assert ent2["assertions"]["isHistorical"] is True
    assert ent2["candidates"] == ["I10"]
    assert parsed["text"][ent2["start"]:ent2["end"]] == "tăng huyết áp"


def test_convert_english_to_tagged():
    """Verify that convert_english_to_tagged converts English text + offsets to tagged XML format."""
    text = "Patient has chest pain and no diabetes."
    entities = [
        {
            "start": 12,
            "end": 22,
            "text": "chest pain",
            "type": "TRIỆU_CHỨNG",
            "assertions": {"isNegated": False}
        },
        {
            "start": 30,
            "end": 38,
            "text": "diabetes",
            "type": "CHẨN_ĐOÁN",
            "assertions": {"isNegated": True}
        }
    ]
    tagged = convert_english_to_tagged(text, entities)
    assert '<entity type="TRIỆU_CHỨNG" neg="false" fam="false" hist="false">chest pain</entity>' in tagged
    assert '<entity type="CHẨN_ĐOÁN" neg="true" fam="false" hist="false">diabetes</entity>' in tagged


def test_templated_generation():
    """Verify templated notes are generated with valid structure and exact offsets."""
    note = generate_single_note()
    assert "text" in note
    assert "entities" in note
    
    # Run offset checks
    text = note["text"]
    for ent in note["entities"]:
        assert text[ent["start"]:ent["end"]] == ent["text"]


def test_entity_swapping():
    """Verify that entity swapping shifts offsets of other entities correctly."""
    note = {
        "text": "Bệnh nhân bị đau ngực và ho.",
        "entities": [
            {
                "start": 13,
                "end": 21,
                "text": "đau ngực",
                "type": "TRIỆU_CHỨNG",
                "assertions": {"isNegated": False}
            },
            {
                "start": 25,
                "end": 27,
                "text": "ho",
                "type": "TRIỆU_CHỨNG",
                "assertions": {"isNegated": False}
            }
        ]
    }
    
    # We force the swap by mocking random.random to always swap
    import random
    orig_random = random.random
    try:
        random.random = lambda: 0.0  # Force swap (since 0.0 < 0.5)
        swapped = swap_entities(note)
        
        # Verify offsets match modified text
        text = swapped["text"]
        for ent in swapped["entities"]:
            assert text[ent["start"]:ent["end"]] == ent["text"]
    finally:
        random.random = orig_random


def test_negation_flipping():
    """Verify that negation flipping adds or removes cues and shifts offsets correctly."""
    note = {
        "text": "Bệnh nhân không khó thở.",
        "entities": [
            {
                "start": 16,
                "end": 23,
                "text": "khó thở",
                "type": "TRIỆU_CHỨNG",
                "assertions": {"isNegated": True}
            }
        ]
    }
    
    import random
    orig_random = random.random
    try:
        random.random = lambda: 0.0  # Force flip
        flipped = flip_negations(note)
        
        text = flipped["text"]
        for ent in flipped["entities"]:
            assert text[ent["start"]:ent["end"]] == ent["text"]
            assert ent["assertions"]["isNegated"] is False
    finally:
        random.random = orig_random


def test_quality_filter():
    """Test offset validation, overlap resolution, and global filter note logic."""
    text = "Bệnh nhân ho và đau ngực."
    entities = [
        # Correct entity
        {"start": 10, "end": 12, "text": "ho", "type": "TRIỆU_CHỨNG"},
        # Overlapping entity (keep longer one)
        {"start": 16, "end": 24, "text": "đau ngực", "type": "TRIỆU_CHỨNG"},
        {"start": 16, "end": 19, "text": "đau", "type": "TRIỆU_CHỨNG"},
        # Out of bounds entity
        {"start": 50, "end": 60, "text": "invalid", "type": "TRIỆU_CHỨNG"}
    ]
    
    valid_ents = validate_entity_offsets(text, entities)
    assert len(valid_ents) == 3  # The out of bounds one is filtered
    
    resolved = resolve_overlapping_entities(valid_ents)
    assert len(resolved) == 2  # The overlapping 'đau' is removed in favor of 'đau ngực'
    
    # Check that we kept 'ho' and 'đau ngực'
    texts = {e["text"] for e in resolved}
    assert "ho" in texts
    assert "đau ngực" in texts
