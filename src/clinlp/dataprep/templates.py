"""Workstream C (Week 2): template/slot-filler synthetic note generator.
Constructs clinical notes segment-by-segment to calculate exact entity character offsets.
"""
from __future__ import annotations

import random
from typing import Any

from clinlp.ner.dictionaries import SYMPTOM_TERMS, DIAGNOSIS_TERMS, DRUG_TERMS, LAB_TEST_TERMS
from clinlp.schema import ConceptType

# Lab Test Results (KET_QUA_XET_NGHIEM)
LAB_RESULTS: list[str] = [
    "bình thường", "trong giới hạn bình thường", "âm tính", "tăng nhẹ", "giảm", "5.2 mmol/L", "98%", "dương tính"
]

# Mapping templates to segments
# Each template is a list of either static strings or tuple (concept_type, key_in_assertions_dict, candidates_list)
# assertions_dict will hold keys: isNegated, isFamily, isHistorical
TEMPLATES = [
    [
        "Bệnh nhân nhập viện vì ",
        (ConceptType.TRIEU_CHUNG, None, None), # standard symptom, no special assertions
        ", có tiền sử ",
        (ConceptType.CHAN_DOAN, "isHistorical", None), # historical diagnosis
        ". Khám lâm sàng không ghi nhận ",
        (ConceptType.TRIEU_CHUNG, "isNegated", None), # negated symptom
        "."
    ],
    [
        "Gia đình có tiền sử ",
        (ConceptType.CHAN_DOAN, "isFamily", None), # family diagnosis
        ". Hiện tại bệnh nhân đang dùng ",
        (ConceptType.THUOC, None, None), # drug
        " liều hàng ngày."
    ],
    [
        "Kết quả xét nghiệm ",
        (ConceptType.TEN_XET_NGHIEM, None, None), # test name
        " cho thấy ",
        (ConceptType.KET_QUA_XET_NGHIEM, None, None), # test result
        "."
    ],
    [
        "Khám phổi không phát hiện dấu hiệu của ",
        (ConceptType.CHAN_DOAN, "isNegated", None), # negated diagnosis
        ", tiền sử trước đây từng bị ",
        (ConceptType.CHAN_DOAN, "isHistorical", None), # historical diagnosis
        " nhưng đã điều trị khỏi."
    ],
    [
        "Được chẩn đoán xác định là ",
        (ConceptType.CHAN_DOAN, None, None), # diagnosis
        ", kê đơn điều trị bằng ",
        (ConceptType.THUOC, None, None), # drug
        "."
    ]
]


def _get_random_entity_for_type(concept_type: ConceptType) -> tuple[str, list[str]]:
    """Return a random entity text and its candidates codes list."""
    if concept_type == ConceptType.TRIEU_CHUNG:
        return random.choice(SYMPTOM_TERMS), []
    elif concept_type == ConceptType.CHAN_DOAN:
        return random.choice(DIAGNOSIS_TERMS), []
    elif concept_type == ConceptType.THUOC:
        return random.choice(DRUG_TERMS), []
    elif concept_type == ConceptType.TEN_XET_NGHIEM:
        return random.choice(LAB_TEST_TERMS), []
    elif concept_type == ConceptType.KET_QUA_XET_NGHIEM:
        return random.choice(LAB_RESULTS), []
    return "bệnh", []


def generate_single_note() -> dict[str, Any]:
    """Assemble a single note segment-by-segment and keep track of character offsets."""
    template = random.choice(TEMPLATES)
    text = ""
    entities = []
    
    for segment in template:
        if isinstance(segment, str):
            text += segment
        elif isinstance(segment, tuple):
            concept_type, assertion_key, candidates = segment
            entity_text, entity_codes = _get_random_entity_for_type(concept_type)
            
            start = len(text)
            text += entity_text
            end = len(text)
            
            assertions = {
                "isNegated": False,
                "isFamily": False,
                "isHistorical": False
            }
            if assertion_key:
                assertions[assertion_key] = True
                
            entities.append({
                "start": start,
                "end": end,
                "text": entity_text,
                "type": concept_type.value,
                "assertions": assertions,
                "candidates": entity_codes
            })
            
    return {
        "text": text,
        "entities": entities
    }


def generate_templated_notes(num_notes: int) -> list[dict[str, Any]]:
    """Generate multiple templated notes."""
    return [generate_single_note() for _ in range(num_notes)]
