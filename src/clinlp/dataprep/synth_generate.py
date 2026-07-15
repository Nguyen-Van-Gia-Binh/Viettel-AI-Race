"""Workstream C (Week 2): Teacher-based synthetic note generator.
Prompts the Teacher LLM to write realistic Vietnamese clinical notes containing XML tags
for entities and attributes, then parses the XML to calculate exact character offsets.
"""
from __future__ import annotations

import re
import json
import logging
from typing import Any
import xml.etree.ElementTree as ET

from clinlp.dataprep.teacher_client import TeacherClient
from clinlp.schema import ConceptType

logger = logging.getLogger(__name__)

# Regex pattern to extract <entity> tags
ENTITY_PATTERN = re.compile(r'<entity\s+([^>]+)>(.*?)</entity>', re.DOTALL)

def parse_tagged_text(tagged_text: str) -> dict[str, Any]:
    """Parse text containing <entity type="X" neg="true">text</entity> tags,
    strip the tags to get clean text, and calculate correct character offsets.
    """
    clean_text = ""
    entities = []
    
    # We walk through the text and parse tags sequentially
    last_idx = 0
    for match in ENTITY_PATTERN.finditer(tagged_text):
        # Add static text before the match
        clean_text += tagged_text[last_idx:match.start()]
        
        attr_str = match.group(1)
        entity_content = match.group(2)
        
        # Parse attributes from attr_str (e.g. type="TRIỆU_CHỨNG" neg="true")
        # A simple regex to parse key-value pairs
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', attr_str))
        
        start = len(clean_text)
        clean_text += entity_content
        end = len(clean_text)
        
        # Determine assertions
        is_negated = attrs.get("neg", "false").lower() == "true"
        is_family = attrs.get("fam", "false").lower() == "true"
        is_historical = attrs.get("hist", "false").lower() == "true"
        
        # Concept type conversion (case insensitive matching)
        raw_type = attrs.get("type", "").upper()
        concept_type = ConceptType.TRIEU_CHUNG.value
        for t in ConceptType:
            if t.value.upper() == raw_type or t.name.upper() == raw_type:
                concept_type = t.value
                break
                
        # Parse candidates
        candidates_str = attrs.get("codes", "")
        candidates = [c.strip() for c in candidates_str.split(",") if c.strip()]
        
        entities.append({
            "start": start,
            "end": end,
            "text": entity_content,
            "type": concept_type,
            "assertions": {
                "isNegated": is_negated,
                "isFamily": is_family,
                "isHistorical": is_historical
            },
            "candidates": candidates
        })
        
        last_idx = match.end()
        
    # Append the remaining static text
    clean_text += tagged_text[last_idx:]
    
    return {
        "text": clean_text,
        "entities": entities
    }


def generate_teacher_notes(client: TeacherClient, count: int = 5) -> list[dict[str, Any]]:
    """Prompt the teacher LLM to generate synthetic clinical notes containing XML tags."""
    prompt = """Bạn là một chuyên gia y tế và kỹ sư dữ liệu y khoa lâm sàng tiếng Việt.
Hãy viết một văn bản bệnh án y khoa tiếng Việt thực tế và gắn thẻ các thực thể lâm sàng bằng thẻ XML `<entity>`.

Cấu trúc thẻ như sau:
`<entity type="[LOẠI]" neg="[true/false]" fam="[true/false]" hist="[true/false]" codes="[MÃ]">Tên thực thể</entity>`

Trong đó:
- `type` nhận một trong năm giá trị: "TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC", "QUY_TRÌNH", "BỘ_PHẬN_CƠ_THỂ".
- `neg` đặt thành "true" nếu thực thể bị phủ định (ví dụ: "không ho", "không đau ngực").
- `fam` đặt thành "true" nếu liên quan đến tiền sử gia đình (ví dụ: "bố bị tăng huyết áp").
- `hist` đặt thành "true" nếu là tiền sử bệnh nền cũ của bản thân (ví dụ: "tiền sử đái tháo đường").
- `codes` chứa mã ICD-10 hoặc mã RxCUI nếu có (ví dụ: codes="J18.9" hoặc codes="360047"). Để trống nếu không rõ mã.

Ví dụ mẫu:
Bệnh nhân nam 54 tuổi, nhập viện vì <entity type="TRIỆU_CHỨNG">đau ngực</entity> dữ dội lan ra <entity type="BỘ_PHẬN_CƠ_THỂ">vai trái</entity>. Tiền sử bản thân ghi nhận <entity type="CHẨN_ĐOÁN" hist="true" codes="I10">tăng huyết áp</entity> điều trị bằng <entity type="THUỐC" hist="true" codes="360047">amlodipine</entity>. Không ghi nhận <entity type="TRIỆU_CHỨNG" neg="true">khó thở</entity>. Gia đình có bố bị <entity type="CHẨN_ĐOÁN" fam="true" codes="I25.9">bệnh mạch vành</entity>.

Hãy viết chính xác {count} đoạn văn bản bệnh án độc lập khác nhau, mỗi đoạn cách nhau bởi dòng '---'.
Hãy đảm bảo có độ đa dạng cao, chứa đủ các trường hợp phủ định (negation), tiền sử gia đình (family), tiền sử cá nhân (historical), các tên thuốc khác nhau và chẩn đoán khác nhau.
"""
    
    messages = [
        {"role": "user", "content": prompt.format(count=count)}
    ]
    
    try:
        response_text = client.chat_completion(messages)
        notes = []
        # Split notes by '---'
        sections = response_text.split("---")
        for sec in sections:
            sec_clean = sec.strip()
            if not sec_clean:
                continue
            try:
                parsed = parse_tagged_text(sec_clean)
                if parsed["entities"]:
                    notes.append(parsed)
            except Exception as e:
                logger.error("Failed to parse section: %s. Error: %s", sec_clean, e)
        return notes
    except Exception as e:
        logger.error("TeacherClient failed to generate notes: %s", e)
        return []
