"""Workstream C (Week 2): i2b2/n2c2 dataset translation and adaptation.
Converts annotated English datasets to tagged text, translates them using the Teacher LLM
while keeping entity tags intact, and parses the result to calculate Vietnamese offsets.
"""
from __future__ import annotations

import logging
from typing import Any

from clinlp.dataprep.teacher_client import TeacherClient
from clinlp.dataprep.synth_generate import parse_tagged_text

logger = logging.getLogger(__name__)

def convert_english_to_tagged(text: str, entities: list[dict[str, Any]]) -> str:
    """Insert XML tags into an English note based on entity offsets.
    Sorts entities in reverse order to keep offsets valid during string modification.
    """
    sorted_ents = sorted(entities, key=lambda x: x["start"], reverse=True)
    tagged_text = text
    
    for ent in sorted_ents:
        start = ent["start"]
        end = ent["end"]
        ent_text = ent["text"]
        concept_type = ent["type"]
        
        # Format assertions
        assertions = ent.get("assertions", {})
        neg = "true" if assertions.get("isNegated") else "false"
        fam = "true" if assertions.get("isFamily") else "false"
        hist = "true" if assertions.get("isHistorical") else "false"
        
        tag = f'<entity type="{concept_type}" neg="{neg}" fam="{fam}" hist="{hist}">{ent_text}</entity>'
        tagged_text = tagged_text[:start] + tag + tagged_text[end:]
        
    return tagged_text


def translate_tagged_note(client: TeacherClient, tagged_english_text: str) -> dict[str, Any]:
    """Translate a tagged English note into Vietnamese, instructing the LLM
    to preserve the <entity> tags around translated medical terms.
    """
    prompt = f"""Dưới đây là một đoạn bệnh án y khoa bằng tiếng Anh đã được gắn thẻ thực thể y tế bằng thẻ XML `<entity>`.
Hãy dịch toàn bộ văn bản này sang tiếng Việt y khoa tự nhiên.
QUAN TRỌNG: Bạn PHẢI giữ nguyên các thẻ `<entity>` và thuộc tính của chúng (type, neg, fam, hist, codes) xung quanh các thuật ngữ y tế tương ứng sau khi dịch. Không thay đổi tên thuộc tính hoặc cấu trúc thẻ.

Bệnh án gốc:
{tagged_english_text}

Bản dịch tiếng Việt (chứa đầy đủ các thẻ XML):
"""
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    try:
        response_text = client.chat_completion(messages)
        # Parse the translated tagged text
        return parse_tagged_text(response_text.strip())
    except Exception as e:
        logger.error("Failed to translate tagged note: %s", e)
        # Fallback to an empty schema
        return {"text": "", "entities": []}


def adapt_external_dataset(
    client: TeacherClient,
    english_notes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Translate a batch of English notes with annotations to Vietnamese."""
    vietnamese_notes = []
    for note in english_notes:
        try:
            tagged = convert_english_to_tagged(note["text"], note["entities"])
            translated = translate_tagged_note(client, tagged)
            if translated["entities"]:
                vietnamese_notes.append(translated)
        except Exception as e:
            logger.error("Failed to adapt note: %s. Error: %s", note, e)
    return vietnamese_notes
