"""Workstream C (Week 2): quality filter for dataset generation.
Validates entity schemas, character offsets, resolves overlaps, and filters out malformed samples.
"""
from __future__ import annotations

import logging
from typing import Any

from clinlp.schema import ConceptType

logger = logging.getLogger(__name__)

def validate_entity_offsets(text: str, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Verify that:
    1. start and end offsets are within bounds.
    2. text[start:end] exactly matches ent['text'].
    """
    valid_entities = []
    for ent in entities:
        start = ent.get("start")
        end = ent.get("end")
        ent_text = ent.get("text")
        
        if start is None or end is None or ent_text is None:
            logger.warning("Skipping entity with missing offset/text: %s", ent)
            continue
            
        if start < 0 or end > len(text) or start >= end:
            logger.warning("Offset out of bounds: %d-%d for text of length %d", start, end, len(text))
            continue
            
        actual_text = text[start:end]
        if actual_text != ent_text:
            # Try to fix mild casing/whitespace mismatches by searching nearby
            logger.warning(
                "Offset text mismatch: expected '%s' but found '%s' in text[%d:%d]",
                ent_text, actual_text, start, end
            )
            continue
            
        valid_entities.append(ent)
    return valid_entities


def resolve_overlapping_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove overlapping entities. If two entities overlap, keep the longer one."""
    # Sort entities by start offset ascending, then by length descending
    sorted_ents = sorted(entities, key=lambda x: (x["start"], -(x["end"] - x["start"])))
    
    clean_entities: list[dict[str, Any]] = []
    for ent in sorted_ents:
        overlap = False
        start = ent["start"]
        end = ent["end"]
        
        for keep_ent in clean_entities:
            # Check overlap condition: start1 < end2 and start2 < end1
            if start < keep_ent["end"] and keep_ent["start"] < end:
                overlap = True
                logger.info("Resolving overlap: skipping '%s' (%d-%d) in favor of '%s' (%d-%d)",
                            ent["text"], start, end, keep_ent["text"], keep_ent["start"], keep_ent["end"])
                break
                
        if not overlap:
            clean_entities.append(ent)
            
    return clean_entities


def filter_note(note: dict[str, Any]) -> dict[str, Any] | None:
    """Validate and filter a single note. Returns the cleaned note,
    or None if the note is completely invalid.
    """
    text = note.get("text", "")
    if not text.strip():
        return None
        
    entities = note.get("entities", [])
    if not entities:
        # Notes with no entities can be useful negative samples or can be filtered
        return {"text": text, "entities": []}
        
    # 1. Validate offsets
    valid_ents = validate_entity_offsets(text, entities)
    
    # 2. Resolve overlaps
    resolved_ents = resolve_overlapping_entities(valid_ents)
    
    # 3. Validate Concept Types
    final_ents = []
    valid_types = {t.value for t in ConceptType}
    for ent in resolved_ents:
        if ent.get("type") in valid_types:
            final_ents.append(ent)
        else:
            logger.warning("Invalid concept type: %s", ent.get("type"))
            
    return {
        "text": text,
        "entities": final_ents
    }
