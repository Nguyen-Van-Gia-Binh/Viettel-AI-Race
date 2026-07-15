"""Workstream C (Week 2): data augmentation for clinical entity linking.
Implements entity swapping, negation flipping, and offset shifts to augment training data.
"""
from __future__ import annotations

import random
from typing import Any

from clinlp.ner.dictionaries import NEGATION_CUES, SYMPTOM_TERMS, DIAGNOSIS_TERMS, DRUG_TERMS, LAB_TEST_TERMS
from clinlp.schema import ConceptType
from clinlp.dataprep.templates import LAB_RESULTS

def swap_entities(note: dict[str, Any]) -> dict[str, Any]:
    """Randomly swap some entities in a note with other entities of the same type.
    Correctly updates the text and shifts offsets of all entities in the note.
    """
    text = note["text"]
    entities = [ent.copy() for ent in note["entities"]]
    
    # Sort entities in reverse order of start offset to perform safe string replacement
    sorted_indices = sorted(range(len(entities)), key=lambda i: entities[i]["start"], reverse=True)
    
    for idx in sorted_indices:
        if random.random() > 0.5:
            continue
            
        ent = entities[idx]
        start = ent["start"]
        end = ent["end"]
        ent_text = ent["text"]
        concept_type = ent["type"]
        
        # Pick a new entity text
        if concept_type == ConceptType.TRIEU_CHUNG.value:
            new_text = random.choice(SYMPTOM_TERMS)
        elif concept_type == ConceptType.CHAN_DOAN.value:
            new_text = random.choice(DIAGNOSIS_TERMS)
        elif concept_type == ConceptType.THUOC.value:
            new_text = random.choice(DRUG_TERMS)
        elif concept_type == ConceptType.TEN_XET_NGHIEM.value:
            new_text = random.choice(LAB_TEST_TERMS)
        elif concept_type == ConceptType.KET_QUA_XET_NGHIEM.value:
            new_text = random.choice(LAB_RESULTS)
        else:
            continue
            
        # Update text
        text = text[:start] + new_text + text[end:]
        diff = len(new_text) - len(ent_text)
        
        # Update current entity
        ent["text"] = new_text
        ent["end"] = start + len(new_text)
        
        # Shift all subsequent entities (those starting after current entity)
        for other_ent in entities:
            if other_ent is not ent and other_ent["start"] > start:
                other_ent["start"] += diff
                other_ent["end"] += diff
                
    return {
        "text": text,
        "entities": entities
    }


def flip_negations(note: dict[str, Any]) -> dict[str, Any]:
    """Find a symptom or diagnosis, and randomly flip its negation status.
    If it is negated, make it positive by removing the negation cue.
    If it is positive, insert a negation cue before it.
    Updates the text and shifts all entity offsets accordingly.
    """
    text = note["text"]
    entities = [ent.copy() for ent in note["entities"]]
    
    # Sort descending
    sorted_indices = sorted(range(len(entities)), key=lambda i: entities[i]["start"], reverse=True)
    
    for idx in sorted_indices:
        if random.random() > 0.4:
            continue
            
        ent = entities[idx]
        start = ent["start"]
        end = ent["end"]
        assertions = ent.setdefault("assertions", {"isNegated": False, "isFamily": False, "isHistorical": False})
        
        if assertions.get("isNegated"):
            # Try to remove a negation cue.
            prefix = text[:start]
            found_cue = None
            for cue in NEGATION_CUES["pre_negation"]:
                if prefix.rstrip().endswith(cue):
                    found_cue = cue
                    break
            
            if found_cue:
                # Remove the cue and space
                cue_start = prefix.rstrip().rfind(found_cue)
                text = text[:cue_start] + text[start:]
                diff = -(start - cue_start)
                
                assertions["isNegated"] = False
                
                # Shift all entities starting from or after the original start
                for other_ent in entities:
                    if other_ent["start"] >= start:
                        other_ent["start"] += diff
                        other_ent["end"] += diff
        else:
            # Add a negation cue before the entity
            cue = "không "
            text = text[:start] + cue + text[start:]
            diff = len(cue)
            
            assertions["isNegated"] = True
            
            # Shift all entities starting from or after the original start
            for other_ent in entities:
                if other_ent["start"] >= start:
                    other_ent["start"] += diff
                    other_ent["end"] += diff
                    
    return {
        "text": text,
        "entities": entities
    }
