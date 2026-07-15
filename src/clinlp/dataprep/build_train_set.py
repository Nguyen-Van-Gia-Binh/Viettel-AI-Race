"""Workstream C (Week 2): train set aggregation and coverage reporting.
Combines multiple synthetic data sources (templated, generated, augmented),
filters them using the quality filter, and saves to `data/synthetic/train_final.jsonl`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from clinlp.dataprep.templates import generate_templated_notes
from clinlp.dataprep.synth_generate import generate_teacher_notes
from clinlp.dataprep.teacher_client import TeacherClient
from clinlp.dataprep.augment_testset import swap_entities, flip_negations
from clinlp.dataprep.quality_filter import filter_note

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_PATH = REPO_ROOT / "data" / "synthetic" / "train_final.jsonl"

def compile_dataset(
    num_templated: int = 100,
    num_teacher: int = 10,
    run_augmentation: bool = True,
    client: TeacherClient | None = None
) -> list[dict[str, Any]]:
    """Generate and compile notes from multiple sources."""
    raw_notes = []
    
    # 1. Generate templated notes (Strategy 3)
    logger.info("Generating %d template-based notes...", num_templated)
    raw_notes.extend(generate_templated_notes(num_templated))
    
    # 2. Generate teacher LLM notes (Strategy 1)
    if num_teacher > 0:
        if client is None:
            try:
                client = TeacherClient()
            except ValueError as e:
                logger.warning("TeacherClient not initialized (missing API key). Skipping teacher notes generation. %s", e)
                client = None
                
        if client:
            logger.info("Generating %d teacher LLM notes...", num_teacher)
            raw_notes.extend(generate_teacher_notes(client, count=num_teacher))
            
    # 3. Apply augmentation (Strategy 4)
    compiled_notes = []
    for note in raw_notes:
        compiled_notes.append(note)
        if run_augmentation:
            # Add entity swapped version
            compiled_notes.append(swap_entities(note))
            # Add negation flipped version
            compiled_notes.append(flip_negations(note))
            
    # 4. Filter and clean notes (Strategy 5)
    final_notes = []
    for note in compiled_notes:
        clean_note = filter_note(note)
        if clean_note:
            final_notes.append(clean_note)
            
    return final_notes


def print_coverage_report(notes: list[dict[str, Any]]) -> dict[str, int]:
    """Calculate and print entity type and assertion coverage report."""
    stats = {
        "total_notes": len(notes),
        "total_entities": 0,
        "isNegated": 0,
        "isFamily": 0,
        "isHistorical": 0
    }
    
    type_counts: dict[str, int] = {}
    
    for note in notes:
        for ent in note["entities"]:
            stats["total_entities"] += 1
            ent_type = ent["type"]
            type_counts[ent_type] = type_counts.get(ent_type, 0) + 1
            
            assertions = ent.get("assertions", {})
            if assertions.get("isNegated"):
                stats["isNegated"] += 1
            if assertions.get("isFamily"):
                stats["isFamily"] += 1
            if assertions.get("isHistorical"):
                stats["isHistorical"] += 1
                
    print("=== DATASET COVERAGE REPORT ===")
    print(f"Total Notes: {stats['total_notes']}")
    print(f"Total Entities: {stats['total_entities']}")
    print("--- Entity Types ---")
    for t, c in type_counts.items():
        print(f"  {t}: {c}")
    print("--- Assertions ---")
    print(f"  isNegated: {stats['isNegated']}")
    print(f"  isFamily: {stats['isFamily']}")
    print(f"  isHistorical: {stats['isHistorical']}")
    print("===============================")
    
    # Merge dictionaries for return value
    return {**stats, **type_counts}


def build_and_save_train_set(
    out_path: str | Path = DEFAULT_OUT_PATH,
    num_templated: int = 100,
    num_teacher: int = 0
) -> int:
    """Build train set and write to JSON lines format."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    notes = compile_dataset(num_templated=num_templated, num_teacher=num_teacher)
    
    with open(out_path, "w", encoding="utf-8") as fh:
        for note in notes:
            fh.write(json.dumps(note, ensure_ascii=False) + "\n")
            
    print_coverage_report(notes)
    return len(notes)


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
    logging.basicConfig(level=logging.INFO)
    build_and_save_train_set(num_templated=50, num_teacher=0)
