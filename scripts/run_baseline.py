"""Week-1 gate: run the rule-based baseline (Workstream B's `ner_fn` +
Workstream A's `link` as `link_fn`) over every `input/input/{1..100}.txt`
through `pipeline.run_pipeline()`, and write `output/{n}.json`.

This is the integration point `CLAUDE.md` reserves for whoever assembles the
two workstreams — it does not redesign `pipeline.py`, `ner/pipeline_adapter.py`,
or `linking/link.py`, only wires the three together.

Usage:
    PYTHONIOENCODING=utf-8 python scripts/run_baseline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = REPO_ROOT / "input" / "input"
OUTPUT_DIR = REPO_ROOT / "output"
N_FILES = 100

sys.path.insert(0, str(REPO_ROOT / "src"))

from clinlp.io_utils import read_text, write_json  # noqa: E402
from clinlp.linking.link import link as link_fn  # noqa: E402
from clinlp.ner.pipeline_adapter import ner_fn  # noqa: E402
from clinlp.pipeline import run_pipeline  # noqa: E402
from clinlp.validate import Severity  # noqa: E402


def main() -> int:
    total_errors = 0
    total_warnings = 0
    total_concepts = 0

    for n in range(1, N_FILES + 1):
        source_text = read_text(INPUT_DIR / f"{n}.txt")
        result = run_pipeline(source_text, ner_fn=ner_fn, link_fn=link_fn)

        write_json(OUTPUT_DIR / f"{n}.json", [c.to_json_dict() for c in result.concepts])
        total_concepts += len(result.concepts)

        errors = [i for i in result.issues if i.severity == Severity.ERROR]
        warnings = [i for i in result.issues if i.severity == Severity.WARNING]
        total_errors += len(errors)
        total_warnings += len(warnings)
        if errors:
            print(f"{n}.txt: {len(errors)} ERROR issue(s)")
            for issue in errors:
                print(f"    {issue}")

    print(
        f"\nwrote {N_FILES} files to {OUTPUT_DIR}, "
        f"{total_concepts} concepts total, "
        f"{total_errors} ERROR issue(s), {total_warnings} WARNING issue(s)"
    )
    return 1 if total_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
