"""Week-1 gate: run the rule-based baseline (Workstream B's ner_fn +
Workstream A's link as link_fn) over every input/input/{1..100}.txt
through pipeline.run_pipeline(), and write output/{n}.json.
"""
from __future__ import annotations

import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

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


def process_file(n: int) -> tuple[int, int, int, int, list, list]:
    try:
        source_text = read_text(INPUT_DIR / f"{n}.txt")
        result = run_pipeline(source_text, ner_fn=ner_fn, link_fn=link_fn)
        write_json(OUTPUT_DIR / f"{n}.json", [c.to_json_dict() for c in result.concepts])
        errors = [i for i in result.issues if i.severity == Severity.ERROR]
        warnings = [i for i in result.issues if i.severity == Severity.WARNING]
        return n, len(result.concepts), len(errors), len(warnings), errors, warnings
    except Exception as e:
        print(f"Error processing {n}.txt: {e}", file=sys.stderr)
        return n, 0, 1, 0, [str(e)], []


def main() -> int:
    total_errors = 0
    total_warnings = 0
    total_concepts = 0

    print("Starting processing 100 files in parallel...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(process_file, range(1, N_FILES + 1)))

    results.sort(key=lambda x: x[0])

    for n, num_concepts, num_errors, num_warnings, errors, warnings in results:
        total_concepts += num_concepts
        total_errors += num_errors
        total_warnings += num_warnings
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
