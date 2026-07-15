"""Week-1 gate: zip `output/{1..100}.json` into `output.zip` at the repo root,
matching `Topic.md`'s submission format exactly (flat archive, `1.json` ...
`100.json`, no subdirectory).

Refuses to build the zip if any of the 100 files is missing or not a JSON
array, rather than silently shipping a partial/malformed submission.

Usage:
    PYTHONIOENCODING=utf-8 python scripts/make_submission.py
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "output"
SUBMISSION_ZIP = REPO_ROOT / "output.zip"
N_FILES = 100

sys.path.insert(0, str(REPO_ROOT / "src"))

from clinlp.io_utils import read_json  # noqa: E402


def main() -> int:
    missing = []
    malformed = []
    for n in range(1, N_FILES + 1):
        path = OUTPUT_DIR / f"{n}.json"
        if not path.exists():
            missing.append(n)
            continue
        try:
            data = read_json(path)
        except Exception as e:  # noqa: BLE001 - report, don't crash mid-check
            malformed.append((n, str(e)))
            continue
        if not isinstance(data, list):
            malformed.append((n, f"expected a JSON array, got {type(data).__name__}"))

    if missing:
        print(f"ERROR: {len(missing)} output file(s) missing: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if malformed:
        print(f"ERROR: {len(malformed)} output file(s) malformed:")
        for n, err in malformed[:10]:
            print(f"    {n}.json: {err}")
    if missing or malformed:
        print("Refusing to build output.zip. Run scripts/run_baseline.py first.")
        return 1

    with zipfile.ZipFile(SUBMISSION_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for n in range(1, N_FILES + 1):
            zf.write(OUTPUT_DIR / f"{n}.json", arcname=f"{n}.json")

    print(f"wrote {SUBMISSION_ZIP} with {N_FILES} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
