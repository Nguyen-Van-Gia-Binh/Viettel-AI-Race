"""Workstream A: build `data/ICD-10/icd10_processed.csv` from the CMS-provided
fixed-width reference files.

Two things this module owns:

1. **Code reformatting.** `icd10cm_order_2024.txt`/`icd10cm_codes_2024.txt` store
   codes without the decimal point (e.g. "K219"), but `Topic.md`'s worked
   example and every downstream consumer (`linking/link.py`, the submission
   schema) use the dotted style ("K21.9"). `reformat_icd10_code()` is the one
   place that conversion happens.

2. **KB CSV export.** `build_icd10_kb()` parses the fixed-width order file
   (preferred — it carries both a long and a short description per code,
   unlike `icd10cm_codes_2024.txt` which only has one description column) and
   writes `code,long_description,short_description,description,vi_aliases`.

Column layout of `icd10cm_order_2024.txt`, confirmed by direct inspection
(there is no machine-readable header row in the file itself; column offsets
below were derived by counting characters against known rows and cross-checked
against `data/ICD-10/icd10OrderFiles.pdf`):

    [0:5]   seq_num (5 digits)
    [5]     space
    [6:14]  code, left-justified, space-padded to 8 chars (max real code is 7
            chars, e.g. "S52209K" / "C441021")
    [14]    HIPAA-valid-for-submission flag: '1' = billable leaf code,
            '0' = category/header code, not billable. NOT used to filter rows
            here — Topic.md's own worked example expects "K21.0" as a
            candidate, and K21.0 is itself a '0'-flag header code, so
            filtering to billable-only would silently drop a code the
            organizers themselves use.
    [15]    space
    [16:76] "long description", space-padded to 60 chars — despite the name,
            this is often the *abbreviated* one (CMS truncates/abbreviates
            words to fit the fixed width, e.g. "dis" for "disease").
    [76]    space
    [77:]   "short description", unbounded width to end of line — despite the
            name, this is usually the *fuller*, untruncated wording. Used as
            the primary `description` column for that reason.

Bilingual (EN + Vietnamese) descriptions are `Plan.md`'s stated follow-up
(`linking/translate.py`, not yet built — needs either an MT pass or a curated
glossary at real coverage). The CMS source files are English-only. The small
`VI_ALIASES` table below is a stopgap, not a substitute for that: it only
covers a handful of common diagnoses (the `Topic.md` GERD worked example plus
a few terms that recur across `input/input/*.txt`), each cross-checked by hand
against the order file so the code is verifiably correct. It exists purely so
`linking/link.py`'s baseline fuzzy matcher has *something* Vietnamese to match
against for CHẨN_ĐOÁN entities — real coverage requires the real translation
module.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ORDER_FILE = REPO_ROOT / "data" / "ICD-10" / "icd10cm_order_2024.txt"
DEFAULT_CODES_FILE = REPO_ROOT / "data" / "ICD-10" / "icd10cm_codes_2024.txt"
DEFAULT_OUT_CSV = REPO_ROOT / "data" / "ICD-10" / "icd10_processed.csv"

_CODE_START, _CODE_END = 6, 14
_FLAG_IDX = 14
_LONG_START, _LONG_END = 16, 76
_SHORT_START = 77

# Seed Vietnamese alias table: dotted ICD-10 code -> Vietnamese phrases that
# should fuzzy-match to it. Every entry was manually cross-checked against
# `icd10cm_order_2024.txt`'s English description (see docstring above).
# NOT comprehensive — see module docstring.
VI_ALIASES: dict[str, list[str]] = {
    # Topic.md's own worked example: "bệnh trào ngược dạ dày - thực quản" is
    # ambiguous about esophagitis status, so the organizers list BOTH codes.
    "K21.0": [
        "trào ngược dạ dày - thực quản",
        "trào ngược dạ dày thực quản",
        "bệnh trào ngược dạ dày - thực quản",
        "bệnh trào ngược dạ dày thực quản",
        "GERD",
    ],
    "K21.9": [
        "trào ngược dạ dày - thực quản",
        "trào ngược dạ dày thực quản",
        "bệnh trào ngược dạ dày - thực quản",
        "bệnh trào ngược dạ dày thực quản",
        "GERD",
    ],
    "I10": ["tăng huyết áp", "cao huyết áp"],
    "E11.9": ["đái tháo đường", "đái tháo đường type 2", "đái tháo đường tuýp 2"],
    "I21.9": ["nhồi máu cơ tim"],
    "I60.9": ["xuất huyết dưới nhện"],
    "J18.9": ["viêm phổi"],
    "K81.9": ["viêm túi mật"],
    "K80.20": ["sỏi mật"],
    "L03.90": ["viêm mô tế bào"],
}


def reformat_icd10_code(raw_code: str) -> str:
    """"K219" -> "K21.9"; "A00" (only 3 chars, a category header with nothing
    to put after a dot) -> "A00" unchanged.

    Rule: insert '.' right after the 3rd character, only if a 4th+ character
    exists. This matches every example in `Topic.md` (K21.0, K21.9) and every
    real ICD-10-CM code (max 3 chars before the dot, up to 4 after, e.g.
    "S52.209K").
    """
    code = raw_code.strip()
    if len(code) <= 3:
        return code
    return f"{code[:3]}.{code[3:]}"


def parse_order_file(path: str | Path = DEFAULT_ORDER_FILE) -> list[dict]:
    """Parse the fixed-width order file into one dict per code. See module
    docstring for the column layout this relies on."""
    rows: list[dict] = []
    with open(path, encoding="utf-8", newline="") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            raw_code = line[_CODE_START:_CODE_END].strip()
            if not raw_code:
                continue
            flag = line[_FLAG_IDX:_FLAG_IDX + 1]
            long_desc = line[_LONG_START:_LONG_END].strip()
            short_desc = line[_SHORT_START:].strip()
            rows.append(
                {
                    "code": reformat_icd10_code(raw_code),
                    "raw_code": raw_code,
                    "billable": flag == "1",
                    "long_description": long_desc,
                    # short_description is usually the fuller/untruncated text;
                    # fall back to long_description on the rare row where it's
                    # empty rather than emit a blank description.
                    "short_description": short_desc or long_desc,
                }
            )
    return rows


def parse_codes_file(path: str | Path = DEFAULT_CODES_FILE) -> dict[str, str]:
    """Parse the simpler 2-column `icd10cm_codes_2024.txt` (code + single
    description, whitespace-delimited) as a supplementary code->description
    map, used only to fill in a description for a code that's somehow missing
    one from the order file."""
    out: dict[str, str] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            raw_code, desc = parts
            out[reformat_icd10_code(raw_code)] = desc.strip()
    return out


def build_icd10_kb(
    order_path: str | Path = DEFAULT_ORDER_FILE,
    codes_path: Optional[str | Path] = DEFAULT_CODES_FILE,
    out_path: str | Path = DEFAULT_OUT_CSV,
    vi_aliases: dict[str, list[str]] = VI_ALIASES,
) -> int:
    """Parse the CMS order file (+ optional codes file as a description
    fallback), attach the seed Vietnamese alias table, and write
    `icd10_processed.csv`. Returns the row count written."""
    rows = parse_order_file(order_path)
    fallback_desc = parse_codes_file(codes_path) if codes_path else {}

    out_rows = []
    for row in rows:
        description = row["short_description"] or fallback_desc.get(row["code"], "")
        aliases = ";".join(vi_aliases.get(row["code"], []))
        out_rows.append(
            {
                "code": row["code"],
                "billable": "1" if row["billable"] else "0",
                "description": description,
                "long_description": row["long_description"],
                "short_description": row["short_description"],
                "vi_aliases": aliases,
            }
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["code", "billable", "description", "long_description", "short_description", "vi_aliases"],
        )
        writer.writeheader()
        writer.writerows(out_rows)
    return len(out_rows)


def load_icd10_kb(path: str | Path = DEFAULT_OUT_CSV) -> list[dict]:
    """Load the already-built `icd10_processed.csv`. Returns `[]` (not an
    error) if the CSV hasn't been built yet, so callers like `linking/link.py`
    can degrade gracefully rather than crash at import time."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


if __name__ == "__main__":
    n = build_icd10_kb()
    print(f"wrote {n} ICD-10 rows to {DEFAULT_OUT_CSV}")
