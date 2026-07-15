"""Workstream A: build `data/ICD-10/icd10_processed.csv` from the standard
Vietnamese ICD-10 Excel dataset (`danh-muc-ma-benh-tat-excel.xlsx`).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXCEL_FILE = REPO_ROOT / "data" / "ICD-10" / "danh-muc-ma-benh-tat-excel.xlsx"
DEFAULT_OUT_CSV = REPO_ROOT / "data" / "ICD-10" / "icd10_processed.csv"

# Seed Vietnamese alias table: dotted ICD-10 code -> Vietnamese phrases that
# should fuzzy-match to it.
VI_ALIASES: dict[str, list[str]] = {
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
    exists. Removes existing dot first to handle pre-formatted inputs robustly.
    """
    code = str(raw_code).replace(".", "").strip()
    if len(code) <= 3:
        return code
    return f"{code[:3]}.{code[3:]}"


def build_icd10_kb(
    excel_path: str | Path = DEFAULT_EXCEL_FILE,
    out_path: str | Path = DEFAULT_OUT_CSV,
    vi_aliases: dict[str, list[str]] = VI_ALIASES,
) -> int:
    """Parse the standard Vietnamese ICD-10 Excel file, apply code formatting,
    attach the seed/manual Vietnamese alias table, and write `icd10_processed.csv`.
    Returns the row count written.
    """
    import pandas as pd

    excel_path = Path(excel_path)
    df = pd.read_excel(excel_path, sheet_name=0, header=2)
    # Skip numbering row (first row of parsed df is index 0 containing numbers 1 to 29)
    df = df.iloc[1:]

    out_rows = []
    for _, row in df.iterrows():
        raw_code = str(row.get("MÃ BỆNH", "")).strip()
        if not raw_code or pd.isna(row.get("MÃ BỆNH")):
            continue

        formatted_code = reformat_icd10_code(raw_code)
        
        # Check if non-billable (i.e. column 25 is not empty/null)
        non_billable_val = row.get("MÃ KHÔNG ĐƯỢC SỬ DỤNG VÌ CÓ MÃ 4 HOẶC 5 KÝ TỰ CỤ THỂ HƠN")
        billable = "0" if (pd.notna(non_billable_val) and str(non_billable_val).strip() != "") else "1"

        # Descriptions
        vi_name = str(row.get("TÊN BỆNH", "")).strip()
        en_name = str(row.get("DISEASE NAME WHO 2019 (ENGLISH)", "")).strip()

        # Build vi_aliases list
        aliases_list = []
        
        # Add code without dot (MÃ BỆNH KHÔNG DẤU)
        raw_code_no_dot = str(row.get("MÃ BỆNH KHÔNG DẤU", "")).strip()
        if raw_code_no_dot and raw_code_no_dot != formatted_code:
            aliases_list.append(raw_code_no_dot)
            
        # Add manual aliases
        manual = vi_aliases.get(formatted_code, [])
        for m in manual:
            if m not in aliases_list:
                aliases_list.append(m)
                
        vi_aliases_str = ";".join(aliases_list)

        out_rows.append(
            {
                "code": formatted_code,
                "billable": billable,
                "description": vi_name,
                "long_description": en_name,
                "short_description": vi_name,
                "vi_aliases": vi_aliases_str,
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

