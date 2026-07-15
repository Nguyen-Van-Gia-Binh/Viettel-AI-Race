#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Flatten structured_file/{id}.json → flattern_file/{id}.json

Format đầu ra (tiện cho BERT classifier):
  [
    {
      "value": "metoprolol",
      "tags": ["tien su benh", "thuoc truoc khi nhap vien", "ten thuoc"]
    },
    ...
  ]

Mỗi entry = một string leaf trong JSON, tags = đường dẫn key cha đã chuẩn hóa.
Không có bất kỳ heuristic type-inference nào — BERT sẽ xử lý phần đó.

Chạy:
    python scripts/flatten_structured.py              # tất cả file
    python scripts/flatten_structured.py 1 5 10       # chỉ file 1, 5, 10
    python scripts/flatten_structured.py --force      # ghi đè
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import unicodedata
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT      = Path(__file__).resolve().parents[1]
STRUCTURED_DIR = REPO_ROOT / "structured_file"
FLATTERN_DIR   = REPO_ROOT / "flattern_file"


# ── Chuẩn hóa key: bỏ dấu, lowercase, gạch_dưới → khoảng trắng ────────────

def _normalize_key(key: str) -> str:
    """tiền_sử_bệnh → tien su benh"""
    # Bỏ dấu tiếng Việt
    normalized = unicodedata.normalize("NFD", key)
    no_accent = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    # Gạch dưới → space, lowercase, trim
    return no_accent.replace("_", " ").lower().strip()


# ── Logic flatten đơn giản (chỉ extract, không suy luận type) ──────────────

def flatten_to_tagged_list(data: dict | list) -> list[dict]:
    """Duyệt đệ quy toàn bộ JSON, trả về list[{value, tags}].

    Mỗi string leaf → một entry với tags là đường dẫn key cha đã chuẩn hóa.
    """
    results: list[dict] = []

    def traverse(item, path: list[str]) -> None:
        if isinstance(item, str):
            value = item.strip()
            if value:
                results.append({
                    "value": value,
                    "tags": [_normalize_key(k) for k in path],
                })

        elif isinstance(item, (int, float)):
            value = str(item).strip()
            if value:
                results.append({
                    "value": value,
                    "tags": [_normalize_key(k) for k in path],
                })

        elif isinstance(item, dict):
            for k, v in item.items():
                traverse(v, path + [k])

        elif isinstance(item, list):
            for element in item:
                # List elements không thêm index vào path — giữ path nguyên
                traverse(element, path)

    traverse(data, [])
    return results


# ── Xử lý từng file ─────────────────────────────────────────────────────────

def _clean_bom(s: str) -> str:
    return s.lstrip("\ufeff").strip()


def flatten_one(structured_path: Path, force: bool = False) -> dict:
    stem = structured_path.stem
    out_path = FLATTERN_DIR / f"{stem}.json"

    if out_path.exists() and not force:
        logger.info("Skip %s (already exists, use --force to overwrite)", stem)
        return {"file": stem, "skipped": True, "entries": 0}

    try:
        raw = _clean_bom(structured_path.read_text(encoding="utf-8-sig"))
        data = json.loads(raw)
    except Exception as e:
        logger.error("Failed to load %s: %s", structured_path.name, e)
        return {"file": stem, "skipped": False, "error": str(e), "entries": 0}

    entries = flatten_to_tagged_list(data)

    out_path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("OK  %s -> flattern_file/%s.json  (%d entries)", structured_path.name, stem, len(entries))
    return {"file": stem, "skipped": False, "entries": len(entries)}


# ── Entrypoint ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Flatten structured_file/ -> flattern_file/")
    parser.add_argument("ids", nargs="*", help="File IDs (e.g. 1 2 5). Empty = all files.")
    parser.add_argument("--force", "-f", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    FLATTERN_DIR.mkdir(exist_ok=True)

    if args.ids:
        paths = []
        for fid in args.ids:
            p = STRUCTURED_DIR / f"{fid}.json"
            if not p.exists():
                logger.warning("Not found: structured_file/%s.json", fid)
            else:
                paths.append(p)
    else:
        paths = sorted(
            STRUCTURED_DIR.glob("*.json"),
            key=lambda p: (not p.stem.isdigit(), int(p.stem) if p.stem.isdigit() else p.stem),
        )

    if not paths:
        logger.error("No files found in structured_file/")
        return 1

    total_entries = 0
    total_skipped = 0
    total_errors  = 0

    for p in paths:
        res = flatten_one(p, force=args.force)
        total_entries += res.get("entries", 0)
        if res.get("skipped"):
            total_skipped += 1
        if res.get("error"):
            total_errors += 1

    print(
        f"\nResult: {len(paths)} file(s) | "
        f"{len(paths) - total_skipped - total_errors} processed | "
        f"{total_skipped} skipped | "
        f"{total_errors} errors | "
        f"{total_entries} total entries"
    )
    return 1 if total_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
