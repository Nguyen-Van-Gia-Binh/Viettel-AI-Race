"""Centralized filesystem I/O — the only place pipeline code should touch text/JSON
files on disk.

Every read/write here is explicit about encoding and newline handling so Vietnamese
text is never silently corrupted (Windows defaults to cp1252) and character offsets
are never silently shifted by universal-newline translation on a CRLF file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def read_text(path: str | os.PathLike) -> str:
    """Read a corpus file exactly as its bytes decode to UTF-8 text.

    newline="" disables Python's universal-newline translation, so a CRLF file
    is read without \\r\\n being collapsed to \\n mid-read. The 100 provided
    files are all LF-only today, but the private test set comes from a
    different export pipeline — this is cheap insurance against a silent
    offset shift if it contains CRLF.
    """
    with open(path, encoding="utf-8", newline="") as fh:
        return fh.read()


def write_json(path: str | os.PathLike, obj: Any) -> None:
    """Write JSON as UTF-8 with readable (non-\\u-escaped) Vietnamese text."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def read_json(path: str | os.PathLike) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_jsonl(path: str | os.PathLike, records: list[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")


def read_jsonl(path: str | os.PathLike) -> list[Any]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
