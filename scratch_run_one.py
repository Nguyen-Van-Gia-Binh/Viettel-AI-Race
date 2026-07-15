# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, "src")

from clinlp.io_utils import read_text
from clinlp.ner.llm_extract import llm_ner_fn

source_text = read_text("input/input/1.txt")
try:
    print("Querying LLM for 1.txt...")
    results = list(llm_ner_fn(source_text))
    print("Success! Number of results:", len(results))
    for idx, (text, c_type, assertions) in enumerate(results, 1):
        print(f"{idx}. Text: {text!r} | Type: {c_type.name} | Assertions: {[a.value for a in assertions]}")
except Exception as e:
    print("FAILED with exception:", e)
