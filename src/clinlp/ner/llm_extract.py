# -*- coding: utf-8 -*-
"""LLM-based clinical semantic parsing and flattening.

Uses the configured local LLM to output a hierarchical semantic parsing JSON structure,
which is then flattened in Python into competition-compatible concepts and assertions.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

from clinlp.llm.client import LLMClient
from clinlp.schema import ASSERTION_ELIGIBLE_TYPES, Assertion, ConceptType
from clinlp.ner.pipeline_adapter import RawMention
from clinlp.ner.dictionaries import DIAGNOSIS_TERMS, DRUG_TERMS, LAB_TEST_TERMS, SYMPTOM_TERMS

logger = logging.getLogger(__name__)

# Pre-build dictionary sets for fast O(1) fallback typing
DIAGNOSIS_SET = {t.lower() for t in DIAGNOSIS_TERMS}
DRUG_SET = {t.lower() for t in DRUG_TERMS}
LAB_TEST_SET = {t.lower() for t in LAB_TEST_TERMS}
SYMPTOM_SET = {t.lower() for t in SYMPTOM_TERMS}

PROMPT_PATH = Path(__file__).resolve().parents[3] / "configs" / "system_prompt.md"
if PROMPT_PATH.exists():
    SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")
else:
    SYSTEM_PROMPT = """Bạn là một chuyên gia Biomedical NLP, Clinical Information Extraction và Clinical Semantic Parsing.
Nhiệm vụ của bạn là đọc một văn bản bệnh án tiếng Việt và chuyển đổi thông tin thành một JSON duy nhất có cấu trúc ngữ nghĩa."""


def detect_assertions_from_str(s: str) -> set[Assertion]:
    s_lower = s.lower()
    found = set()
    if "tiền sử" in s_lower or "tien su" in s_lower or "historical" in s_lower:
        found.add(Assertion.HISTORICAL)
    if "phủ định" in s_lower or "phu dinh" in s_lower or "negated" in s_lower or "không" in s_lower or "negative" in s_lower:
        found.add(Assertion.NEGATED)
    if "người nhà" in s_lower or "nguoi nha" in s_lower or "family" in s_lower:
        found.add(Assertion.FAMILY)
    return found


def detect_assertions_from_key(key: str) -> set[Assertion]:
    key_lower = key.lower()
    found = set()
    if "tiền_sử" in key_lower or "tiền sử" in key_lower or "history" in key_lower:
        found.add(Assertion.HISTORICAL)
    if "phủ_định" in key_lower or "phủ định" in key_lower or "negat" in key_lower:
        found.add(Assertion.NEGATED)
    if "người_nhà" in key_lower or "người nhà" in key_lower or "family" in key_lower:
        found.add(Assertion.FAMILY)
    return found


def get_parent_key_type(key: str) -> ConceptType | None:
    k = key.lower()
    if any(x in k for x in ["thuốc", "thuoc", "drug", "medication", "điều_trị", "dùng"]):
        return ConceptType.THUOC
    if any(x in k for x in ["triệu_chứng", "trieu_chung", "symptom", "lâm_sàng", "triệu chứng"]):
        return ConceptType.TRIEU_CHUNG
    if any(x in k for x in ["chẩn_đoán", "chan_doan", "diagnosis", "bệnh", "benh", "tiền_sử", "tiền sử", "tien_su", "tien su"]):
        return ConceptType.CHAN_DOAN
    if any(x in k for x in ["xét_nghiệm", "xet_nghiem", "lab", "test", "chẩn_đoán_hình_ảnh", "thủ_thuật", "thu_thuat", "thủ_tục", "thu_tuc", "imaging", "procedure"]):
        return ConceptType.TEN_XET_NGHIEM
    return None


def find_value_by_fuzzy_key(item: dict, target_patterns: list[str]) -> any:
    """Finds a value in a dictionary where the key matches any of target_patterns fuzzy-style."""
    for k, v in item.items():
        k_normalized = k.lower().replace("_", "").strip()
        for pattern in target_patterns:
            pat_normalized = pattern.lower().replace("_", "").replace(" ", "").strip()
            if pat_normalized in k_normalized:
                return v
            # Handle "tên"/"ten" encoding mangling: any 3-char key starting with 't' and ending with 'n'
            if pat_normalized in ("ten", "tên") and len(k_normalized) == 3 and k_normalized.startswith("t") and k_normalized.endswith("n"):
                return v
            # Handle "kết quả"/"ket_qua" encoding mangling: key contains "ket", "qua", "result"
            if pat_normalized in ("ketqua", "kếtquả") and ("ket" in k_normalized or "qua" in k_normalized or "result" in k_normalized or "val" in k_normalized):
                return v
    return None


def flatten_semantic_json(data: dict) -> Iterator[tuple[str, ConceptType, set[Assertion], str | None]]:
    """Recursively traverses the structured semantic JSON from the LLM,
    inferring ConceptType and Assertions. Yields (text, concept_type, assertions, optional_result).
    """
    def _has_field(item: dict, patterns: list[str]) -> bool:
        val = find_value_by_fuzzy_key(item, patterns)
        if val is None:
            return False
        if isinstance(val, str) and not val.strip():
            return False
        return True

    def traverse(item, parent_key=None, inherited_assertions=None):
        if inherited_assertions is None:
            inherited_assertions = set()

        if isinstance(item, dict):
            new_assertions = inherited_assertions.copy()
            
            # Detect any assertions from the dict keys or context values
            for k, v in item.items():
                if isinstance(v, str):
                    k_norm = k.lower().replace("_", "").replace(" ", "").strip()
                    if k_norm in ("ngữcảnh", "ngucanh", "context") or (k_norm.startswith("ng") and k_norm.endswith("h")):
                        new_assertions.update(detect_assertions_from_str(v))
                new_assertions.update(detect_assertions_from_key(k))

            # Check if this object represents a concept (typically has a "tên" key)
            text_val = find_value_by_fuzzy_key(item, ["ten", "tên", "name", "text"])
            if text_val and isinstance(text_val, str):
                text_val = text_val.strip()
                if text_val:
                    # Inferred type resolution
                    inferred_type = None

                    # A. Refine based on unique fields inside the object
                    has_drug_fields = any(_has_field(item, [f]) for f in ["liều lượng", "lieu_luong", "dosage", "đường dùng", "duong_dung", "route", "tần suất", "tan_suat", "frequency"])
                    has_symptom_fields = any(_has_field(item, [f]) for f in ["vị trí", "vi_tri", "location", "mức độ", "muc_do", "severity", "diễn tiến", "dien_tien"])
                    has_test_fields = any(_has_field(item, [f]) for f in ["giá trị tham chiếu", "reference_range"])
                    has_result_field = _has_field(item, ["kết quả", "ket_qua", "result"])

                    if has_drug_fields:
                        inferred_type = ConceptType.THUOC
                    elif has_symptom_fields:
                        inferred_type = ConceptType.TRIEU_CHUNG
                    elif has_test_fields:
                        inferred_type = ConceptType.TEN_XET_NGHIEM
                    elif has_result_field:
                        inferred_type = ConceptType.TEN_XET_NGHIEM

                    # B. Dictionary lookup fallback (trust ground-truth dictionaries first)
                    if not inferred_type:
                        lower_text = text_val.lower()
                        if lower_text in DRUG_SET:
                            inferred_type = ConceptType.THUOC
                        elif lower_text in DIAGNOSIS_SET:
                            inferred_type = ConceptType.CHAN_DOAN
                        elif lower_text in LAB_TEST_SET:
                            inferred_type = ConceptType.TEN_XET_NGHIEM
                        elif lower_text in SYMPTOM_SET:
                            inferred_type = ConceptType.TRIEU_CHUNG

                    # C. Check parent key hint if fields are ambiguous and not in dictionaries
                    if not inferred_type and parent_key:
                        inferred_type = get_parent_key_type(parent_key)

                    # D. Default fallback if still unknown
                    if not inferred_type:
                        inferred_type = ConceptType.CHAN_DOAN

                    # Parse optional test result
                    result_val = find_value_by_fuzzy_key(item, ["kết quả", "ket_qua", "result"])
                    result_str = None
                    if result_val and isinstance(result_val, (str, int, float)):
                        result_str = str(result_val).strip()

                    yield text_val, inferred_type, new_assertions, result_str

            # Recursively traverse nested dicts/lists
            for k, v in item.items():
                # Skip processed attributes
                k_lower = k.lower().replace("_", "").strip()
                if any(x in k_lower for x in ["ten", "name", "text", "ngucanh", "context", "ketqua", "result", "donvi", "unit", "lieuluong", "dosage", "duongdung", "route", "tansuat", "frequency", "vitri", "location", "mucdo", "severity", "dientien", "giatrithamchieu"]):
                    continue
                yield from traverse(v, parent_key=k, inherited_assertions=new_assertions)

        elif isinstance(item, list):
            for element in item:
                yield from traverse(element, parent_key=parent_key, inherited_assertions=inherited_assertions)

    yield from traverse(data)


def _align_casing_with_source(source: str, extracted_text: str) -> str:
    """Helper to realign casing errors from LLM back to source text representation."""
    if not extracted_text:
        return extracted_text
    
    idx = source.lower().find(extracted_text.lower())
    if idx != -1:
        return source[idx : idx + len(extracted_text)]
        
    return extracted_text


def _clean_json_string(s: str) -> str:
    """Strip any markdown blocks or think blocks if present."""
    s = s.strip()
    if "</think>" in s:
        s = s.split("</think>", 1)[1].strip()
    if s.startswith("```json"):
        s = s[7:]
    if s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()



def llm_ner_fn(source_text: str) -> Iterator[RawMention]:
    """LLM-based ner_fn: calls the local LLM, caches raw JSON to structured_file/,
    then flattens in-memory to yield RawMentions for the pipeline.

    Pipeline stages:
        input/*.txt  →  [LLM call]  →  structured_file/{id}.json  (raw semantic JSON)
                     →  [flatten_semantic_json]  →  flattern_file/{id}.json  (done separately)
                     →  [pipeline]  →  output/{id}.json
    """
    import hashlib

    # ----- resolve a stable file id (prefer numeric stem, fallback to md5) -----
    file_id = None
    input_dir = Path("input/input")
    if input_dir.exists():
        normalized_source = source_text.replace("\r\n", "\n").strip()
        for path in sorted(input_dir.glob("*.txt")):
            try:
                if path.read_text(encoding="utf-8").replace("\r\n", "\n").strip() == normalized_source:
                    file_id = path.stem
                    break
            except Exception:
                pass
    if not file_id:
        file_id = hashlib.md5(source_text.encode("utf-8")).hexdigest()

    # ----- structured_file/ = canonical cache for raw LLM JSON output ---------
    structured_dir = Path("structured_file")
    structured_dir.mkdir(exist_ok=True)
    structured_path = structured_dir / f"{file_id}.json"

    # backward-compat: also check old flattern_file/ cache location
    legacy_dir = Path("flattern_file")
    legacy_path = legacy_dir / f"{file_id}.json"

    response_content: str | None = None

    # 1. Try new cache location first
    if structured_path.exists():
        try:
            response_content = structured_path.read_text(encoding="utf-8")
            logger.info("Loaded structured cache for %s", file_id)
        except Exception as e:
            logger.warning("Failed to read structured cache for %s: %s", file_id, e)

    # 2. Backward-compat: migrate old flattern_file cache if found
    if not response_content and legacy_path.exists():
        try:
            response_content = legacy_path.read_text(encoding="utf-8")
            logger.info("Migrating legacy cache %s → structured_file/", file_id)
            structured_path.write_text(response_content, encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to migrate legacy cache for %s: %s", file_id, e)

    # 3. Call LLM if no cache found
    if not response_content:
        client = LLMClient()
        try:
            response_content = client.chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Bệnh án cần trích xuất:\n---\n{source_text}\n---\n/no_think"}
                ],
                response_format={"type": "json_object"}
            )
            try:
                structured_path.write_text(response_content, encoding="utf-8")
                logger.info("Saved structured JSON for %s → structured_file/", file_id)
            except Exception as e:
                logger.warning("Failed to write structured cache for %s: %s", file_id, e)
        except Exception as e:
            logger.error("LLM call failed for %s: %s", file_id, e)
            raise

    # ----- in-memory flatten → yield RawMentions for pipeline -----------------
    try:
        cleaned_json = _clean_json_string(response_content)
        data = json.loads(cleaned_json)

        seen: set = set()
        for text, concept_type, assertions, result in flatten_semantic_json(data):
            text = text.strip()
            if not text:
                continue

            final_assertions: list[str] = []
            if concept_type in ASSERTION_ELIGIBLE_TYPES:
                final_assertions = sorted([a.value for a in assertions])

            key = (text.lower(), concept_type, tuple(final_assertions))
            if key not in seen:
                seen.add(key)
                aligned_text = _align_casing_with_source(source_text, text)
                yield aligned_text, concept_type, [Assertion(a) for a in final_assertions]

            if result:
                result = result.strip()
                if result:
                    r_key = (result.lower(), ConceptType.KET_QUA_XET_NGHIEM, ())
                    if r_key not in seen:
                        seen.add(r_key)
                        aligned_result = _align_casing_with_source(source_text, result)
                        yield aligned_result, ConceptType.KET_QUA_XET_NGHIEM, []

    except Exception as e:
        logger.error("Failed to parse/flatten LLM response for %s: %s", file_id, e)
        raise
