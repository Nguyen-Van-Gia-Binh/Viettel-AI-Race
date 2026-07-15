from clinlp.schema import Assertion, ConceptType
from typing import Iterator

DIAGNOSIS_SET = set()
DRUG_SET = set()
LAB_TEST_SET = set()
SYMPTOM_SET = set()

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
    if any(x in k for x in ["chẩn_đoán", "chan_doan", "diagnosis", "bệnh", "benh"]):
        return ConceptType.CHAN_DOAN
    if any(x in k for x in ["xét_nghiệm", "xet_nghiem", "lab", "test", "chẩn_đoán_hình_ảnh", "thủ_thuật", "thu_thuat", "imaging", "procedure"]):
        return ConceptType.TEN_XET_NGHIEM
    return None

def find_value_by_fuzzy_key(item: dict, target_patterns: list[str]) -> any:
    for k, v in item.items():
        k_normalized = k.lower().replace("_", "").strip()
        for pattern in target_patterns:
            pat_normalized = pattern.lower().replace("_", "").replace(" ", "").strip()
            if pat_normalized in k_normalized:
                return v
            if pat_normalized in ("ten", "tên") and len(k_normalized) == 3 and k_normalized.startswith("t") and k_normalized.endswith("n"):
                return v
            if pat_normalized in ("ketqua", "kếtquả") and ("ket" in k_normalized or "qua" in k_normalized or "result" in k_normalized or "val" in k_normalized):
                return v
    return None


def flatten_semantic_json(data: dict) -> Iterator[tuple[str, ConceptType, set[Assertion], str | None]]:
    def traverse(item, parent_key=None, inherited_assertions=None):
        if inherited_assertions is None:
            inherited_assertions = set()

        safe_parent = str(parent_key).encode("ascii", "backslashreplace").decode("ascii")
        print(f"DEBUG: traverse called with type(item)={type(item)}, parent_key={safe_parent}, inherited_assertions={inherited_assertions}")

        if isinstance(item, dict):
            new_assertions = inherited_assertions.copy()
            for k, v in item.items():
                if isinstance(v, str):
                    k_lower = k.lower()
                    if k_lower in ("ngữ_cảnh", "ngu_canh", "context") or (len(k_lower) == 8 and k_lower.startswith("ng") and k_lower.endswith("h")):
                        new_assertions.update(detect_assertions_from_str(v))
                new_assertions.update(detect_assertions_from_key(k))

            text_val = find_value_by_fuzzy_key(item, ["ten", "tên", "name", "text"])
            safe_keys = [k.encode("ascii", "backslashreplace").decode("ascii") for k in item.keys()]
            safe_text = str(text_val).encode("ascii", "backslashreplace").decode("ascii") if text_val else None
            print(f"DEBUG: dict keys={safe_keys}, text_val={safe_text}")
            if text_val and isinstance(text_val, str):
                text_val = text_val.strip()
                if text_val:
                    inferred_type = None
                    has_drug_fields = any(find_value_by_fuzzy_key(item, [f]) is not None for f in ["liều lượng", "lieu_luong", "dosage", "đường dùng", "duong_dung", "route", "tần suất", "tan_suat", "frequency"])
                    has_symptom_fields = any(find_value_by_fuzzy_key(item, [f]) is not None for f in ["vị trí", "vi_tri", "location", "mức độ", "muc_do", "severity", "diễn tiến", "dien_tien"])
                    has_test_fields = any(find_value_by_fuzzy_key(item, [f]) is not None for f in ["giá trị tham chiếu", "reference_range"])
                    has_result_field = find_value_by_fuzzy_key(item, ["kết quả", "ket_qua", "result"]) is not None

                    if has_drug_fields:
                        inferred_type = ConceptType.THUOC
                    elif has_symptom_fields:
                        inferred_type = ConceptType.TRIEU_CHUNG
                    elif has_test_fields:
                        inferred_type = ConceptType.TEN_XET_NGHIEM
                    elif has_result_field:
                        inferred_type = ConceptType.TEN_XET_NGHIEM

                    if not inferred_type and parent_key:
                        inferred_type = get_parent_key_type(parent_key)

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

                    if not inferred_type:
                        inferred_type = ConceptType.CHAN_DOAN

                    result_val = find_value_by_fuzzy_key(item, ["kết quả", "ket_qua", "result"])
                    result_str = None
                    if result_val and isinstance(result_val, (str, int, float)):
                        result_str = str(result_val).strip()

                    print(f"DEBUG: YIELDING {text_val} of type {inferred_type}")
                    yield text_val, inferred_type, new_assertions, result_str

            for k, v in item.items():
                k_lower = k.lower().replace("_", "").strip()
                # Skip processed attributes
                if any(x in k_lower for x in ["ten", "name", "text", "ngucanh", "context", "ketqua", "result", "donvi", "unit", "lieuluong", "dosage", "duongdung", "route", "tansuat", "frequency", "vitri", "location", "mucdo", "severity", "dientien", "giatrithamchieu"]):
                    safe_k = k.encode("ascii", "backslashreplace").decode("ascii")
                    print(f"DEBUG: skipping key={safe_k}")
                    continue
                yield from traverse(v, parent_key=k, inherited_assertions=new_assertions)

        elif isinstance(item, list):
            for element in item:
                yield from traverse(element, parent_key=parent_key, inherited_assertions=inherited_assertions)

    yield from traverse(data)

data = {
    "bệnh_sử": {
        "triệu_chứng": [
            {
                "tên": "ho",
                "ngữ_cảnh": "phủ định"
            }
        ]
    },
    "tiền_sử": {
        "chẩn_đoán": [
            {
                "tên": "tăng huyết áp"
            }
        ]
    },
    "xét_nghiệm": [
        {
            "tên": "Troponin",
            "kết_quả": "0.01",
            "đơn_vị": "ng/mL"
        }
    ]
}

print("Running flatten...")
results = list(flatten_semantic_json(data))
print("Results:", results)
