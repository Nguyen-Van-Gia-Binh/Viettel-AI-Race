"""Cue and term dictionaries for the rule-based NER + assertion baseline
(CLAUDE.md Workstream B, Week 1 — dictionary/regex only, no LLM yet).

`NEGATION_CUES` is seeded **verbatim** from `Sota_research_report.md` §4.2 —
do not add cues beyond this list here; if a real corpus example needs a cue
that isn't in it, that's a signal to revisit the source research doc, not to
quietly extend this dict.
"""
from __future__ import annotations

NEGATION_CUES: dict[str, list[str]] = {
    "pre_negation": [
        "không", "không có", "chưa", "chưa từng", "không thấy",
        "không ghi nhận", "phủ nhận", "loại trừ", "không bị",
        "chưa phát hiện", "không xuất hiện", "âm tính",
    ],
    "post_negation": [
        "(-)", "âm tính", "không", "bình thường", "trong giới hạn",
    ],
    "family_cues": [
        "bố", "mẹ", "cha", "anh", "chị", "em", "ông", "bà",
        "người nhà", "gia đình", "bố bệnh nhân", "mẹ bệnh nhân",
        "người thân",
    ],
    "historical_cues": [
        "tiền sử", "trước đây", "trước nhập viện", "trước khi nhập viện",
        "thuốc đang dùng trước", "bệnh nền", "bệnh kèm", "đã từng",
        "năm trước", "tháng trước", "từ trước",
    ],
    "section_historical": [
        "TIỀN SỬ", "Tiền sử bệnh", "Tiền sử gia đình",
        "Thuốc trước khi nhập viện", "Bệnh nền", "Bệnh kèm",
    ],
}


# --- NER seed dictionaries -------------------------------------------------
# Terms below are drawn from real occurrences in input/input/*.txt (hand
# surveyed files: 1, 2, 3, 15, 22, 31, 37, 41, 55, 56, 84, 90 — spanning
# cardiology, GI, neuro, OB, and ED notes) — not invented wholesale, per
# CLAUDE.md instructions. This is a Week-1 seed list, expected to grow as
# more of the corpus is surveyed; extend rather than replace.

SYMPTOM_TERMS: list[str] = [
    "đánh trống ngực", "khó thở", "buồn nôn", "nôn", "nôn ra máu",
    "đổ mồ hôi", "chóng mặt", "đau ngực", "đau bụng", "sốt", "ho",
    "ho khan", "tiêu chảy", "mệt mỏi", "ngất xỉu", "phù mắt cá chân",
    "khó chịu vùng ngực", "giọng khàn", "đau chân", "quầng sáng",
    "ăn uống kém", "đau quanh vết mổ", "thai máy tốt", "ra huyết âm đạo",
    "vỡ ối", "rỉ ối", "cơn co tử cung", "chảy nước mũi",
]

DIAGNOSIS_TERMS: list[str] = [
    "xơ gan do rượu", "hội chứng não gan", "hội chứng ruột kích thích",
    "loét tá tràng", "viêm thực quản", "loét thực quản",
    "viêm dạ dày ruột do virus", "xuất huyết nội sọ", "tăng huyết áp",
    "bệnh tim mạch do xơ vữa động mạch", "phình động mạch chủ", "rung nhĩ",
    "nhồi máu cơ tim", "suy tim", "bệnh thận mạn tính", "đái tháo đường",
    "tăng kali máu", "khối máu tụ dưới màng cứng", "tổn thương dây thanh quản",
    "thiếu máu cơ tim cục bộ",
]

DRUG_TERMS: list[str] = [
    "metoprolol", "doxycycline", "atenolol", "aspirin", "omeprazole",
    "torsemide", "insulin glargine", "isosorbide", "rosuvastatin",
    "carvedilol", "lasix", "insulin", "dextrose",
]

LAB_TEST_TERMS: list[str] = [
    "ecg", "điện tâm đồ", "x-quang ngực", "phân tích nước tiểu",
    "công thức sinh hóa máu", "công thức máu", "chức năng gan", "troponin",
    "ct sọ não", "siêu âm tim", "holter", "kali", "creatinine", "ure",
    "hemoglobin", "hba1c", "bnp", "glucose", "spo2", "nghiệm pháp gắng sức",
    "test hơi thở h. pylori", "nội soi",
]
