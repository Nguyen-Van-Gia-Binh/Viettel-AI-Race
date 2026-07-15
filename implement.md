# Kế Hoạch Triển Khai — Hybrid NLP Pipeline

> Tài liệu này chốt kiến trúc đã thảo luận và định nghĩa rõ nhiệm vụ của từng thành phần trong pipeline lâm sàng.

---

## Tổng Quan Kiến Trúc

```
input/{id}.txt
      │
      ▼  [Bước 1]
┌─────────────────────────────────────────────────────┐
│  LLM  (Qwen3-8B Q4_K_M — self-hosted Ollama)       │
│  Nhiệm vụ:                                          │
│    - Đọc văn bản lâm sàng tự do                    │
│    - Tạo JSON phân cấp theo ngữ nghĩa đề mục        │
│    - Nhúng thông tin assertion vào field ngữ_cảnh   │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
             structured_file/{id}.json
             (raw hierarchical JSON)
                       │
      ▼  [Bước 2]
┌─────────────────────────────────────────────────────┐
│  Python Flatten  (scripts/flatten_structured.py)    │
│  Nhiệm vụ:                                          │
│    - Duyệt đệ quy toàn bộ JSON                     │
│    - Extract mỗi string leaf → {value, tags}        │
│    - tags = đường dẫn key cha (bỏ dấu, lowercase)  │
│    - KHÔNG suy luận type — để BERT làm              │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
             flattern_file/{id}.json
             [{value, tags}, ...]
                       │
      ▼  [Bước 3]
┌─────────────────────────────────────────────────────┐
│  BERT Classifier  (ViHealthBERT / XLM-RoBERTa)      │
│  Nhiệm vụ:                                          │
│    - Input: value + tags (context)                  │
│    - Output: type (5 nhãn lâm sàng hoặc BỎ_QUA)    │
│    - Lọc bỏ entity nhiễu phi lâm sàng               │
└──────────────────────┬──────────────────────────────┘
                       │
      ▼  [Bước 4 — Đã xử lý ở Bước 1]
┌─────────────────────────────────────────────────────┐
│  Assertion Extraction từ LLM JSON  (Python đọc)     │
│  Nhiệm vụ:                                          │
│    - LLM đã gán assertions vào field "assertions"   │
│      trong structured JSON (cải thiện system_prompt)│
│    - Python chỉ đọc và map sang schema chuẩn        │
│      {"isHistorical", "isNegated", "isFamily"}      │
└──────────────────────┬──────────────────────────────┘
                       │
      ▼  [Bước 5]
┌─────────────────────────────────────────────────────┐
│  Position Finder  (clinlp/position.py)              │
│  Nhiệm vụ:                                          │
│    - Tìm vị trí ký tự [start, end] của value       │
│      trong source text gốc (fuzzy string search)    │
└──────────────────────┬──────────────────────────────┘
                       │
      ▼  [Bước 6]
┌─────────────────────────────────────────────────────┐
│  Entity Linking  (clinlp/linking/link.py)           │
│  Nhiệm vụ: (chỉ cho CHẨN_ĐOÁN và THUỐC)           │
│    - BM25 sparse retrieval (rank_bm25)              │
│    - Dense retrieval (multilingual-e5-small)        │
│    - Reciprocal Rank Fusion → Top-5 candidates     │
│    - Fallback: rapidfuzz string match               │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
             output/{id}.json
             (format nộp cuộc thi)
```

---

## Phân Công Nhiệm Vụ Từng Thành Phần

| Field trong output | Ai xử lý | Cách thực hiện |
|---|---|---|
| `text` | **BERT** | Lấy từ `value` trong flattern_file sau khi BERT xác nhận là entity hợp lệ |
| `type` | **BERT** | Fine-tuned classifier: 5 nhãn + BỎ_QUA (để lọc nhiễu) |
| `assertions` | **LLM** | Gán trực tiếp vào field `"assertions"` trong structured JSON qua system_prompt |
| `position` | **Python** | Fuzzy string search `value` trong source text |
| `candidates` | **BM25 + E5 + RRF** | Retrieval pipeline đã có sẵn trong `link.py` |

---

## Chi Tiết Từng Bước

### Bước 1 — LLM Semantic Parsing

**File**: `src/clinlp/ner/llm_extract.py` → `llm_ner_fn()`  
**Config**: `configs/system_prompt.md`  
**Output**: `structured_file/{id}.json`

**Yêu cầu với LLM:**
- Nhóm thông tin theo đề mục ngữ nghĩa (tiền sử, triệu chứng, chẩn đoán, ...)
- Đặt thông tin assertion vào field `"ngữ_cảnh"` với giá trị rõ ràng:
  - `"tiền sử"` → isHistorical
  - `"phủ định"` → isNegated
  - `"người nhà"` / `"gia đình"` → isFamily
- Dùng key tên field mô tả rõ: `"tên_thuốc"`, `"tên_triệu_chứng"`, `"ghi_chú"`, ...
- **KHÔNG** sửa lỗi chính tả hay chuẩn hóa từ ngữ trong trường `"tên"` — giữ nguyên chữ gốc

---

### Bước 2 — Python Flatten

**File**: `scripts/flatten_structured.py`  
**Input**: `structured_file/{id}.json`  
**Output**: `flattern_file/{id}.json`

**Logic**:
```python
traverse(item, path):
    if str → {value: item, tags: [normalize(k) for k in path]}
    if dict → traverse(v, path + [k]) for each k,v
    if list → traverse(el, path) for each el
```

**Normalize key**: bỏ dấu + lowercase + gạch_dưới → khoảng trắng  
**Không** có bất kỳ type-inference hay assertion-inference nào ở bước này.

---

### Bước 3 — BERT Classifier

**Model đề xuất**: `demdecuong/ViHealthBERT-base-uncased` (135M) hoặc `xlm-roberta-base` (270M)  
**File cần tạo**: `src/clinlp/ner/bert_classifier.py`

**Input format** (mỗi sample):
```python
text_input = "[value] " + " | ".join(tags)
# Ví dụ: "metoprolol tien su benh | thuoc truoc khi nhap vien | ten thuoc"
```

**Output — 1 đầu ra (Type Classification only)**:

| Task | Labels | Ghi chú |
|---|---|---|
| Type | TRIỆU_CHỨNG, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM, CHẨN_ĐOÁN, THUỐC, BỎ_QUA | 6 class |

> ℹ️ Assertions đã được LLM xử lý ở Bước 1 — BERT **không** cần làm isNegated.

**Tạo training data**:
1. Dùng 100 file `input/*.txt` làm seed
2. LLM (Qwen3-8B) sinh synthetic examples với augmentation
3. Script có sẵn: `src/clinlp/dataprep/synth_generate.py`
4. Filter: `src/clinlp/dataprep/quality_filter.py`

---

### Bước 4 — Assertion do LLM gán (không cần code thêm)

**Không tạo file mới** — assertion được đọc thẳng từ `structured_file/{id}.json`.

**Thay vào đó: cải thiện `configs/system_prompt.md`** để LLM output field `"assertions"` chuẩn:

```json
{
  "triệu_chứng": [
    {
      "tên": "ho đờm xanh",
      "assertions": []
    },
    {
      "tên": "không buồn nôn",
      "assertions": ["isNegated"]
    }
  ],
  "tiền_sử_bệnh": {
    "thuốc_trước_khi_nhập_viện": [
      {
        "tên": "metoprolol",
        "assertions": ["isHistorical"]
      }
    ]
  },
  "tiền_sử_gia_đình": {
    "bệnh_lý": [
      {
        "tên": "tăng huyết áp",
        "assertions": ["isFamily"]
      }
    ]
  }
}
```

**Rule trong system_prompt cần thêm:**
- Mỗi object có `"tên"` PHẢI có field `"assertions"` (list, có thể rỗng `[]`)
- Giá trị hợp lệ: `"isNegated"`, `"isHistorical"`, `"isFamily"`
- `isHistorical`: khái niệm thuộc tiền sử, thuốc đang dùng trước nhập viện
- `isNegated`: khái niệm bị phủ định trong câu (không, chưa, phủ nhận, âm tính)
- `isFamily`: khái niệm liên quan đến tình trạng người nhà/gia đình

**Python trong `llm_extract.py`** chỉ cần đọc field `"assertions"` và map:
```python
raw_assertions = item.get("assertions", [])
final = [a for a in raw_assertions if a in {"isNegated", "isHistorical", "isFamily"}]
```

---

### Bước 5 — Position Finder

**File**: `src/clinlp/position.py` (đã có)  
**Logic**: Fuzzy string search tìm `value` trong `source_text` gốc.

> **Chú ý quan trọng**: BERT phải giữ nguyên chuỗi text gốc (không normalize), nếu không position sẽ không tìm được.

---

### Bước 6 — Entity Linking (Candidates)

**File**: `src/clinlp/linking/link.py` (đã có)  
**Chỉ áp dụng cho**: `CHẨN_ĐOÁN` (→ ICD-10) và `THUỐC` (→ RxNorm)

Stack hiện tại:
1. `retrieval.py`: BM25 (rank_bm25) + Dense (multilingual-e5-small) → Top-50
2. Reciprocal Rank Fusion → Top-5
3. Fallback: `rapidfuzz` nếu retrieval libraries không khả dụng

**Cải thiện tiềm năng** (nếu còn thời gian): Thêm `BAAI/bge-reranker-base` để rerank Top-15 → Top-5.

---

## Thứ Tự Ưu Tiên Triển Khai

```
[Đã có]   ✅ Bước 1 — LLM extraction + structured_file cache
[Đã có]   ✅ Bước 2 — Python flatten → flattern_file
[Đã có]   ✅ Bước 5 — Position finder
[Đã có]   ✅ Bước 6 — BM25 + E5 + RRF linking

[Cần làm] 🔴 Bước 1 (cải thiện) — Thêm field "assertions" vào system_prompt
              - Cập nhật configs/system_prompt.md
              - Cập nhật llm_extract.py để đọc field "assertions" trực tiếp
              - Tái sinh structured_file/ với prompt mới

[Cần làm] 🔴 Bước 3 — BERT Classifier (type only)
              - Tạo training data từ flattern_file + LLM augmentation
              - Fine-tune ViHealthBERT (chỉ 1 task: type classification)
              - Tích hợp vào pipeline thay thế heuristic hiện tại

[Tùy chọn] 🔵 Reranker (BAAI/bge-reranker) để tăng accuracy candidates
```

---

## Các File Liên Quan

```
Viettel-AI-Race/
├── configs/
│   ├── system_prompt.md          # System prompt cho LLM
│   └── model.json                # Config model LLM
├── scripts/
│   ├── flatten_structured.py     # Bước 2 — flatten
│   └── run_baseline.py           # Chạy toàn bộ pipeline
├── src/clinlp/
│   ├── llm/client.py             # LLM client (httpx + Ollama)
│   ├── ner/
│   │   ├── llm_extract.py        # Bước 1 — LLM extraction + assertion reading
│   │   ├── bert_classifier.py    # [CẦN TẠO] Bước 3 — BERT type classifier
│   │   ├── pipeline_adapter.py   # Adapter nối NER → pipeline
│   │   └── dictionaries.py       # Seed dictionaries
│   ├── linking/
│   │   ├── link.py               # Bước 6 — entry point linking
│   │   └── retrieval.py          # BM25 + E5 + RRF
│   ├── position.py               # Bước 5 — position finder
│   ├── pipeline.py               # Orchestrator toàn pipeline
│   └── schema.py                 # ConceptType, Assertion enum
├── structured_file/              # Cache raw LLM JSON
└── flattern_file/                # Flat {value, tags} cho BERT
```

---

## Các Ràng Buộc Cuộc Thi

| Ràng buộc | Giá trị |
|---|---|
| Model size tối đa | 9B params (self-hosted) |
| Không dùng API ngoài | Tất cả inference phải offline |
| Output format | `output/{1..100}.json` — list of concept dicts |
| Entity types | TRIỆU_CHỨNG, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM, CHẨN_ĐOÁN, THUỐC |
| Assertions | isNegated, isHistorical, isFamily (chỉ cho TRIỆU_CHỨNG/CHẨN_ĐOÁN/THUỐC) |
| Candidates | ICD-10 (CHẨN_ĐOÁN), RxNorm (THUỐC) |
