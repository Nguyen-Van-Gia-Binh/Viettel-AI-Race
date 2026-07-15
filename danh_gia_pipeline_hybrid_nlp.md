# Đánh giá Pipeline Hybrid NLP cho bài toán trích xuất thực thể lâm sàng tiếng Việt

## 1. Đánh giá tổng quan

Pipeline hiện tại được đánh giá ở mức:

\[
\boxed{7/10\text{ về mặt thiết kế hệ thống}}
\]

và khoảng:

\[
\boxed{5.5-6/10\text{ về độ an toàn khi chạy trên tập test hiện tại}}
\]

Pipeline có tư duy kiến trúc tốt khi chia bài toán thành các bước độc lập gồm extraction, classification, assertion, position và entity linking, thay vì yêu cầu một mô hình duy nhất xử lý toàn bộ. Tuy nhiên, điểm yếu lớn nhất là Qwen đang đứng ở đầu pipeline. Nếu bước đầu trích thiếu, cắt sai hoặc chuẩn hóa lại entity thì các bước sau gần như không thể khôi phục lại.

---

## 2. Những điểm làm tốt

### 2.1. Phân rã bài toán hợp lý

Pipeline hiện tại tách thành:

1. LLM semantic parsing.
2. Flatten JSON.
3. BERT phân loại type.
4. Assertion extraction.
5. Position finding.
6. Entity linking.

Đây là cách tiếp cận hợp lý với một bài toán lâm sàng phức tạp vì các trường đầu ra như `type`, `assertions`, `position` và `candidates` có bản chất khác nhau, không nên bị ép vào cùng một mô hình.

Việc chỉ áp dụng entity linking cho `CHẨN_ĐOÁN` và `THUỐC` cũng hợp lý, tránh thực hiện linking không cần thiết cho triệu chứng hoặc kết quả xét nghiệm.

### 2.2. Giữ nguyên chuỗi gốc để tìm position

Pipeline đã xác định đúng rằng `value` không được chuẩn hóa hoặc sửa chính tả trước khi tìm vị trí. Đây là quyết định quan trọng vì tập test có nhiều lỗi dính từ và chuỗi bất thường như:

```text
atenololtrong ngày
cảm giáckhó chịu
doxycyclinebactrim
```

Nếu mô hình sửa thành “atenolol trong ngày”, position finder sẽ không còn tìm được span gốc.

### 2.3. Kết hợp sparse và dense retrieval

Stack:

\[
\text{BM25}+\text{multilingual-e5-small}+\text{RRF}
\]

là một baseline entity linking tốt.

BM25 xử lý tốt khớp từ chính xác, E5 hỗ trợ cách diễn đạt gần nghĩa, còn RRF giúp giảm phụ thuộc vào một phương pháp retrieval duy nhất.

Ví dụ:

```text
tăng huyết áp ↔ hypertension
nhồi máu cơ tim ↔ myocardial infarction
metoprolol ↔ RxNorm concept
```

### 2.4. Phù hợp với yêu cầu triển khai offline

Qwen3-8B chạy local, BERT classifier, BM25 và E5 đều phù hợp với yêu cầu inference offline và giới hạn mô hình dưới 9B tham số.

---

## 3. Điểm yếu lớn nhất: LLM là single point of failure

Pipeline hiện tại:

```text
Raw document
    ↓
Qwen extraction
    ↓
Flatten
    ↓
BERT classification
```

BERT chỉ nhìn thấy những chuỗi mà Qwen đã trích ra. Do đó:

\[
\text{Recall toàn pipeline}
\leq
\text{Recall của Qwen extraction}
\]

Nếu Qwen bỏ sót một entity, BERT không có cách nào tìm lại.

Ví dụ với câu:

```text
monitor holter cho thấy nhịp xoang chiếm ưu thế.
Ghi nhận ngoại tâm thu nhĩ và ngoại tâm thu thất xuất hiện thường xuyên.
```

Qwen có thể:

- Gộp toàn câu thành một entity.
- Chỉ trích `monitor holter`.
- Bỏ mất `ngoại tâm thu nhĩ`.
- Bỏ mất `ngoại tâm thu thất`.
- Sửa cách viết của chuỗi gốc.

Khi đó, BERT chỉ có thể phân loại candidate đã có, không thể sửa lỗi span hoặc bổ sung entity bị bỏ sót.

### Khuyến nghị

Cần bổ sung một nhánh candidate generation độc lập:

```text
Raw document
 ├── Qwen candidate extraction
 ├── Dictionary / regex extraction
 └── Optional XLM-R token NER
          ↓
     Candidate union
```

Tập candidate cuối cùng:

\[
C_{\text{final}}
=
C_{\text{LLM}}
\cup
C_{\text{dictionary}}
\cup
C_{\text{regex}}
\cup
C_{\text{NER}}
\]

Cách này giúp tăng recall và giảm phụ thuộc hoàn toàn vào Qwen.

---

## 4. Flatten toàn bộ string leaf là quá rộng

Script flatten hiện tại lấy mọi string leaf trong JSON và chuyển thành:

```json
{
  "value": "...",
  "tags": ["..."]
}
```

Cách này giúp hạn chế bỏ sót thông tin nhưng sẽ sinh ra rất nhiều candidate nhiễu như:

```text
hôm nay
vài tuần
bên phải
nghiêm trọng
không đặc hiệu
ngày nhập viện
được chuyển đến cấp cứu
cảm thấy tốt hơn
```

BERT phải học một lớp `BỎ_QUA` cực kỳ đa dạng. Trong điều kiện không có train thật và chủ yếu dùng synthetic data, đây là rủi ro lớn.

### Đề xuất phân loại field

#### Loại A — Candidate chắc chắn

Các key như:

```text
tên thuốc
triệu chứng
chẩn đoán
tên xét nghiệm
kết quả xét nghiệm
```

Đưa trực tiếp sang classifier.

#### Loại B — Candidate có điều kiện

Các key như:

```text
phát hiện
ghi chú
diễn biến
kết quả hình ảnh
```

Chỉ lấy noun phrase hoặc span y khoa phù hợp.

#### Loại C — Metadata

Các key như:

```text
thời gian
mức độ
vị trí
tần suất
yếu tố làm nặng
yếu tố làm giảm
```

Không đưa trực tiếp thành entity. Chỉ sử dụng làm context.

Ví dụ nên lưu:

```json
{
  "value": "đau ngực",
  "context_attributes": {
    "vị_trí": "trái",
    "thời_gian": "20 giây",
    "mức_độ": "nhẹ"
  }
}
```

---

## 5. Input cho BERT hiện tại thiếu ngữ cảnh

Input hiện tại:

```python
text_input = "[value] " + " | ".join(tags)
```

Cấu trúc này chưa đủ để phân biệt các trường hợp khó.

Ví dụ:

```text
value = "bình thường"
tags = ["kết quả"]
```

Model không thể biết đây là:

- Kết quả ECG bình thường.
- Xét nghiệm máu bình thường.
- Buồng trứng bình thường.
- Tình trạng bệnh nhân bình thường.
- Một chuỗi không nên trích.

### Input đề xuất

```text
[VALUE] bình thường
[SECTION] Đánh giá tại bệnh viện > Kết quả xét nghiệm
[CONTEXT] Bảng công thức máu bình thường.
```

Hoặc đánh dấu trực tiếp span:

```text
[SECTION] Kết quả xét nghiệm
Bảng công thức máu <e>bình thường</e>.
```

Input tốt nhất nên gồm:

\[
\text{value}
+
\text{parent key}
+
\text{full section path}
+
\text{source sentence}
+
\text{previous sentence}
\]

### Model đề xuất

Với đặc trưng test pha Việt–Anh, nhiều abbreviation và lỗi dịch:

\[
\boxed{\texttt{xlm-roberta-large}}
\]

nên là model chính.

Nếu tài nguyên hạn chế:

\[
\boxed{\texttt{xlm-roberta-base}}
\]

Không cần CRF vì Qwen đã tạo candidate span trước đó. Nên sử dụng:

```python
AutoModelForSequenceClassification
```

---

## 6. Nên đổi classifier thành hai head

Một softmax sáu lớp gồm năm loại thực thể và `BỎ_QUA` chưa phải cấu trúc tối ưu, vì `BỎ_QUA` có bản chất khác hẳn năm nhãn lâm sàng.

### Head 1 — Candidate validation

\[
\text{KEEP}
\quad \text{vs.} \quad
\text{DROP}
\]

### Head 2 — Type classification

\[
\{
\text{TRIỆU\_CHỨNG},
\text{TÊN\_XÉT\_NGHIỆM},
\text{KẾT\_QUẢ\_XÉT\_NGHIỆM},
\text{CHẨN\_ĐOÁN},
\text{THUỐC}
\}
\]

Loss tổng:

\[
\mathcal{L}
=
\lambda_1\mathcal{L}_{keep}
+
\lambda_2\mathcal{L}_{type}
\]

Trong đó, type loss chỉ tính cho sample có nhãn `KEEP`.

Cách này giúp tách rõ hai nhiệm vụ:

1. Chuỗi này có phải entity hợp lệ không?
2. Nếu có, nó thuộc loại nào?

---

## 7. Việc giao toàn bộ assertion cho Qwen là rủi ro

Pipeline hiện để Qwen gán:

- `isNegated`
- `isHistorical`
- `isFamily`

sau đó Python chỉ lọc các giá trị hợp lệ.

Cách này đơn giản nhưng chưa đủ chắc chắn.

### 7.1. `isHistorical` có thể xác định phần lớn bằng section

Nếu entity nằm trong:

```text
Tiền sử bệnh
Thuốc trước khi nhập viện
Các bệnh lý mạn tính
```

thì gần như chắc chắn là historical.

Không cần để LLM tự suy luận hoàn toàn.

### 7.2. `isNegated` cần scope detection

Các từ khóa phủ định thường gặp:

```text
không
chưa
phủ nhận
âm tính
không ghi nhận
không thấy
không phát hiện
```

Cần xác định phạm vi phủ định đến dấu phẩy, liên từ hoặc cuối câu.

### 7.3. “Âm tính” không phải lúc nào cũng chỉ là assertion

Ví dụ:

```text
soi tươi ký sinh trùng âm tính
```

Ở đây, `âm tính` có thể là:

- `KẾT_QUẢ_XÉT_NGHIỆM`
- đồng thời thể hiện xét nghiệm không phát hiện tác nhân.

Nếu chỉ map từ khóa “âm tính” thành `isNegated`, pipeline có thể làm mất nhãn kết quả xét nghiệm.

### Đề xuất assertion hybrid

```text
LLM assertion
+
Section rule
+
Negation scope rule
+
Conflict resolver
```

Ví dụ:

```python
if section.startswith("tiền sử"):
    add("isHistorical")

if negation_scope_contains(entity):
    add("isNegated")

if section == "tiền sử gia đình":
    add("isFamily")
```

---

## 8. Position finder bằng fuzzy search có thể sai

Nhiều file có cùng một entity xuất hiện nhiều lần trong tài liệu.

Ví dụ `đánh trống ngực` có thể xuất hiện ở:

- Lý do nhập viện.
- Triệu chứng hiện tại.
- Đặc điểm triệu chứng.
- Tình trạng ngay trước nhập viện.

Nếu position finder chỉ tìm chuỗi gần giống đầu tiên, nhiều entity có thể bị map về cùng một vị trí.

### Đề xuất

Qwen nên trả thêm:

```json
{
  "value": "đánh trống ngực",
  "evidence_sentence": "Tình trạng ngay trước khi nhập viện: Tiếp tục cảm thấy đánh trống ngực",
  "occurrence_index": 4
}
```

Position finder thực hiện:

1. Tìm `evidence_sentence`.
2. Tìm `value` bên trong evidence.
3. Chuyển offset cục bộ thành offset toàn tài liệu.
4. Chỉ dùng fuzzy search toàn document làm fallback.

---

## 9. BERT học hoàn toàn từ synthetic data là rủi ro cao

Khi không có train thật, classifier sẽ học từ:

- Dataset ngoài được remap.
- Synthetic data do Qwen sinh.
- Rule-generated data.

Model có thể học phong cách dữ liệu sinh thay vì phong cách test.

Test có nhiều lỗi như:

```text
bình thườngbình thường
atenololtrong
doxycyclinebactrim
lower abdominal pain
VS98.3 12987 56 18 99RA
```

Synthetic corpus phải chủ động mô phỏng:

- Code-mixing Việt–Anh.
- Dính từ.
- Thiếu dấu.
- Lặp từ.
- Dấu sinh tồn bị dính.
- Abbreviation y khoa.
- Câu dịch máy không tự nhiên.
- Liều và đường dùng thuốc kiểu `25mg po bid`.
- Xét nghiệm ghi ngắn như `AST 319`, `EF30`, `WBC 11.6`.

Nếu không có augmentation này, BERT có thể làm giảm chất lượng so với Qwen.

---

## 10. Dataset đề xuất

### Dataset chính

\[
\boxed{\text{i2b2/VA 2010}}
\]

Lý do:

- Clinical notes.
- Discharge summary.
- Nhãn problem, test, treatment.
- Có assertion.
- Phong cách gần với tập test hơn các dataset y khoa tiếng Việt hiện có.

Mapping:

| i2b2 | Schema đích |
|---|---|
| Problem | TRIỆU_CHỨNG hoặc CHẨN_ĐOÁN |
| Test | TÊN_XÉT_NGHIỆM |
| Test result/context | KẾT_QUẢ_XÉT_NGHIỆM |
| Treatment | THUỐC hoặc BỎ_QUA |
| Absent | isNegated |
| Historical | isHistorical |
| Associated with someone else | isFamily |

### Corpus continual pre-training

\[
\boxed{\text{MIMIC-IV-Note hoặc corpus clinical notes tương đương}}
\]

Dùng để học:

- Văn phong bệnh án.
- Hospital course.
- Medical shorthand.
- Test result.
- Imaging report.
- Medication notation.
- Section structure.

### Dataset thuốc

\[
\boxed{\text{n2c2 2018}}
\]

Phù hợp với:

- Drug.
- Strength.
- Dose.
- Route.
- Frequency.
- Duration.
- Reason.
- Adverse event.

Ví dụ:

```text
metoprolol → THUỐC
25mg → BỎ_QUA
po → BỎ_QUA
bid → BỎ_QUA
```

### Dataset tiếng Việt phụ

\[
\boxed{\text{VietMed-NER}}
\]

Dùng để bổ sung:

- Disease/symptom.
- Drug/chemical.
- Diagnostics.
- Organ.
- Unit.
- Treatment.
- Surgery.

Không nên train chỉ bằng VietMed-NER vì dữ liệu này là hội thoại y tế, không giống hoàn toàn clinical notes.

### Dataset không nên dùng làm nguồn chính

- ViMQ: phong cách câu hỏi bệnh nhân.
- PhoNER-COVID19: phong cách tin tức dịch tễ.
- ViHealthBERT corpus: hữu ích cho representation nhưng không thay thế được clinical-note training data.

---

## 11. Entity linking cần confidence gate

Entity linking chỉ áp dụng cho `CHẨN_ĐOÁN` và `THUỐC` là đúng. Tuy nhiên, nếu classifier gán nhầm:

```text
đau ngực → CHẨN_ĐOÁN
```

pipeline sẽ cố map triệu chứng sang ICD-10.

Nên thêm confidence gate:

```python
if type == "CHẨN_ĐOÁN" and confidence >= 0.80:
    link_icd10()
elif type == "THUỐC" and confidence >= 0.85:
    link_rxnorm()
else:
    candidates = []
```

Có thể bổ sung dictionary validation:

```text
THUỐC:
- Khớp RxNorm/ATC dictionary.
- Hoặc khớp pattern tên hoạt chất.

CHẨN_ĐOÁN:
- Khớp ICD dictionary.
- Hoặc nằm trong section chẩn đoán.
```

---

## 12. Kiến trúc đề xuất cuối cùng

```text
Raw clinical note
        │
        ├──────────────────────────────────┐
        │                                  │
        ▼                                  ▼
Qwen semantic extraction          Rule/dictionary extraction
        │                                  │
        └──────── Candidate union ─────────┘
                         │
                         ▼
       Candidate context reconstruction
   value + section + sentence + surrounding text
                         │
                         ▼
              XLM-R type classifier
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        KEEP/DROP head        5-class type head
              │                     │
              └──────────┬──────────┘
                         ▼
               Assertion ensemble
        LLM + section rules + negation scope
                         │
                         ▼
           Exact-context position finder
                fuzzy search fallback
                         │
                         ▼
              Entity linking gate
       BM25 + E5 + RRF + optional reranker
                         │
                         ▼
                    Final JSON
```

---

## 13. Thứ tự ưu tiên sửa pipeline

### Mức bắt buộc

1. Thêm câu gốc và section path vào input classifier.
2. Không flatten mọi string thành candidate.
3. Thêm rule/dictionary candidate generator.
4. Position theo evidence sentence.
5. Assertion bằng hybrid rule + LLM.
6. Dùng XLM-R thay cho model đơn ngữ làm classifier chính.

### Mức nên làm

7. Đổi classifier thành hai head: `KEEP/DROP` và `5-class type`.
8. Thêm confidence và disagreement handling.
9. Sinh hard negative `BỎ_QUA` đa dạng.
10. Mô phỏng lỗi dịch, lỗi dính từ và code-mixing trong synthetic data.
11. Chỉ thêm reranker sau khi type classification đã ổn định.

---

## 14. Đánh giá theo hạng mục

| Hạng mục | Điểm |
|---|---:|
| Kiến trúc tổng thể | 8/10 |
| Khả năng chạy offline | 9/10 |
| Entity linking | 8/10 |
| Candidate extraction | 6/10 |
| Context preservation | 5/10 |
| Assertion handling | 6/10 |
| Position accuracy | 6/10 |
| Khả năng thích nghi test nhiễu | 5/10 |

Sau khi bổ sung candidate union, source context, hybrid assertion và evidence-based position, pipeline có thể đạt mức thiết kế khoảng:

\[
\boxed{8.5/10}
\]

---

## 15. Kết luận

Pipeline hiện tại đúng hướng và có tính triển khai cao, đặc biệt ở cách phân tách nhiệm vụ và thiết kế entity linking. Tuy nhiên, phiên bản hiện tại phụ thuộc quá mạnh vào Qwen extraction và làm mất nhiều context tại bước flatten.

Điểm nghẽn lớn nhất không phải chỉ là lựa chọn BERT nào, mà là:

\[
\boxed{
\text{candidate recall}
+
\text{context preservation}
+
\text{synthetic-to-test distribution gap}
}
\]

Lựa chọn đề xuất cuối cùng:

\[
\boxed{\text{Model chính: XLM-R-large}}
\]

\[
\boxed{\text{Model phụ: ViHealthBERT}}
\]

\[
\boxed{\text{Dataset chính: i2b2/VA 2010 được dịch và remap}}
\]

\[
\boxed{\text{Continual pre-training: MIMIC-IV-Note hoặc clinical notes tương đương}}
\]

\[
\boxed{\text{Dataset thuốc: n2c2 2018}}
\]

\[
\boxed{\text{Dataset tiếng Việt phụ: VietMed-NER}}
\]
