# Plan: Chiến lược thực thi cho cuộc thi Clinical NLP Viettel

## Context

Repo hiện đang ở giai đoạn khung sườn (scaffolding), chưa có pipeline hoạt động: `output/` rỗng, không có code NER/assertion/entity-linking, chỉ có `src/survey.py` (khảo sát corpus) và ICD-10 CMS reference đã tải (chưa xử lý). `CLAUDE.md` đã chốt hướng kiến trúc (rule-based baseline trước, sau đó nâng cấp bằng LLM ≤9B + retrieval) nhưng chưa có kế hoạch thực thi cụ thể để 3 người triển khai song song trong 3-4 tuần.

Bài toán: đọc văn bản y khoa tiếng Việt tự do → xuất `output/N.json` gồm span, `type` (5 nhãn), `position` (offset ký tự chính xác trên file gốc), `assertions` (isNegated/isFamily/isHistorical), `candidates` (mã ICD-10/RxNorm). Điểm số: `0.3*text_score + 0.3*assertions_score + 0.4*candidates_score`, trong đó sai `type` bị phạt kép (zero cả 3 metric), và `candidates_score` (entity linking) chiếm trọng số cao nhất. Ràng buộc thi: mô hình lúc infer phải self-host ≤9B tham số, không gọi API ngoài; nhưng **API ngoài (GPT-4/Claude/Gemini) được phép dùng để chuẩn bị dữ liệu offline** — đây là lợi thế thật sự vì team đã có budget cho việc này.

Thông tin đã xác nhận với người dùng: **3-4+ tuần**, **GPU ≥16GB VRAM** (local/cloud riêng), **có budget API ngoài**, **team 3 người**. Mục tiêu của plan này là chuyển hướng kiến trúc đã được nghiên cứu kỹ (trong `Topic_Analysis.md` và `Sota_research_report.md`) thành một roadmap cụ thể: có submission hợp lệ sớm làm lưới an toàn, module hóa để 3 người làm song song không giẫm chân nhau, và **có khả năng đo lường/cải tiến liên tục** (không phải chỉ chạy một lần cho ra kết quả).

## Quyết định kiến trúc chính

1. **Tạo package thật `src/clinlp/`** thay vì để script rời rạc — ranh giới module map đúng theo ranh giới ownership của 3 người, giảm tối đa xung đột merge.
2. **Xây "hạ tầng chung" (schema, position calculator, validator, scorer) trước tiên, trong tuần 1**, do 1 người phụ trách — mọi module khác (rule baseline, LLM pipeline, eval) đều phụ thuộc vào đây, và đây chính là điều kiện để pipeline "dễ cải tiến từng phần" mà không phá vỡ phần còn lại.
3. **Interface rõ ràng giữa các bước** (`ner.infer()` → `position.py` → `linking.link()` → `validate.py`) để có thể thay thế/nâng cấp độc lập từng thành phần (đổi embedding model, thử LLM khác, cải tiến rule assertion) mà không đụng các phần khác.
4. **RxNorm: xử lý song song 2 hướng** — đăng ký UTS account (đầy đủ nhất) + REST API bulk endpoint `allconcepts` theo term-type (không auth, đã từng test trong repo) làm phương án dự phòng — không chờ hướng nào cũng không block.
5. **Eval harness tái tạo đúng công thức chấm điểm**, chạy trên 2 gold set tự xây (synthetic có nhãn + spot-check thủ công 10-15/100 file thật) — đây là mảnh hiện đang thiếu hoàn toàn trong mọi tài liệu và là rủi ro "bay mù" lớn nhất nếu không có.
6. **Reproducibility là việc làm liên tục từ Day 0**, không phải dồn vào cuối — vì top ~15 đội phải nộp source+weights+data+README để BTC build lại và chấm trên private test set.

## Cấu trúc file/module dưới `src/`

```
pyproject.toml / requirements.txt     # NEW — dependency pin, pip install -e .
.gitignore                            # NEW — venv, __pycache__, embedding index lớn, model weights
README.md                             # NEW — viết dần từ Day 1, chính là deliverable "top 15 nộp source"
Makefile (hoặc run.ps1/run.sh)        # NEW — make baseline / make eval / make submission / make data

src/clinlp/
  __init__.py
  io_utils.py        # [C] read_text()/write_json() luôn encoding="utf-8", newline="" — fix 1 lần lỗi
                      #     cp1252/CRLF thay vì rủi ro rải rác nhiều nơi
  schema.py           # [C] Concept/Type/Assertion — CONTRACT chung, viết Day 1 trước khi làm song song
  preprocessing.py    # [C] Unicode NFC normalize — CHỈ dùng cho bản làm việc khi prompt LLM;
                      #     position.py luôn match ngược lại file gốc CHƯA normalize
  position.py         # [C] Bước 3 — str.find()/regex chính xác, xử lý duplicate span (nth-occurrence),
                      #     log rõ khi gặp near-miss (vd thiếu dấu cách trong 1.txt/31.txt), never silent-drop
  validate.py         # [C] Bước 6 — JSON schema, position bounds, type/assertion consistency, duplicate check
  sections.py         # [B] Bước 1 — parser section header (mở rộng từ regex heuristic của survey.py),
                      #     trả về span + propagation map cho isHistorical/isFamily

  ner/
    dictionaries.py   # [B] Từ điển triệu chứng/chẩn đoán/thuốc + cue phủ định/gia đình/tiền sử
                       #     (seed từ Sota_research_report.md §4.2 + abbr_counter của survey.py)
    rules.py          # [B] Rule-based NER — baseline tuần 1
    assertions.py     # [B] 3-layer: Layer1 NegEx rule, Layer2 section-propagation, Layer3 hook LLM (tuần 3+)
    prompts.py         # [B] Prompt template: zero-shot extraction, SLENDER format, teacher-gen prompt
    llm_extract.py     # [B] Gọi Qwen2.5-7B LOCAL qua constrained decoding (Outlines/GBNF) — module DUY NHẤT
                       #     giữ handle model lúc infer
    train_qlora.py      # [B] QLoRA fine-tune (peft+trl+bitsandbytes), input từ train_final.jsonl (Workstream C)
    infer.py            # [B] Entrypoint thống nhất: extract(text, mode="rules"|"llm_zeroshot"|"llm_qlora")

  linking/
    icd10_kb.py        # [A] Load CMS files, reformat K219→K21.9, build mô tả song ngữ EN+gloss VN
    rxnorm_kb.py        # [A] Lấp gap RxNorm (xem mục dưới)
    translate.py        # [A] VN entity → EN gloss cho tra ICD-10 (batch offline, KHÔNG gọi lúc infer)
    retrieval.py         # [A] BM25 + dense (SapBERT-XLMR/multilingual-E5) + Reciprocal Rank Fusion
    rerank.py            # [A] Cross-encoder (ms-marco-MiniLM-L-12-v2)
    verify_llm.py        # [A] LLM verify — CHỈ chọn từ candidate list đã retrieve (chống hallucination),
                          #     dùng CHUNG model instance với ner/llm_extract.py qua llm/client.py
    link.py              # [A] Entrypoint thống nhất: link(entity_text, type, context) -> list[codes]
                          #     — input chỉ cần string, nên A có thể test độc lập, KHÔNG cần chờ B xong NER

  dataprep/
    teacher_client.py   # [C] Wrap GPT-4/Claude/Gemini API — tách vật lý khỏi llm/client.py để không thể
                         #     lỡ tay wire API ngoài vào pipeline lúc infer (organizer có thể grep-verify)
    synth_generate.py    # [C] Strategy 1 — teacher sinh note + tự gán nhãn
    translate_adapt.py   # [C] Strategy 2 — dịch/adapt i2b2/n2c2; position LUÔN tính lại qua position.py
    augment_testset.py   # [C] Strategy 4 — paraphrase/entity-swap/negation-flip trên 100 file thật
                         #     (LOẠI TRỪ các file dùng làm gold-set thủ công — tránh leakage)
    templates.py          # [C] Strategy 3 — template/slot-filler (fallback nếu thiếu đa dạng)
    quality_filter.py     # [C] Strategy 5 — check consistency/overlap/position trước khi vào train set
    build_train_set.py    # [C] Gộp tất cả → data/synthetic/train_final.jsonl + báo cáo coverage theo
                          #     type/assertion (để thấy thiếu isFamily/isHistorical sớm, không phải sau khi train)

  eval/
    scorer.py          # [C] Tái tạo ĐÚNG công thức chấm (WER, Jaccard, weighted Jaccard, double-penalty
                       #     khi sai type) — viết + unit test TUẦN 1, mọi thứ sau đó được đo bằng cái này
    goldset.py          # [C] Quản lý 2 gold set (never train trên đây)
    run_eval.py          # [C] CLI: chạy pipeline (mode bất kỳ) trên gold set, in breakdown theo từng metric

  llm/
    client.py          # [C khởi tạo interface, B implement] — MỘT model server local (vllm/llama.cpp)
                       # dùng chung bởi ner/llm_extract.py và linking/verify_llm.py

  pipeline.py           # [C, cập nhật liên tục] Orchestrator: preprocessing → sections → ner.infer →
                        # position → linking.link (chỉ CHẨN_ĐOÁN/THUỐC) → validate → schema objects.
                        # Có flag mode="baseline"|"hybrid" — baseline luôn chạy được kể cả khi hybrid đang xây

scripts/
  run_baseline.py        # [C] input/input/*.txt → output/*.json ở mode "baseline" — DELIVERABLE TUẦN 1
  run_inference.py       # [C] mode "hybrid", có --self-consistency N cho lần nộp cuối
  build_icd10_index.py   # [A] Build KB một lần: reformat + embedding + BM25 index
  build_rxnorm_index.py  # [A] Tương tự, thay thế ý định ban đầu của download_icd10.py
  make_submission.py     # [C] Validate từng output/N.json rồi mới zip — từ chối zip nếu có file lỗi
  smoke_test.sh/.ps1     # [C] Chạy run_baseline trên vài file từ clean checkout — check reproducibility,
                         #     chạy lại mỗi tuần
  survey.py               # giữ nguyên, MOVE từ src/ — nuôi cue-list cho sections.py/dictionaries.py

  download_icd10.py       # Giữ ở src/ root, đánh dấu rõ "deprecated/reference-only" trong README

tests/
  test_position.py         # [C] Test case thật: "cảm giáckhó chịu" (1.txt), "chuyển dạBệnh sử" (31.txt),
                           #     token dày "VS98.3 12987 56 18 99RA" (3.txt), duplicate-entity nth-occurrence
  test_scorer.py            # [C] Test công thức chấm khớp ví dụ tính tay, gồm case double-penalty
  test_schema_validate.py   # [C]
  test_icd10_reformat.py    # [A] K219 → K21.9-style

configs/
  model.yaml     # base model id, QLoRA adapter path, quantization, retrieval top-K
  paths.yaml     # đường dẫn KB/index/data

data/
  ICD-10/icd10_processed.csv   # NEW [A] — mã dotted + mô tả song ngữ
  RxNorm/                       # NEW [A] — xem mục lấp gap dưới
  embeddings/                   # NEW — FAISS/npy index (KHÔNG commit thẳng, xem rủi ro #5)
  synthetic/                    # NEW — jsonl theo từng strategy + train_final.jsonl
  eval/                         # NEW — goldset_synthetic.jsonl, goldset_manual_spotcheck.jsonl
```

## Phân chia 3 người (theo dependency thật, không phải chia đều)

| Người                                       | Phụ trách                                                                                            | Vì sao không bị block                                                                                                                          |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **A — KB & Entity Linking**                 | `linking/*`, ICD-10/RxNorm processed data, `build_*_index.py`                                        | `link()` chỉ nhận string — A tự test bằng ~30 mẫu chẩn đoán/thuốc lấy từ ví dụ trong `Topic.md` mà không cần chờ NER của B xong                |
| **B — NER/Assertion/LLM**                   | `ner/*`, `sections.py`, fine-tuning                                                                  | Tuần 1 làm rule-based (`rules.py`/`assertions.py`) không cần dữ liệu training; QLoRA (phụ thuộc duy nhất vào C) chủ động để tuần 2 mới bắt đầu |
| **C — Hạ tầng/Position/Validate/Data/Eval** | `io_utils.py`, `position.py`, `validate.py`, `pipeline.py`, `dataprep/*`, `eval/*`, scaffolding repo | Không phụ thuộc ai — chính là nền tảng A/B import vào, nên C làm trước để A/B không phải chờ                                                   |

**Điểm tích hợp bắt buộc:** Ngày 2-3 (A/B bắt đầu dùng `schema.py`+`position.py`) → cuối tuần 1 (ghép rule+assertion+basic-linking → submission đầu tiên) → tuần 2 (`train_final.jsonl` → `train_qlora.py`; `scorer.py` chấm mọi mode) → tuần 3 (A/B dùng chung 1 model instance qua `llm/client.py` — cần 1 buổi pair thật sự, không chỉ handoff).

## Lấp gap RxNorm (chưa có gì, cần ~100K+ concept)

Chạy song song từ Day 1:
- **Path 1 (chính):** đăng ký NLM UTS account (free) → tải RxNorm Full Monthly Release (`RXNCONSO.RRF`) → lọc tiếng Anh + term type liên quan (IN/PIN/SCD/SBD/BN/GPCK/BPCK) → `rxnorm_kb.py` xuất `data/RxNorm/rxnorm_processed.csv`.
- **Path 2 (dự phòng, không cần auth, đã smoke-test trong repo):** REST API endpoint `/REST/allconcepts.json?tty=...` — lấy **theo term-type**, không phải theo từng tên thuốc, nên đủ nhanh cho bulk build. Cache raw response vào `data/RxNorm/raw_rest_cache/`.
- Dù path nào xong trước, output cuối là 1 file local — **không bao giờ gọi REST API lúc infer thật**, chỉ dùng lúc build KB offline.

## Eval harness (mảnh hiện đang thiếu hoàn toàn)

Vì 100 file thật không có ground truth, xây 2 gold set do C quản lý, không dùng để train:
1. `goldset_synthetic.jsonl` — teacher LLM sinh (batch riêng, tách khỏi train_final.jsonl) — rẻ, cho tín hiệu nhanh nhưng hơi lạc quan (in-distribution với style của teacher).
2. `goldset_manual_spotcheck.jsonl` — chọn 10-15/100 file thật (trải đều theo độ dài: file ngắn như `31.txt`, file dài có cấu trúc như `3.txt`, file văn xuôi không cấu trúc), cả 3 người tự gán nhãn theo `schema.py`, chéo-kiểm tra một phần để đồng thuận — cũng là cách buộc cả team hiểu rõ ranh giới nhãn trước khi tự động hóa. **Phải loại các file này khỏi nguồn của `augment_testset.py`** để tránh leakage.

`eval/scorer.py` tái tạo đúng công thức (WER cấp từ, Jaccard, weighted Jaccard, double-penalty khi sai type). `eval/run_eval.py` chạy bất kỳ mode nào trên cả 2 gold set → mọi lần cải tiến (thêm reranker, thêm LLM-verify, đổi model...) đều có số đo trước/sau, không đoán mò.

## Roadmap 4 tuần (rút gọn còn 3 tuần thì cắt phần đánh dấu [có thể cắt])

**Tuần 0-1 (Day 0-5) — Baseline an toàn + lấy KB + Eval harness**
- C: `io_utils`, `position`, `validate` (+ unit test theo case lỗi thật), `pipeline` skeleton, `scorer` + unit test, gán nhãn thủ công gold-set.
- B: `sections.py`, `dictionaries.py`, `rules.py`, `assertions.py` (chỉ Layer 1+2, chưa LLM).
- A: `icd10_kb.py` (reformat dotted code), khởi động cả 2 path RxNorm, `link.py` bản fuzzy/exact match đơn giản.
- **Gate cứng (hết Day 3):** có `output.zip` hợp lệ schema, đủ 100 file, position đúng — nếu trễ, cả 3 người dừng làm feature mới để fix baseline trước.
- **Gate cuối tuần 1:** eval harness chạy được, baseline đã có điểm số làm mốc; ICD-10/RxNorm processed CSV tồn tại.

**Tuần 2 (Day 6-12) — Sinh dữ liệu quy mô + LLM pass đầu tiên + retrieval**
- C: chạy đủ 5 strategy sinh dữ liệu → `train_final.jsonl` + báo cáo coverage (chú ý `isFamily` chắc chắn hiếm, cần oversample có chủ đích trong prompt).
- B: serving Qwen2.5-7B local, zero-shot + constrained decoding (chấm ngay bằng harness — đây là lưới an toàn thứ 2 không cần data training), bắt đầu `train_qlora.py` v1 khi data sẵn sàng.
- A: `retrieval.py` (BM25+dense+RRF), `translate.py` (VN→EN cho ICD-10).
- **Quyết định (cuối tuần 2):** nếu QLoRA v1 không rõ ràng thắng zero-shot, KHÔNG đổ thêm thời gian tuần 3 vào train tiếp — quay về zero-shot + constrained decoding (rẻ hơn để lặp) và dồn thời gian cho entity linking (40% điểm số).

**Tuần 3 (Day 13-19) — Ghép hybrid pipeline đầy đủ**
- A: `rerank.py`, `verify_llm.py`.
- B: Layer 3 LLM cho assertion khó; thử SLENDER format **[có thể cắt]**.
- C: ghép `pipeline.py` mode "hybrid", chạy eval harness sau mỗi mảnh ghép để biết chính xác thành phần nào tăng/giảm điểm.
- **Gate cuối tuần 3:** đóng băng kiến trúc — sau mốc này chỉ tối ưu, không mở rộng thêm.

**Tuần 4 (Day 20-28, hoặc 20-21 nếu chỉ có 3 tuần) — Tối ưu, đóng gói, nộp bài**
- Error analysis theo trọng số điểm (candidates 40% > text/assertions 30%), đặc biệt săn case sai `type` (double-penalty).
- Self-consistency voting **[có thể cắt]** — chỉ chạy 1 lần cho bản nộp cuối.
- Đóng gói reproducibility: pin `requirements.txt`, hoàn thiện README, chạy `smoke_test` từ clone sạch trên máy chưa từng dev.
- Chuẩn bị sẵn bundle nộp source cho top-15 (không đợi được thông báo mới làm).

## Rủi ro kỹ thuật riêng của repo này (chưa tài liệu nào nêu)

1. **Encoding nguy hiểm hơn cả in ra console** — `open()`/`json.dump()` không chỉ định `encoding="utf-8"` sẽ âm thầm hỏng tiếng Việt trên Windows. Fix 1 lần trong `io_utils.py`.
2. **CRLF** — 100 file hiện tại toàn LF, nhưng private test set (pipeline export khác) có thể có CRLF, gây lệch offset âm thầm. `read_text()` nên mở với `newline=""`.
3. **Không có dependency manifest** → nguy cơ "chạy được trên máy tôi" đúng lúc quan trọng nhất (BTC rebuild cho top-15). Pin version từ Day 0.
4. **`bitsandbytes`/`vllm`/flash-attention không ổn định trên Windows thuần** — máy dev hiện không có GPU. Nên train/serve trên Linux (WSL2 hoặc cloud GPU), quyết định từ đầu, tránh mất cả buổi debug install ở tuần 2.
5. **Chưa có `.gitignore`, 1 commit bulk-add tất cả** — tránh commit thẳng embedding index (vài trăm MB) hay model weights vào git; commit CSV đã xử lý (nhỏ, diff được) + script build lại index deterministic, document rebuild step trong README.
6. **Không bao giờ gọi REST API (RxNorm/ICD-10/dịch) lúc infer thật** — chỉ `dataprep/teacher_client.py` và `scripts/build_*_index.py` được phép đụng network; `linking/*` và `ner/llm_extract.py` không import `requests`/`urllib` — nên coi đây là invariant có thể review (grep-check), không chỉ là guideline.
7. **Rủi ro "big-bang integration"** — 3 người làm riêng 3-4 tuần dễ dẫn tới lần chạy end-to-end đầu tiên là ở tuần 4. `smoke_test` chạy định kỳ (Day 3, cuối mỗi tuần) để buộc tích hợp sớm.
8. **Git workflow** — dùng nhánh ngắn hạn theo module (`ner/`, `linking/`, `dataprep+eval/`) merge thường xuyên, vì ranh giới module đã được chọn để giảm tối đa xung đột — nên nói rõ với team để tránh quay lại thói quen commit thẳng vào `main`.

## Verification / cách kiểm chứng plan này hoạt động

- **Cuối Day 3:** chạy `scripts/run_baseline.py` rồi `scripts/make_submission.py` → phải ra `output.zip` với đủ 100 file JSON hợp lệ schema (`validate.py` pass), không lỗi position out-of-bounds.
- **Cuối tuần 1:** chạy `eval/run_eval.py --mode baseline` trên cả 2 gold set → có điểm số cụ thể theo cả 3 metric (text/assertions/candidates) làm mốc so sánh.
- **Mỗi lần thêm module mới** (retrieval, reranker, QLoRA, LLM-verify): chạy lại `eval/run_eval.py` so với mốc trước đó — chỉ giữ lại thay đổi nếu điểm tăng hoặc không giảm đáng kể ở metric khác.
- **`tests/test_position.py`**: chạy `pytest tests/` phải pass, đặc biệt các case lấy từ dữ liệu thật (thiếu dấu cách, token dày đặc, duplicate entity).
- **Cuối mỗi tuần:** chạy `scripts/smoke_test.sh`/`.ps1` từ một clone sạch (hoặc ít nhất `pip install -e .` lại từ đầu) để đảm bảo repo luôn ở trạng thái build-lại-được — mô phỏng đúng việc BTC sẽ làm với top-15 đội.
- **Trước khi nộp cuối:** kiểm tra thủ công `linking/*` và `ner/llm_extract.py` không import `requests`/`urllib` (đảm bảo không lỡ gọi API ngoài lúc infer).

## File quan trọng cần tạo trước tiên
- `src/clinlp/schema.py` — contract chung, phải có trước khi 3 người làm song song.
- `src/clinlp/position.py` — tính offset chính xác, chặn đường baseline tuần 1, rủi ro cao nhất về correctness (C7 trong Topic_Analysis.md).
- `src/clinlp/eval/scorer.py` — tái tạo đúng công thức điểm, là thứ duy nhất biến "pipeline chạy được" thành "pipeline đang cải tiến".
- `src/clinlp/linking/link.py` — entrypoint entity linking (40% điểm số), interface độc lập NER nên A bắt đầu được ngay.
- `src/clinlp/pipeline.py` — điểm ghép nối cuối cùng của cả 3 workstream thành 1 submission chạy được.
