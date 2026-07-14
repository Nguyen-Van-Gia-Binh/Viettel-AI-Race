# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Entry for a Viettel-hosted clinical NLP competition. The system must read free-form Vietnamese
medical text (doctor notes, discharge summaries, lab results, EHR excerpts) and output a structured
list of medical concepts: span, type, ICD-10/RxNorm codes, and contextual assertions.

Read these in order before making design decisions — they are the actual spec/strategy, not just notes:
1. `Topic.md` — official problem statement and submission rules from the organizers (authoritative).
2. `Topic_Analysis.md` — our breakdown of the scoring formula, challenges, and candidate approaches.
3. `Sota_research_report.md` — research on SOTA techniques mapped to each subtask (NER, entity linking, assertion detection, data synthesis).

## Output contract (from `Topic.md` — do not deviate)

- Input: `input/input/{1..100}.txt`, free-form Vietnamese clinical text. **No training labels are
  provided** — this is a cold-start problem; any training data must be synthesized.
- For each `N.txt`, produce `output/N.json`: a JSON array of concept objects with fields:
  - `text` — exact substring from the source file
  - `type` — one of `TRIỆU_CHỨNG | TÊN_XÉT_NGHIỆM | KẾT_QUẢ_XÉT_NGHIỆM | CHẨN_ĐOÁN | THUỐC`
  - `position` — `[start, end]`, 0-indexed **character** offsets into the raw input file (not bytes, not tokens)
  - `assertions` — subset of `["isNegated", "isFamily", "isHistorical"]`; only applies to `TRIỆU_CHỨNG`, `CHẨN_ĐOÁN`, `THUỐC`
  - `candidates` — ICD-10 codes (for `CHẨN_ĐOÁN`) or RxNorm codes (for `THUỐC`) only; omitted for other types
- Final submission is `output.zip` containing `1.json` … `100.json`.

## Critical constraints (competition rules)

- The model used **at inference time** must be self-hosted, ≤9B parameters, no external API calls
  (no GPT-4/Claude/etc. in the live pipeline).
- External APIs/LLMs **are** allowed for offline prep — generating synthetic training data, building
  the ICD-10/RxNorm reference lookups, etc. — just never called live against the test set.
- Top ~15 teams must submit reproducible source + model weights + data + README so organizers can
  rebuild and re-score on a private test set; keep the pipeline reproducible from a clean checkout.

## Scoring — drives engineering priority

```
final_score = 0.3 * text_score (1 - WER on `text` spans)
            + 0.3 * assertions_score (Jaccard)
            + 0.4 * candidates_score (Jaccard, weighted by len(ground_truth)+1)
```

- `candidates_score` (ICD-10/RxNorm linking) is weighted highest — it's the main lever.
- Getting `type` wrong is a **double penalty**: scored as both a spurious prediction and a missed
  gold entity, zeroing out all three sub-metrics for that item. Prefer precision over recall on type
  classification.
- Position must always be computed by exact Python string matching (`str.find`/regex) against the
  untouched original file text — never let a model emit position numbers directly, and never run text
  through a normalization step that shifts character offsets before matching back against the original.

## Repo layout

- `input/input/*.txt` — the 100 provided test records; the only data given by the organizers.
- `data/ICD-10/` — manually downloaded ICD-10-CM reference files (CMS, public domain, no login):
  - `icd10cm_order_2024.txt` — fixed-width: seq_num, code (no decimal point), a flag digit, long description, short description
  - `icd10cm_codes_2024.txt` — simpler 2-column list: code (no decimal point) + description
  - `*_addenda_2024.txt` — year-over-year changes; `*.pdf` — official format docs
  - Codes in these files omit the decimal point (e.g. `K219`, not `K21.9`) — reformat to match the
    dotted style used in `Topic.md`'s examples before emitting candidates.
  - RxNorm reference data has not been downloaded yet. See `Sota_research_report.md` §3/§7.3: the
    full NLM RxNorm release needs a free UTS account; the RxNorm REST API (`rxnav.nlm.nih.gov`) needs
    no auth and is fine for offline lookups during data prep.
- `src/` — pipeline scripts, built incrementally, no package structure yet:
  - `survey.py` — corpus statistics over `input/input` (length distributions, recurring Vietnamese
    section headers like "Tiền sử bệnh" / "Thuốc trước khi nhập viện", abbreviation frequency).
  - `download_icd10.py` — bulk-downloads ICD-10-CM from the public NLM Clinical Table Search API by
    paginating over letter + 2-digit code prefixes (single requests cap at 500 results). Reference
    only — actual ICD-10 data now comes from the manually downloaded CMS files above, not this script.
- `output/` — empty; will hold generated `{1..100}.json` predictions.

## Commands

- Windows console defaults to cp1252 and cannot print Vietnamese text — any script that prints
  Vietnamese strings to stdout must be run with UTF-8 output forced:
  ```
  PYTHONIOENCODING=utf-8 python src/<script>.py
  ```
- No test suite, linter, or build system is set up yet.

## Architecture direction

Agreed approach (see `Topic_Analysis.md` §5 and `Sota_research_report.md` §6 for full detail):
rule-based baseline first — dictionary/regex NER, Vietnamese NegEx-style assertion rules, section-header
propagation for `isHistorical`, direct lookup into the downloaded ICD-10/RxNorm data for candidates —
to get an early valid submission, then layer in a self-hosted ≤9B LLM (candidate: Qwen2.5-7B-Instruct +
QLoRA) for NER/assertion refinement and a retrieval pipeline (BM25 + embeddings + reranking) for entity
linking. Keep this separation of concerns at every stage:
- LLM/rules → reasoning-heavy steps (span detection, type classification, assertion inference)
- Python code → precision-heavy steps (character position calculation, JSON schema validation)
- Retrieval, not free generation → knowledge-heavy steps (ICD-10/RxNorm code assignment), so the
  model never hallucinates a code that doesn't exist in the reference data
