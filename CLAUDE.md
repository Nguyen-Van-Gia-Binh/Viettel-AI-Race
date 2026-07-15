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
4. `Plan.md` — **the approved execution roadmap.** Workstream split (A = KB & entity linking,
   B = NER/assertion/LLM, C = shared infra/eval — originally planned for 3 people, now all being
   picked up by one person across separate conversations), phased 4-week timeline with go/no-go
   gates, full `src/clinlp/` module map, the RxNorm data-acquisition plan, the eval-harness design,
   and a repo-specific risk list. Read this before starting new work — it explains *why* the module
   boundaries below are where they are.

## Current status

**Workstream C (shared infra) is built and tested — reuse these, do not recreate them:**
- `src/clinlp/schema.py` — `Concept`/`ConceptType`/`Assertion`, the shared output contract every
  other module codes against.
- `src/clinlp/io_utils.py` — the only place code should touch corpus/JSON files
  (`read_text`/`write_json`/`read_jsonl`/`write_jsonl`, always UTF-8, newline-safe).
- `src/clinlp/preprocessing.py` — `normalize_for_llm()`, NFC normalization for the LLM-facing working
  copy only. Never feed its output into position calculation.
- `src/clinlp/position.py` — `PositionResolver`, 3-tier span resolution (exact → whitespace-flexible →
  Unicode-flexible), assigns duplicate entities to distinct occurrences in order, never guesses.
- `src/clinlp/validate.py` — `validate_concepts()`/`has_errors()`, position/type/assertion/candidates/
  duplicate checks, returns a list of issues instead of raising on the first one.
- `src/clinlp/eval/scorer.py` — `score_file()`/`corpus_score()`, reimplements the scoring formula
  below. Documents an explicit assumption about gold/pred matching (same-type + overlapping-position,
  greedy max-overlap) since the organizers have not published a reference scorer — revisit this module
  first if that assumption turns out wrong.
- `src/clinlp/pipeline.py` — `run_pipeline(source_text, ner_fn, link_fn)`. **Deviates from `Plan.md`'s
  literal `mode="baseline"|"hybrid"` flag**: instead of a branch inside the module, it takes `ner_fn`/
  `link_fn` as injected callables, so baseline vs. hybrid is just "which functions you pass in" at the
  call site (future `scripts/run_baseline.py` vs `scripts/run_inference.py`), not internal branching.
- `scripts/survey.py` — moved from `src/`, unchanged.
- `tests/test_position.py`, `test_schema_validate.py`, `test_scorer.py`, `test_pipeline.py` — 28
  passing tests. Several are grounded in real edge cases from `input/input/*.txt` (repeated entities
  in `1.txt`, the missing-space line-wrap join in `31.txt`, the dense vitals token in `1.txt`) —
  extend this pattern rather than writing purely synthetic test cases.

**Workstream B (NER/assertion) is built and tested — Week-1 rule-based scope, no LLM yet:**
- `src/clinlp/sections.py` — `parse_sections()`/`build_section_index()`/`SectionIndex`, a
  full-line header regex (extends `scripts/survey.py`'s heuristic) that deliberately does NOT match
  "Label: value" lines like "Thuốc trước khi nhập viện: aspirin ..." (`input/input/22.txt`) — those
  still end up historical via the *enclosing* numbered section ("1. Tiền sử bệnh"), not their own
  header. Flat (non-nested) section spans are enough for the two propagation rules needed.
- `src/clinlp/ner/dictionaries.py` — `NEGATION_CUES` seeded verbatim from
  `Sota_research_report.md` §4.2 (pre/post negation, family, historical, section_historical cues) +
  a symptom/diagnosis/drug dictionary sampled from real terms in `input/input/*.txt`.
- `src/clinlp/ner/rules.py` — `extract_raw_matches()`, regex + dictionary-based span/type detection.
- `src/clinlp/ner/assertions.py` — `detect_assertions()`/`detect_negation()`/etc.: Layer 1 NegEx
  (clause-scoped, long-range pre-negation, narrow post-negation window, implicit `(-)`/`(+)` symbols,
  a structural exception for the "không ... cản quang" ["without contrast"] radiology idiom) + Layer
  2 section propagation via `sections.py`. Layer 3 (LLM) is a stub for Week 3.
- `src/clinlp/ner/pipeline_adapter.py` — `ner_fn(source_text)`, the final callable wired into
  `run_pipeline()`; matches `pipeline.NerFn` exactly.
- `tests/test_sections.py`, `test_ner_rules.py`, `test_assertions.py` — grounded in real lines from
  `input/input/1.txt` (long-range negation, post-negation non-leak), `22.txt` (section propagation,
  radiology-idiom exception), `31.txt` (implicit `(±)` symbols).

**Workstream A (entity linking) is built and tested — Week-1 fuzzy-match scope, no
BM25/embedding/reranker yet:**
- `src/clinlp/linking/icd10_kb.py` — `reformat_icd10_code()` (`K219`→`K21.9`; 3-char category
  codes like `A00` left undotted), `build_icd10_kb()` parses the fixed-width order file into
  `data/ICD-10/icd10_processed.csv` (97k+ rows). Descriptions are English-only from the CMS source;
  a small hand-verified `VI_ALIASES` table seeds Vietnamese phrases for a handful of recurring
  diagnoses (GERD, tăng huyết áp, đái tháo đường, viêm phổi, ...) — real bilingual coverage is still
  `linking/translate.py`, not yet built (see "known gaps" below).
- `src/clinlp/linking/rxnorm_kb.py` — Path 2 (RxNav REST `allconcepts.json` bulk endpoint, no auth,
  fetched per term-type IN/PIN/SCD/SBD/BN/GPCK/BPCK) implemented and run:
  `data/RxNorm/rxnorm_processed.csv` (52k+ rows), raw responses cached in
  `data/RxNorm/raw_rest_cache/`. Path 1 (NLM UTS full release) is still open — needs manual account
  signup, left as follow-up. **Known Path-2 gap:** `allconcepts` only returns currently-active
  concepts; `Topic.md`'s own two worked-example RxCUIs (360047, 1660761) are historical/obsolete and
  patched in individually via a targeted `historystatus.json` lookup (tagged
  `source=historical_supplement` in the CSV) — a 2-row documented exception, not a general mechanism.
- `src/clinlp/linking/link.py` — `link(entity_text, concept_type, source_text) -> list[str]`,
  matches `pipeline.LinkFn` exactly. Fuzzy-matches against the two CSVs above via `rapidfuzz`.
  **Uses `fuzz.partial_ratio`, not `fuzz.WRatio`** — found empirically while adding tests that
  `WRatio` ties dozens of unrelated same-length KB rows together (e.g. RxCUI 1660761 for "Capsaicin
  0.38 MG/ML" fell out of the top 40 entirely); `partial_ratio` ranks the true match #1 on every
  `Topic.md` worked example tested. No network import in this file — only `rxnorm_kb.py`/`icd10_kb.py`
  touch the network, both offline/build-time only.
- `tests/test_icd10_reformat.py`, `test_link.py` — reformat edge cases + `Topic.md`'s own worked
  examples (GERD → K21.0/K21.9, Chlorpheniramine → 360047, Capsaicin → 1660761).

**Integration — Week-1 gate PASSED:**
- `scripts/run_baseline.py` — wires `ner.pipeline_adapter.ner_fn` + `linking.link.link` through
  `pipeline.run_pipeline()` over all `input/input/{1..100}.txt`, writes `output/{n}.json`.
- `scripts/make_submission.py` — zips `output/{1..100}.json` into `output.zip`, refusing to build it
  if any file is missing/malformed.
- Result: `pytest tests/` is 58/58 passing; `output.zip` built from all 100 files, 854 concepts
  total, 0 validation ERROR issues, 0 WARNING issues (every CHẨN_ĐOÁN/THUỐC concept got at least one
  candidate).
- `scripts/build_icd10_index.py`/`scripts/build_rxnorm_index.py` (thin CLI wrappers around
  `build_icd10_kb()`/`build_rxnorm_kb()`) are not built — those functions are called directly today;
  add the wrapper scripts only if a need to rebuild the KBs standalone comes up.

**Known gaps to prioritize in Week 2 (not blocking the Week-1 gate, but real quality issues):**
- CHẨN_ĐOÁN candidate quality is weak for diagnoses with no `VI_ALIASES` entry — the ICD-10 KB is
  English-only, so fuzzy-matching a Vietnamese entity against it is close to noise for anything not
  hand-seeded (e.g. `output/22.json`'s "khối máu tụ dưới màng cứng" [subdural hematoma] linked to
  unrelated diabetes codes). `linking/translate.py` (bilingual glosses) or a real retrieval swap
  (`retrieval.py`/`rerank.py`, already scoped for weeks 2-3 in `Plan.md`) is the fix, not more
  fuzzy-matching tuning.
- No quantitative score yet — `eval/scorer.py` exists but the gold sets (`Plan.md`'s "Eval harness"
  section, Workstream C responsibility) haven't been built, so baseline quality is currently only
  checked structurally (`validate_concepts`), not against `final_score`.

**Interfaces implemented against** (defined in `src/clinlp/pipeline.py` — unchanged, do not redesign
without updating `pipeline.py` itself):
- `ner_fn(source_text: str) -> Iterable[tuple[str, ConceptType, list[Assertion]]]` — implemented by
  `ner/pipeline_adapter.py:ner_fn`.
- `link_fn(entity_text: str, concept_type: ConceptType, source_text: str) -> list[str]` — implemented
  by `linking/link.py:link`; only called for `CHẨN_ĐOÁN`/`THUỐC`.

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
- All of this is encoded in `src/clinlp/schema.py`'s `Concept` model — use `Concept.to_json_dict()`/
  `Concept.from_json_dict()` rather than hand-building the JSON shape.

## Critical constraints (competition rules)

- The model used **at inference time** must be self-hosted, ≤9B parameters, no external API calls
  (no GPT-4/Claude/etc. in the live pipeline). `ner/llm_extract.py` and everything under `linking/`
  must never import `requests`/`urllib` — only `dataprep/teacher_client.py` and the one-time
  `scripts/build_*_index.py` scripts are allowed to touch the network.
- External APIs/LLMs **are** allowed for offline prep — generating synthetic training data, building
  the ICD-10/RxNorm reference lookups, etc. — just never called live against the test set.
- Top ~15 teams must submit reproducible source + model weights + data + README so organizers can
  rebuild and re-score on a private test set; keep the pipeline reproducible from a clean checkout
  (`pip install -e .` + `pip install -r requirements.txt` must be sufficient for the non-GPU parts).

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
  This is what `src/clinlp/position.py` does — never bypass it.
- `src/clinlp/eval/scorer.py` reimplements this exact formula for offline evaluation (see "Current
  status" above for its documented matching assumption). Run it against a gold set before and after
  every pipeline change — see `Plan.md`'s "Eval harness" section for how the gold sets are built.

## Repo layout

- `input/input/*.txt` — the 100 provided test records; the only data given by the organizers.
- `data/ICD-10/` — manually downloaded ICD-10-CM reference files (CMS, public domain, no login):
  - `icd10cm_order_2024.txt` — fixed-width: seq_num, code (no decimal point), a flag digit, long description, short description
  - `icd10cm_codes_2024.txt` — simpler 2-column list: code (no decimal point) + description
  - `*_addenda_2024.txt` — year-over-year changes; `*.pdf` — official format docs
  - Codes in these files omit the decimal point (e.g. `K219`, not `K21.9`) — reformatted to the
    dotted style by `linking/icd10_kb.py` into `data/ICD-10/icd10_processed.csv` (already built).
  - `data/RxNorm/` — built from the RxNav REST `allconcepts` bulk endpoint (Path 2 of `Plan.md`'s
    "Lấp gap RxNorm" section) by `linking/rxnorm_kb.py`: `rxnorm_processed.csv` + raw responses
    cached in `raw_rest_cache/`. Path 1 (NLM UTS full release) still open as follow-up.
- `src/clinlp/` — the pipeline package (`pip install -e .` makes it importable as `clinlp`). See
  "Current status" above for what exists vs. what's still to build; see `Plan.md` for the full
  intended module map (`ner/`, `linking/`, `dataprep/`, `eval/`, `llm/`).
- `scripts/` — CLI entrypoints. `survey.py` (corpus stats, moved from `src/`), `run_baseline.py`
  (Week-1 rule-based baseline over all 100 files), `make_submission.py` (zips `output/*.json` into
  `output.zip`). `run_inference.py` (future hybrid pipeline), `build_icd10_index.py`,
  `build_rxnorm_index.py` (thin wrappers, not yet needed) are still planned.
- `src/download_icd10.py` — one-time ICD-10 bulk downloader via the NLM Clinical Tables API,
  superseded by the manually downloaded CMS files in `data/ICD-10/`. Kept as reference/history only.
- `tests/` — pytest suite for everything under `src/clinlp/`; run with `pytest tests/` (58 passing).
- `output/` — holds generated `{1..100}.json` predictions from `scripts/run_baseline.py`.
- `output.zip` — Week-1 gate submission artifact, built by `scripts/make_submission.py`.

## Commands

- `pip install -e .` — install the `clinlp` package in editable mode (see `pyproject.toml`).
- `pip install -r requirements.txt` — core deps (no GPU needed) for the baseline pipeline, eval
  harness, and tests. GPU/LLM deps (QLoRA, vllm, etc.) are in `requirements-llm.txt` — install those
  only on a Linux/WSL2 training-serving host, not the plain-Windows dev box (`bitsandbytes`/`vllm`/
  flash-attention have weak-to-absent native Windows support).
- `pytest tests/` — run the test suite.
- Windows console defaults to cp1252 and cannot print Vietnamese text — any script that prints
  Vietnamese strings to stdout must be run with UTF-8 output forced:
  ```
  PYTHONIOENCODING=utf-8 python scripts/<script>.py
  ```
- No linter/build system beyond `pytest` is set up yet.

## Architecture direction

Agreed approach (see `Plan.md` for the full phased roadmap, and `Topic_Analysis.md` §5 /
`Sota_research_report.md` §6 for the original research this was derived from): rule-based baseline
first — dictionary/regex NER, Vietnamese NegEx-style assertion rules, section-header propagation for
`isHistorical`, direct lookup into the downloaded ICD-10/RxNorm data for candidates — to get an early
valid submission, then layer in a self-hosted ≤9B LLM (candidate: Qwen2.5-7B-Instruct + QLoRA) for
NER/assertion refinement and a retrieval pipeline (BM25 + embeddings + reranking) for entity linking.
Keep this separation of concerns at every stage:
- LLM/rules → reasoning-heavy steps (span detection, type classification, assertion inference)
- Python code → precision-heavy steps (character position calculation, JSON schema validation)
- Retrieval, not free generation → knowledge-heavy steps (ICD-10/RxNorm code assignment), so the
  model never hallucinates a code that doesn't exist in the reference data

The module boundaries in `src/clinlp/` map directly onto this: `ner/` and `sections.py` own the
reasoning-heavy steps, `position.py`/`validate.py`/`schema.py` own the precision-heavy steps, and
`linking/` owns the knowledge-heavy retrieval-first entity linking. `pipeline.py`'s `ner_fn`/`link_fn`
injection point is what keeps these swappable independently as the baseline is upgraded to the hybrid
pipeline — see `Plan.md`'s "Quyết định kiến trúc chính" for the reasoning behind this.
