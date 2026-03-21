# FinLLM Research Log

Chronological record of research design decisions, pipeline changes, and rationale.

---

## 2026-03-18 — Initial pipeline design (v1)

- Two-stream approach: Stream A (documented LLM failures), Stream B (inherent task difficulty)
- Both streams treated as parallel failure sources feeding into a joint taxonomy
- Plan documented in `docs/plan_failure_extraction_v1.md`

## 2026-03-18 — Stream B reframe (v2)

**Change:** Stream B is no longer a parallel failure source. It is the evaluation framework for Stream A.

**Rationale:** Accounting/finance has its own definition of "failure" governed by materiality, relevance vs reliability, faithful representation, and prudence. Without this lens, the taxonomy just catalogues technical errors without evaluating which ones actually matter. Stream B answers: "is this actually a failure?", "how significant is it?", "is there a human baseline?"

**Impact:** Streams now execute sequentially (A first, checkpoint, then B informed by A results). Plan documented in `docs/plan_failure_extraction_v2.md`.

## 2026-03-19 — Pipeline implementation (v1 extraction)

- `parse_papers.py`: 249 Stream A PDFs → 5377 chunks (1289 high-priority)
- `build_vectorstore.py`: ChromaDB with all-MiniLM-L6-v2 embeddings
- `llm_interface.py`: Ollama (qwen2.5:14b-instruct) as default backend
- `extract_failure_modes.py`: seed query retrieval → LLM extraction → agglomerative clustering
- Data moved to external drive (`/Volumes/Crucial X9/paper_datasets/finllm/data/`), symlinked back

**First full run results:** 336 failure modes from 144/249 papers, 25 clusters. Dominant cluster: "numerical reasoning error" (144 instances, 50 papers) — likely over-merged.

**Issues identified:**
- 82 of 331 filtered papers have no downloaded PDF (not a download error — papers were never fetched)
- 105 of 249 parsed papers dropped by Phase 1 because `where={"priority": "high"}` filter is too aggressive
- Seed queries mix true failure evidence with generic limitation language
- System prompt too permissive — Qwen extracts vague "limitation" statements that aren't real failure evidence
- Extraction prompt missing useful fields (evidence_type, trigger_or_condition)

Committed as `6317ab7`.

## 2026-03-20 — Extraction pipeline v2 design decisions

**Changes agreed (not yet implemented):**

1. **System prompt** — replace with version that includes explicit exclusion list (no generic future work, no unsupported speculation, no broad "task is challenging" statements)
2. **Extraction prompt** — add 3 fields: `evidence_type` (quantitative_result / qualitative_example / author_interpretation / benchmark_comparison / ablation_result), `trigger_or_condition`, `confidence` (high/medium). Total 9 fields.
3. **Phase 1 retrieval** — remove `where={"priority": "high"}` gate. Query all chunks, use priority as ranking signal only. This should recover the 105 dropped papers.
4. **Queries** — replace 10 generic queries with 8 tighter Tier A queries + 3 finance-specific queries (temporal confusion, entity confusion, hallucinated numbers). Total 11.
5. **Re-run** — delete cached extractions, full run on all candidates.

**Rejected alternatives:**
- Two-pass extraction (evidence spotting → normalization): doubles LLM time (~5h vs 2.5h), normalization achievable via post-hoc clustering
- Tiered query execution (Tier A then Tier B): adds complexity without fixing the real problem (priority filter gate)
- `failure_mechanism` field separate from `failure_category`: Qwen 14B likely to blur the distinction, not worth the risk of malformed JSON

**Risk:** 9 fields is upper limit for Qwen 14B. If malformed JSON rate exceeds ~10%, drop `confidence` first.

## 2026-03-20 — Extraction pipeline v2 implemented

**Changes applied to `src/extract_failure_modes.py`:**

1. System prompt replaced — explicit exclusion list (no generic future work, unsupported speculation, broad "challenging" statements, methodological descriptions without weakness evidence)
2. Extraction prompt now has 9 fields: failure_category, description, evidence_type, evidence, task_type, trigger_or_condition, models_tested, severity, confidence
3. Phase 1 retrieval: removed `where={"priority": "high"}` gate. All chunks are now candidates. Priority used as ranking boost when sorting chunks for extraction context. Section filter query expanded to n_results=200.
4. Queries replaced: 8 focused failure-evidence queries + 3 finance-specific (temporal confusion, entity confusion, hallucinated numbers). Total 11, down from 10 but more targeted.
5. Chunk context per paper increased from 10 to 12 (to accommodate wider retrieval).

**v1 cached extractions deleted before re-run.** v1 results preserved in git history (`6317ab7`).

**Comparison target:** v1 yielded 336 failure modes from 144 papers, 25 clusters. v2 should show: more papers (closer to 249), fewer but higher-quality extractions, better cluster separation.

## 2026-03-20 — v2 sample results and quality audit

**v2.0 sample (10 papers):** 26 failure modes, 5 clusters. Phase 1 now finds 181 candidates (up from 144).

**v2.1 prompt adjustment:** Added explicit guidance that error case studies, qualitative examples, confusion matrices, and per-category performance gaps count as valid evidence. Added "when in doubt, extract with confidence medium" to reduce false negatives.

**v2.1 sample (10 papers):** 22 failure modes, 5 clusters. 4 papers returned empty [].

**Quality audit on 4 papers (2 empty, 2 non-empty):**

| Paper | v2.1 Result | Actual content | Assessment |
|-------|-------------|----------------|------------|
| FinTagging (074) | 0 modes | Has concrete DeepSeek-V3 error case for GAAP concept confusion | **False negative** — error case buried in chunk that's 90% prompt templates |
| FinDABench (107) | 0 modes | Generic "insufficient data coverage" limitations | **Correct reject** |
| FinanceReasoning (167) | 2 modes (duplicates) | Has 4-type error taxonomy from 80 DeepSeek-R1 failure cases | **Under-extracted** — missed error taxonomy |
| FinMaster (196) | 5 modes (all "numerical reasoning error") | Has multi-type error taxonomy (Record/Calculation/Mismatch errors) | **Under-extracted, no diversity** — collapsed distinct categories |

**Root cause:** Qwen 2.5 14B limitations:
1. Cannot extract signal from noisy chunks (prompt templates mixed with error analysis)
2. Collapses distinct error categories into generic "numerical reasoning error"
3. Produces duplicate extractions for the same failure

**Decision:** Compare Qwen 2.5 14B vs Gemini on the same 4 papers. Created `src/compare_models.py` for reproducible comparison.

## 2026-03-21 — Model comparison: Qwen 2.5 14B vs Gemini 3 Flash Preview

Ran identical extraction (same prompts, same chunks) on 4 test papers.

### Results

| Paper | Qwen 14B | Gemini 3 Flash |
|-------|----------|----------------|
| FinTagging (noisy chunk with error case) | 0 modes, 52s | 5 modes, 15s |
| FinDABench (initially expected empty) | 0 modes, 34s | 4 modes, 23s |
| FinanceReasoning (4-type error taxonomy) | 2 modes (duplicates), 54s | 8 modes (diverse), 18s |
| FinMaster (multi-type error taxonomy) | 7 modes (all same category), 106s | 14 modes (14 distinct categories), 28s |

### Key findings

**Qwen 2.5 14B weaknesses:**
- Cannot extract signal from noisy chunks (FinTagging: error case buried in prompt templates → 0 extractions)
- Collapses distinct error categories into "numerical reasoning error" (FinMaster: 7 modes all same label)
- Produces duplicates (FinanceReasoning: 2 identical entries)
- Misses concrete qualitative examples (FinDABench: entity hallucination example ignored)

**Gemini 3 Flash Preview strengths:**
- Extracts from noisy chunks successfully (FinTagging: found semantic ambiguity, concept differentiation failure)
- Preserves paper's own error taxonomy (FinMaster: 14 distinct categories including arithmetic, data parsing, concept confusion, hallucination)
- Finds concrete qualitative examples (FinDABench: entity extraction hallucination — model confused "Rongxin Group" with "Unicredit China")
- 3-4x faster per paper

**Gemini 3 Flash Preview concerns:**
- May over-extract: FinDABench was initially assessed as "generic limitations only" but Gemini found 4 modes including a concrete entity hallucination example. Re-assessment: the paper does contain failure evidence that Qwen missed entirely. Not over-extraction — better recall.
- All confidence ratings are "high" — may not be discriminating enough. Need to verify on papers with genuinely no failure content.
- Free tier has daily quota limits (Gemini 2.0 Flash quota exhausted during testing, switched to Gemini 3 Flash Preview).

### Performance comparison (per paper)

| Metric | Qwen 14B | Gemini 3 Flash |
|--------|----------|----------------|
| Avg modes/paper | 2.3 | 7.8 |
| Avg time/paper | 61s | 21s |
| Distinct categories | 1 | 31 |
| False negatives | 2/4 papers | 0/4 papers |
| Cost | $0 (local) | Free tier (quota limited) |

### Decision

Gemini 3 Flash Preview is substantially better for this task. Options:
1. Use Gemini for full run if quota allows (~181 papers)
2. Use Gemini for a larger sample, then decide
3. Paid Gemini API if free tier insufficient (~$1-2 total)

Raw comparison data saved in `data/processed/model_comparison/`.
