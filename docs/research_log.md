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
- `build_vectorstore.py`: ChromaDB with all-MiniLM-L6-v2 embeddings, cosine similarity, persistent at `data/vectorstore/`
- `llm_interface.py`: Ollama (qwen2.5:14b-instruct) as default backend
- `extract_failure_modes.py`: seed query retrieval → LLM extraction → agglomerative clustering
- Data moved to external drive (`/Volumes/Crucial X9/paper_datasets/finllm/data/`), symlinked back

**First full run results:** 336 failure modes from 144/249 papers, 25 clusters. Dominant cluster: "numerical reasoning error" (144 instances, 50 papers — 43% of all modes).

**Cluster breakdown (25 clusters):**
- numerical reasoning error: 144 instances, 50 papers (dominant — likely over-merged)
- performance degradation: 31, 19 papers
- data bias: 27, 21 papers
- generalizability: 24, 14 papers
- dataset limitation: 20, 20 papers
- hallucinations: 12, 12 papers
- High-precision Computational Distortion, over reliance, token_ceiling, isolated anomaly detection error, framework/API misuse, truthfulness, network/integration barriers, temporal aggregation gap, negative Sharpe ratio, under-fitting, table structure misinterpretation, unobservable_training_overlap, Multivariate Integrated Analysis Deviation (MIAD), statistical significance, domain knowledge gap, non-deterministic behavior, calling functions without providing required arguments, Cross-Lingual Adaptation Issues, input sensitivity

**Cluster naming problem:** Many cluster names are LLM-generated artifacts rather than normalized taxonomy labels (e.g., "Multivariate Integrated Analysis Deviation", "High-precision Computational Distortion"). These need normalization in later iterations.

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

**v2.0 sample (10 papers):** 26 failure modes, 9 clusters. Phase 1 now finds 181 candidates (up from 144).

v2.0 cluster breakdown: numerical reasoning error (15), evidence attribution error (4), domain-specific knowledge gap (3), hallucination (2), critical data missing (2), unanswerable span misclassification (1), over reliance (1), isolated anomaly detection error (1), calling functions without providing required arguments (1).

**v2.1 prompt adjustment:** Added explicit guidance that error case studies, qualitative examples, confusion matrices, and per-category performance gaps count as valid evidence. Added "when in doubt, extract with confidence medium" to reduce false negatives.

**v2.1 sample (10 papers):** 22 failure modes, 5 clusters. 4 papers returned empty [].

v2.1 cluster breakdown: numerical reasoning error (11), poor performance (8), data contamination (1), hallucination (1), irrelevant comments (1).

**v2.1 overcorrection problem:** v2.1 had MORE false negatives than v2.0 despite prompt improvements intended to increase recall. The stricter prompt overcorrected — the explicit exclusion list reduced false positives but also suppressed legitimate extractions. Fewer clusters (5 vs 9) and fewer modes (22 vs 26) suggest the model became too conservative.

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

### Detailed extraction comparison (4 test papers)

**Paper 1: FinTagging (074-042-452-083-012) — Benchmarking LLMs for Extracting and Structuring Financial Information**

| | Qwen 2.5 14B | Gemini 3 Flash Preview |
|---|---|---|
| Modes | 0 | 5 |
| Time | 52s | 15s |
| Categories | — | semantic ambiguity error, fine-grained concept differentiation failure, performance collapse in full-taxonomy settings, knowledge-alignment gap, structure-aware reasoning deficiency |
| Confidence | — | all high |
| Evidence types | — | qualitative_example (2), ablation_result (1), quantitative_result (1), author_interpretation (1) |

Context: Chunk 21 (error_analysis section) contains a concrete DeepSeek-V3 error case for GAAP concept confusion, but the chunk is ~90% prompt templates with the error case in the first 4 sentences. Qwen could not extract signal from noise. Gemini identified 5 distinct failure patterns.

**Paper 2: FinDABench (107-232-497-691-567) — Benchmarking Financial Data Analysis Ability of Large Language Models**

| | Qwen 2.5 14B | Gemini 3 Flash Preview |
|---|---|---|
| Modes | 0 | 4 |
| Time | 34s | 23s |
| Categories | — | poor benchmark performance, entity extraction hallucination, reasoning and technical skill deficiency, metric-performance gap |

Context: Initially assessed as "correct reject" (generic limitations only). Reassessed after Gemini found concrete evidence: entity hallucination where the model confused "Rongxin Group" with "Unicredit China" despite explicit text — a concrete qualitative example Qwen missed entirely. Also found quantitative evidence of poor performance across 41 models. Re-assessment: the paper does contain extractable failure evidence. This is not over-extraction — it is better recall.

**Paper 3: FinanceReasoning (167-398-185-509-339) — Benchmarking Financial Numerical Reasoning**

| | Qwen 2.5 14B | Gemini 3 Flash Preview |
|---|---|---|
| Modes | 2 (duplicates) | 8 (diverse) |
| Time | 54s | 18s |
| Categories | numerical reasoning error (×2, same evidence) | domain knowledge deficit, formula application error, data extraction error, numerical calculation error, context distraction, semantic ambiguity, inference inefficiency, numerical precision error |

Context: Paper documents 4 error types from 80 DeepSeek-R1 failure cases with stratified sampling (20 Easy / 20 Medium / 40 Hard). The paper's own error taxonomy: Misunderstanding of Problem, Formula Application Error, Data Extraction Error, Numerical Calculation Error. Qwen collapsed these into 2 duplicate "numerical reasoning error" entries. Gemini recovered all 4 of the paper's own categories plus found additional patterns from benchmark data (context distraction, semantic ambiguity, inference inefficiency, numerical precision error).

**Paper 4: FinMaster (196-068-497-575-765) — A Holistic Benchmark for Mastering Full-Pipeline Financial Workflows**

| | Qwen 2.5 14B | Gemini 3 Flash Preview |
|---|---|---|
| Modes | 7 (all same category) | 14 (14 distinct) |
| Time | 106s | 28s |
| Categories | numerical reasoning error (×7) | domain knowledge deficiency, critical data omission, numerical precision error, logical inconsistency, performance degradation on complexity, hallucination, logical reasoning error, structural reasoning failure, multi-step reasoning error, financial concept confusion, arithmetic error, financial logic error, data parsing failure, modality limitation |

Context: Paper has 68 chunks and a multi-type error taxonomy (Record Error, Calculation Error, Mismatch Error, multi-error combinations). Qwen collapsed everything into "numerical reasoning error" — extracting 7 instances of the same label from different calculation discrepancies. Gemini preserved the paper's error diversity and added evidence from qualitative examples.

### Decision: Use Gemini 3 Flash Preview for full extraction

**Rationale:**
1. 3.4x more failure modes per paper (7.8 vs 2.3) with meaningful diversity
2. 0 false negatives vs 2/4 missed by Qwen
3. 3x faster (21s vs 61s per paper)
4. Preserves papers' own error taxonomies instead of collapsing them
5. Finds concrete qualitative examples that Qwen ignores

**Implementation:** Free tier with retry policy (15s initial delay, 6 attempts, exponential backoff). Script auto-detects daily quota exhaustion and can resume from cache on re-run. If daily quota caps at ~50 RPD, full run (181 papers) would take ~4 days. If quota is higher, could finish in one session.

**Cost if paid:** ~$0.80 total (181 papers × ~6K input tokens × $0.50/1M + ~500 output tokens × $3.00/1M).

### Pipeline configuration reference

- **ChromaDB:** cosine similarity, all-MiniLM-L6-v2 embeddings, persistent at `data/vectorstore/`
- **Phase 1 retrieval:** 11 seed queries × 40 results each + section filter (`error_analysis`, `limitations`) × 200 results, distance threshold 0.8
- **Phase 2 context:** top 12 chunks per paper, sorted by priority-boosted distance
- **Gemini retry config:** initial_delay=15s, max_delay=120s, exp_base=2, attempts=6, timeout=180s
- **Data storage:** external drive (`/Volumes/Crucial X9/paper_datasets/finllm/data/`) symlinked to project `data/` directory

## 2026-03-21 — First Gemini full run (free tier)

Launched full extraction using Gemini 3 Flash Preview on all 181 candidate papers.

**Run status:** Processed 17/181 papers before daily quota exhaustion.

**Results from first 17 papers:**
- 118 failure modes extracted (6.9 per paper average — consistent with 7.8 avg from 4-paper comparison)
- 24 clusters generated

**Full cluster list (17 papers):**

| Cluster | Instances | Papers |
|---------|-----------|--------|
| numerical reasoning error | 37 | 13 |
| reasoning consistency error | 13 | 9 |
| performance degradation | 9 | 9 |
| domain knowledge deficiency | 8 | 7 |
| factual hallucination | 7 | 7 |
| semantic ambiguity | 6 | 6 |
| evidence attribution error | 4 | 2 |
| contextual inconsistency | 4 | 4 |
| multimodal processing limitation | 4 | 3 |
| table structure misinterpretation | 3 | 3 |
| retrieval-augmented generation failure | 3 | 3 |
| regulatory non-compliance | 3 | 2 |
| temporal confusion | 3 | 2 |
| computational error propagation | 2 | 2 |
| critical data omission | 2 | 2 |
| output nondeterminism | 2 | 1 |
| trend assessment error | 1 | 1 |
| low zero-shot accuracy | 1 | 1 |
| model refusal | 1 | 1 |
| task-specific drift sensitivity | 1 | 1 |
| security vulnerability | 1 | 1 |
| benchmark data contamination | 1 | 1 |
| ranking and ordering error | 1 | 1 |
| overconfidence | 1 | 1 |

**Notable new categories not seen in any Qwen run:** regulatory non-compliance, model refusal, RAG failure, output nondeterminism, security vulnerability, overconfidence. These suggest Gemini is recovering failure modes that Qwen systematically missed.

**Performance:**
- Rate: ~25s per paper, 5s inter-request delay for free tier
- Cost estimate for remaining 164 papers: $0.98 (Gemini 3 Flash Preview) or $0.67 (Gemini 2.5 Flash)

**Next step:** Resume extraction when quota resets. 164 papers remaining.

## 2026-03-21 — Full Gemini extraction (paid tier)

Added billing to Google AI Studio. Cleared cached extractions and re-ran full extraction on all 181 candidate papers.

**Run completed successfully:** 181/181 papers processed, 0 errors, 0 rate limit hits.

**Results:** 645 failure modes from 181 papers, 25 clusters.

| Cluster | Instances | Papers |
|---------|-----------|--------|
| numerical reasoning error | 258 | 110 |
| benchmark underperformance | 89 | 64 |
| information extraction error | 48 | 31 |
| evaluation bias | 39 | 26 |
| hallucination | 35 | 34 |
| context window limitation | 25 | 22 |
| robustness weakness | 22 | 18 |
| temporal confusion | 21 | 21 |
| instruction following failure | 14 | 11 |
| regulatory non-compliance | 13 | 10 |
| model alignment failure | 12 | 10 |
| output instability | 11 | 11 |
| input length limitation | 8 | 8 |
| prompt sensitivity | 7 | 7 |
| evaluation metric insensitivity | 7 | 7 |
| chart type sensitivity | 7 | 7 |
| multi-turn compliance degradation | 7 | 7 |
| schema coverage gap | 5 | 4 |
| incorrect factor weighting | 4 | 4 |
| brittleness | 4 | 4 |
| pipeline integration failure | 3 | 3 |
| overgeneralization | 2 | 2 |
| distributional shift | 2 | 2 |
| backbone model dependency | 1 | 1 |
| label confusion | 1 | 1 |

**Comparison: Qwen v1 full run vs Gemini v2 full run**

| Metric | Qwen 2.5 14B (v1) | Gemini 3 Flash Preview (v2) |
|--------|--------------------|-----------------------------|
| Papers processed | 144 / 249 | 181 / 249 |
| Total failure modes | 336 | 645 |
| Avg modes per paper | 2.3 | 3.6 |
| Clusters | 25 | 25 |
| Dominant cluster share | 43% (numerical reasoning) | 40% (numerical reasoning) |
| Cluster label quality | Many LLM artifacts (e.g., "Multivariate Integrated Analysis Deviation") | Clean, interpretable labels |
| Finance-specific clusters | Few | temporal confusion (21), regulatory non-compliance (13), multi-turn compliance degradation (7), chart type sensitivity (7) |
| Zero-extraction papers | Not tracked | To be verified |
| Runtime | ~2.5 hours | ~65 minutes |
| Cost | $0 (local) | ~$1 (paid API) |

**Key observations:**
1. "Numerical reasoning error" still dominant at 40% — likely needs sub-clustering in taxonomy assembly phase
2. The remaining 60% is well-distributed across 24 meaningful categories
3. Finance-specific failure modes now visible: temporal confusion, regulatory non-compliance, multi-turn compliance degradation, chart type sensitivity, schema coverage gap
4. Cluster labels are clean and interpretable — no LLM naming artifacts
5. 181 vs 144 papers: Gemini processed 37 more papers (recovered by removing priority gate in Phase 1)
6. 68 papers in the vectorstore (249 - 181) still not reached by any seed query — these may not contain failure-relevant content

**Stream A extraction is complete.** This is the checkpoint before proceeding to Stream B.

**Output files:**
- `data/processed/failure_extractions/{lens_id}.json` — per-paper extractions (181 files)
- `data/processed/stream_a_failure_modes.csv` — aggregate (645 rows)
- `data/processed/stream_a_failure_clusters.csv` — clustered (645 rows with cluster assignments)

## 2026-03-22 — Quality review: code-level and LLM-specific extraction risks

Systematic review of the extraction pipeline code (`extract_failure_modes.py`, `llm_interface.py`, `compare_models.py`) for quality issues beyond the four user-identified concerns (omission, misclassification, granularity, noise).

### Code-level issues identified

1. **`extract_json` silent failure** — If LLM returns malformed JSON, `extract_json()` returns `[]` silently. Indistinguishable from a paper that genuinely has no failure evidence. Risk: unknown number of papers may have extraction failures masked as "no failures found."

2. **Gemini system prompt handling** — `GeminiLLM.generate()` concatenates system and user prompts as a single string (`f"{system}\n\n{prompt}"`), unlike Ollama which uses proper role-based messages. Gemini may weight system instructions differently when they appear as user text vs a system instruction parameter. Extraction behavior may subtly differ from what the prompt engineering intended.

3. **Chunk text truncation risk** — Phase 2 takes top 12 chunks per paper with no token budget check. Papers with long chunks could exceed the model's effective context window, causing the LLM to ignore later chunks or degrade extraction quality on them. The 12-chunk cap is arbitrary and not validated against actual token counts.

4. **Priority boost is tiny** — `chunk_sort_key` adds 0.1 to distance for non-high-priority chunks. Since cosine distances in the store range ~0.3–0.8, this boost barely affects sort order. A chunk at distance 0.35 (non-priority) still ranks above a chunk at distance 0.46 (high-priority). The boost is cosmetic rather than functional.

5. **Section filter query text mismatch** — The section filter query uses `"error analysis failure limitation"` as query text but filters to `section_type in [error_analysis, limitations]`. The query text influences which chunks among those sections are returned first (via `n_results=200` cap). A more neutral query or no query text would be more appropriate since the section filter already targets the right content.

### LLM-specific extraction risks

6. **Category label drift** — Gemini generates free-text `failure_category` labels. The same failure may get different labels across papers (e.g., "numerical calculation error" vs "arithmetic error" vs "computational error"). Clustering partially addresses this, but agglomerative clustering on label embeddings may miss semantic equivalences or create false separations.

7. **Confidence field unreliable** — In the 4-paper comparison test, Gemini assigned "high" confidence to every extraction. The confidence field may not discriminate between strong and weak evidence, reducing its utility for QA filtering.

8. **Cross-chunk synthesis gaps** — The extraction prompt shows chunks in sequence but doesn't explicitly instruct the LLM to synthesize across chunks. If a failure description spans two chunks (e.g., error category defined in methods, evidence in results), the LLM may extract partial information or miss the connection.

9. **Position bias** — LLMs exhibit primacy/recency bias. Chunks sorted by distance appear in a fixed order. Failure evidence in later chunks (positions 8-12) may receive less attention than evidence in early chunks (positions 1-3), even if equally important.

### Structural / pipeline risks

10. **No deduplication across papers** — If multiple papers report the same failure (e.g., GPT-4 poor at numerical reasoning), each paper generates separate extractions. The clustering step groups by category label similarity but doesn't deduplicate identical findings. The aggregate CSV may overcount prevalent failure modes.

11. **Clustering on labels only** — Agglomerative clustering uses only the `failure_category` text embedding, ignoring `description`, `task_type`, and `evidence_type`. Two extractions with different labels but identical descriptions end up in different clusters. Conversely, two with similar labels but different actual failures merge.

12. **68 papers never queried** — 249 papers in vectorstore, only 181 reached by Phase 1. See investigation below.

13. **No negative validation** — No mechanism to verify that papers returning `[]` genuinely lack failure content. The false negative rate is unknown beyond the 4-paper comparison test.

14. **Cluster count formula is arbitrary** — `n_clusters = min(max(5, len(categories)//4), 25)`. With 645 modes and ~180 unique category labels, this gives 25 clusters. The formula has no empirical basis and may over- or under-merge.

### Assessment

Issues 1 (silent JSON failure), 6 (label drift), 12 (68 unqueried papers), and 13 (no negative validation) are the highest priority. Issues 4 (priority boost), 5 (query text), and 7 (confidence) are low impact. The remaining issues are medium priority and partially addressable during taxonomy assembly (Step 5).

## 2026-03-22 — Investigation: 68 unqueried papers (249 - 181)

Investigated why 68 papers in the ChromaDB vectorstore were never selected as candidates by Phase 1.

### Root cause: `n_results` caps, NOT distance threshold

Every single one of the 68 missing papers has chunks with similarity distance well below the 0.80 threshold. The problem is purely competitive exclusion from fixed-size result windows.

### Mechanism 1: Seed query cap too small (affects ~57 papers)

Phase 1 runs 11 seed queries with `n_results=40` each. The collection has 5,377 chunks across 249 papers. At 40 slots per query, papers whose most relevant chunk sits at rank 41+ for every query are systematically excluded.

| Best rank of missing paper's chunk | Papers |
|---|---|
| rank ≤ 40 (would have been retrieved) | 0 |
| rank 41–60 (one slot away) | 6 |
| rank 61–100 | 16 |
| rank > 100 | 52 |

Example: paper `008-659-966-483-52X` (Explainable Document Level QA) has a chunk at distance 0.4591 on the "failure cases financial question answering" query, but the 40th slot goes to distance 0.4568 — a margin of 0.0023.

### Mechanism 2: Section filter cap too small (affects ~11 papers)

The section-type filter query (`where section_type in [error_analysis, limitations]`, `n_results=200`) is intended as a catch-all. However, the store contains 274 chunks with those section types, but the query only returns 200. The 74 least-similar chunks are silently dropped, and 11 missing papers have their only error_analysis/limitations chunk in positions 201-274.

### Notable excluded papers

The 68 missing papers include substantive work: BloombergGPT, FinSQL, Golden Touchstone, M³FinMeeting. These are not irrelevant papers — they were competitively excluded.

### Fix (not yet implemented)

Increase `n_results=40` to `80` or `100` in seed queries, and increase `n_results=200` to `400` (or use `collection.count()`) in the section filter. The 0.80 distance threshold does NOT need adjustment.
