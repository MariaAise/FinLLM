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
