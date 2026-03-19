# Plan: FinLLM Failure Taxonomy Pipeline (v2 — reframed)

## Overall goal

Build a taxonomy of LLM failure modes in finance/accounting. The taxonomy is not just a list of technical errors — it evaluates whether each "failure" is actually a failure in accounting/finance terms, how significant it is, and whether it's structural or solvable.

## Key insight driving the reframe

Stream A papers define "failure" as deviation from benchmark ground truth (technical accuracy). But accounting/finance has its own definition of failure governed by principles like:
- **Materiality vs accuracy** — errors below materiality threshold aren't failures
- **Relevance vs reliability** — a precise but irrelevant output is worse than an imprecise but decision-useful one
- **Faithful representation** — completeness, neutrality, not misleading (not same as "correct")
- **Prudence/conservatism** — some directional biases are preferred

Without this framework, the taxonomy just catalogues technical errors. With it, the taxonomy evaluates which technical errors actually matter in practice.

## Two streams, sequential execution

### Stream A (first): Extract technical failure modes
- What goes wrong when LLMs process financial/accounting tasks
- Technical categories: hallucination, numerical reasoning, table extraction, etc.
- Evidence: benchmark results, error analysis sections, limitations

### Stream B (second): Provide the evaluation framework
- How accounting/finance defines success vs failure
- Human expert disagreement rates (the baseline)
- Significance/impact of different error types
- Boundary conditions where any automated system structurally cannot produce a definitive answer

Stream B is NOT a parallel failure source. It's the **lens through which Stream A failures are evaluated**. It answers: "is this actually a failure?", "how significant is it?", "is there a human baseline to compare against?"

---

## Architecture

```
Stream A: PDFs → parse → ChromaDB → extract failure modes → raw taxonomy
                                                                  ↓
Stream B: PDFs → parse → ChromaDB → extract evaluation framework  ↓
                                                                  ↓
                              Evaluate raw taxonomy through Stream B lens
                                                                  ↓
                                                      Final taxonomy
```

Each taxonomy entry has:
- **Failure mode** (what goes wrong — Stream A)
- **Accounting significance** (does this matter? — Stream B)
- **Human baseline** (how do experts perform? — Stream B)
- **Structural vs solvable** (boundary condition or engineering problem? — Stream B)
- **Remedy pattern** (synthesis)

---

## Step 1: Parse PDFs into chunks (both streams)

**File: `src/parse_papers.py`**

- Input: PDFs from `data/papers/stream_{a|b}/`
- Extract text using `pymupdf`
- Detect section headings: Abstract, Introduction, Related Work, Method/Methodology, Results/Experiments, Discussion, Limitations, Error Analysis, Conclusion
- Chunk: one chunk per section (split if > 1500 tokens), fallback to fixed-size (512 tokens, 64 overlap)
- Tag each chunk: paper_id, doi, title, stream, query_block, section_type, chunk_index, priority (high for Results/Error Analysis/Limitations/Discussion)
- Output: `data/processed/stream_{a|b}_chunks.jsonl`
- Manifest: `data/processed/stream_{a|b}_parsed_manifest.csv`
- `--stream A|B` flag

---

## Step 2: Build vector store (both streams)

**File: `src/build_vectorstore.py`**

- Read chunks JSONL, embed with `all-MiniLM-L6-v2`
- ChromaDB persistent at `data/vectorstore/`
- Collections: `stream_a_papers`, `stream_b_papers`
- All metadata stored and filterable
- Reusable for all future queries during paper writing
- `--stream A|B` flag

---

## Step 3: Extract failure modes from Stream A

**File: `src/llm_interface.py`** — swappable wrapper (Ollama default, Gemini/Claude optional)

**File: `src/extract_failure_modes.py`**

### Phase 1: Candidate identification (free — vector store queries)
Query `stream_a_papers` with seed queries targeting failure content:
- "error analysis showing where the model failed on financial tasks"
- "numerical reasoning mistakes in financial question answering"
- "hallucination or factual errors in financial document analysis"
- "model limitations on accounting and auditing tasks"
- "performance breakdown by task type showing failure categories"
- "incorrect extraction from financial tables or SEC filings"
- "robustness failures when prompt or input format changes"
- "bias in financial sentiment or risk assessment"

Retrieve top-k per query, deduplicate by paper_id → candidate set.

### Phase 2: Structured extraction (LLM — candidates only)
For each candidate, retrieve high-priority chunks, send to LLM:

```
Extract every failure mode, error pattern, or limitation reported:
1. failure_category: short label
2. description: 1-2 sentences
3. evidence: direct quote
4. task_type: financial task attempted
5. models_tested: which model(s)
6. severity: author characterization
Return JSON array.
```

- Per-paper JSON: `data/processed/failure_extractions/{lens_id}.json`
- Aggregate: `data/processed/stream_a_failure_modes.csv`
- `--sample N`, `--model qwen2.5:14b|gemini|claude`

### Phase 3: Initial clustering
- Embed `failure_category` labels
- Agglomerative clustering → draft category names
- Output: `data/processed/stream_a_failure_clusters.csv`

**This completes Stream A. Review results before proceeding to Stream B.**

---

## Step 4: Extract evaluation framework from Stream B

**File: `src/extract_evaluation_framework.py`**

Run AFTER Stream A taxonomy draft exists, so the extraction is informed by what failure modes were found.

### Phase 1: Candidate identification
Query `stream_b_papers` with prompts derived from Stream A findings:
- For each Stream A failure cluster, construct a query: "accounting standards and professional judgment related to [failure category task type]"
- Plus general framework queries:
  - "materiality threshold determination and its impact on error significance"
  - "professional judgment disagreement rates in auditing"
  - "fair value estimation uncertainty and acceptable variation"
  - "revenue recognition ambiguity under accounting standards"
  - "going concern assessment criteria and auditor disagreement"

### Phase 2: Structured extraction (LLM)
Different prompt than Stream A — extracting framework, not failures:

```
Extract evaluation criteria and baseline information:
1. accounting_principle: which principle/standard applies (e.g., "materiality", "ASC 820 fair value")
2. success_definition: how accounting/finance defines correct output for this task
3. human_baseline: expert disagreement rate or error rate (if stated)
4. significance_measure: how impact of errors is assessed
5. boundary_condition: is there a structural limit to what any automated system can determine? (yes/no + explanation)
6. evidence: direct quote
Return JSON array.
```

- Output: `data/processed/stream_b_evaluation_framework.csv`
- Per-paper JSON: `data/processed/framework_extractions/{lens_id}.json`

---

## Step 5: Assemble final taxonomy

**File: `src/build_taxonomy.py`**

Merge Stream A failure clusters with Stream B evaluation framework:

1. For each failure cluster from Stream A:
   - Match to relevant Stream B evaluation criteria (embedding similarity between cluster description and accounting_principle + success_definition)
   - Determine: is this failure actually significant by accounting standards?
   - Add human baseline comparison
   - Classify as structural (boundary condition) vs solvable (engineering problem)

2. Output final taxonomy:
   - `data/processed/finllm_failure_taxonomy.csv`:
     - category, description, evidence_count, paper_count
     - accounting_significance (material/immaterial/context-dependent)
     - human_baseline (expert disagreement rate if available)
     - structural_or_solvable
     - remedy_pattern
     - type (documented / documented+evaluated / untested-but-predicted)
   - `data/processed/finllm_failure_taxonomy_detailed.csv`: every instance with assignments

---

## File summary

| File | Purpose |
|------|---------|
| `src/parse_papers.py` | PDF → section-tagged chunks |
| `src/build_vectorstore.py` | Chunks → ChromaDB |
| `src/llm_interface.py` | Swappable LLM wrapper |
| `src/extract_failure_modes.py` | Stream A: extract + cluster failure modes |
| `src/extract_evaluation_framework.py` | Stream B: extract accounting evaluation criteria |
| `src/build_taxonomy.py` | Merge both streams into final taxonomy |

## Dependencies

```
pymupdf
chromadb
ollama
```

## Pipeline (sequential)

```bash
# === STREAM A (do first) ===

# 0. Download Stream A papers
python src/download_papers.py --stream A --email <email>

# 1. Parse
python src/parse_papers.py --stream A

# 2. Build vector store
python src/build_vectorstore.py --stream A

# 3. Extract failure modes
python src/extract_failure_modes.py --stream A --model qwen2.5:14b --sample 10  # test
python src/extract_failure_modes.py --stream A --model qwen2.5:14b              # full

# CHECKPOINT: review stream_a_failure_clusters.csv before continuing

# === STREAM B (do second, informed by Stream A results) ===

# 4. Download Stream B papers
python src/download_papers.py --stream B --email <email>

# 5. Parse
python src/parse_papers.py --stream B

# 6. Build vector store
python src/build_vectorstore.py --stream B

# 7. Extract evaluation framework (uses Stream A clusters as input)
python src/extract_evaluation_framework.py --stream B --model qwen2.5:14b --sample 10
python src/extract_evaluation_framework.py --stream B --model qwen2.5:14b

# === ASSEMBLY ===

# 8. Build final taxonomy
python src/build_taxonomy.py --model qwen2.5:14b
```

## Verification checkpoints

1. After step 1: spot-check 5 parsed papers — sections detected?
2. After step 2: test queries against ChromaDB
3. After step 3 `--sample 10`: review extraction quality
4. **CHECKPOINT**: review failure clusters before starting Stream B
5. After step 7 `--sample 10`: review framework extractions
6. After step 8: review taxonomy — significance assessments sensible? Human baselines present?
