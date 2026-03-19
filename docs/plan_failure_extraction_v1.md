# Plan: FinLLM Failure Mode Taxonomy Pipeline

## Overall goal

Build a taxonomy of LLM failure modes in finance/accounting that captures:
1. **Documented failures** (Stream A) — what the literature already reports
2. **Latent failures** (Stream B) — inherent task difficulty that predicts where LLMs will fail but hasn't been tested yet
3. **The intersection** — where documented failures map to inherent task properties, and where gaps exist (predicted but untested failure modes)

## Architecture

```
Stream A PDFs ──→ parse ──→ ChromaDB ──→ failure extraction ──→ taxonomy
                              ↑                                    ↑
Stream B PDFs ──→ parse ──┘                                       │
                                                                   │
Stream B task properties ──→ map to failure categories ────────────┘
```

The vector store (ChromaDB) serves double duty:
- Targeted retrieval for failure mode extraction (this pipeline)
- Reusable for all future queries during review paper writing

---

## Step 1: Parse PDFs into chunks

**New file: `src/parse_papers.py`**

- Input: PDFs from `data/papers/stream_{a|b}/`
- Extract text using `pymupdf`
- Detect section headings: Abstract, Introduction, Related Work, Method/Methodology, Results/Experiments, Discussion, Limitations, Error Analysis, Conclusion
- Chunk strategy:
  - If sections detected: one chunk per section (split further if section > 1500 tokens)
  - If section detection fails: fixed-size chunks (512 tokens, 64 token overlap)
- Tag each chunk with metadata:
  - `paper_id` (lens_id), `doi`, `title`, `stream`, `query_block`
  - `section_type` (results, limitations, error_analysis, discussion, introduction, methods, other)
  - `chunk_index`, `total_chunks`
  - `priority`: `high` for Results, Error Analysis, Limitations, Discussion; `low` for Introduction, Related Work, Methods
- Output: `data/processed/stream_{a|b}_chunks.jsonl`
- Manifest: `data/processed/stream_{a|b}_parsed_manifest.csv`
- Accept `--stream A|B` flag

---

## Step 2: Build vector store

**New file: `src/build_vectorstore.py`**

- Read `data/processed/stream_{a|b}_chunks.jsonl`
- Embed using `all-MiniLM-L6-v2` via ChromaDB's SentenceTransformer embedding function
- Store in ChromaDB persistent directory: `data/vectorstore/`
- Two collections: `stream_a_papers`, `stream_b_papers`
- All chunk metadata stored and filterable
- Supports:
  - Similarity search with metadata filters (section_type, priority, paper_id)
  - Full-paper retrieval by paper_id
  - Rebuild from scratch if chunking changes
- Accept `--stream A|B` flag

**Dependencies:** `chromadb`

---

## Step 3: Extract failure modes from Stream A

**New file: `src/llm_interface.py`**

Thin swappable wrapper:
- `generate(prompt: str, model: str = "qwen2.5:14b") -> str`
- Default: **Ollama** (local, zero cost)
- Optional: Gemini (`google-genai`), Claude (`anthropic`)
- JSON response parsing with retry
- Rate limiting for cloud backends

**New file: `src/extract_failure_modes.py`**

Two-phase approach using the vector store:

### Phase 1: Identify candidate papers (cheap, via vector store)

Query ChromaDB `stream_a_papers` collection with failure-mode seed queries:

```python
SEED_QUERIES = [
    "error analysis showing where the model failed on financial tasks",
    "numerical reasoning mistakes in financial question answering",
    "hallucination or factual errors in financial document analysis",
    "model limitations on accounting and auditing tasks",
    "performance breakdown by task type showing failure categories",
    "incorrect extraction from financial tables or SEC filings",
    "robustness failures when prompt or input format changes",
    "bias in financial sentiment or risk assessment",
]
```

For each query: retrieve top-k chunks (k=30), filtered to `priority: high`.
Deduplicate by paper_id. Union across all seed queries gives candidate set.

No LLM calls — just vector similarity. Fast and free.

### Phase 2: Extract failure modes (LLM, candidates only)

For each candidate paper:
- Retrieve all high-priority chunks from ChromaDB by paper_id
- Send to LLM with structured extraction prompt:

```
You are analyzing an academic paper about LLMs applied to finance/accounting.
From the following sections, extract every failure mode, error pattern, or
limitation reported. For each, provide:

1. failure_category: short label (e.g., "numerical reasoning error")
2. description: 1-2 sentences describing the failure
3. evidence: direct quote from the text
4. task_type: financial task attempted (e.g., "table QA", "sentiment analysis")
5. models_tested: which model(s) exhibited this failure (if stated)
6. severity: how authors characterize impact (if stated)

Return JSON array. Empty array if no failure modes found.
```

- Save per-paper: `data/processed/failure_extractions/{lens_id}.json`
- Aggregate: `data/processed/stream_a_failure_modes.csv` — one row per failure instance
- Columns: paper_id, doi, title, failure_category, description, evidence, task_type, models_tested, severity
- Stats: papers processed, papers with extractions, total failure instances
- Support `--sample N` and `--model qwen2.5:14b|gemini|claude`

---

## Step 4: Extract task properties from Stream B

**New file: `src/extract_task_properties.py`**

Same infrastructure (llm_interface.py), different prompt. For each Stream B paper:
- Retrieve all high-priority chunks from ChromaDB `stream_b_papers`
- Send to LLM:

```
You are analyzing a paper about professional judgment or estimation
difficulty in accounting/finance. Extract:

1. task_type: the accounting/finance task discussed (e.g., "fair value estimation", "going concern assessment")
2. difficulty_source: what makes this task hard (e.g., "subjective judgment required", "information incomplete", "multiple valid interpretations")
3. human_disagreement: do experts disagree? what is the disagreement rate if stated?
4. evidence: direct quote
5. resolution_mechanism: how is this resolved in practice (e.g., "auditor judgment", "committee review", "regulatory guidance")

Return JSON array.
```

- Output: `data/processed/stream_b_task_properties.csv`
- Per-paper JSON: `data/processed/task_extractions/{lens_id}.json`

---

## Step 5: Build taxonomy

**New file: `src/build_taxonomy.py`**

Three sub-steps:

### 5a: Cluster Stream A failure modes

- Load `stream_a_failure_modes.csv`
- Embed all `failure_category` labels using `all-MiniLM-L6-v2`
- Agglomerative clustering (cosine distance, tunable threshold)
- LLM pass to name each cluster and write a one-paragraph description
- Output: `data/processed/taxonomy_failure_clusters.csv`
  - cluster_id, cluster_name, description, failure_count, paper_count

### 5b: Map Stream B task properties to failure clusters

- Load `stream_b_task_properties.csv`
- For each `difficulty_source`, compute embedding similarity to each failure cluster name+description
- High similarity = this inherent difficulty maps to a known failure mode (validated)
- Low similarity across all clusters = this is a **predicted but untested failure mode** (the novel contribution)
- Output: `data/processed/taxonomy_mapping.csv`
  - task_type, difficulty_source, matched_cluster (or "UNMAPPED"), similarity_score

### 5c: Assemble final taxonomy

- Merge clusters + mappings into final taxonomy table
- Each entry has:
  - **Category** (cluster name)
  - **Type**: documented (Stream A evidence), latent (Stream B prediction), or both
  - **Evidence count** (how many papers report this failure)
  - **Task types affected**
  - **Root cause** (from Stream B: why this task is inherently hard)
  - **Suggested remedy pattern** (from the discussion we had earlier: deterministic+rules, human-in-loop, ensemble, retrieval-augmented)
- Output: `data/processed/finllm_failure_taxonomy.csv`
- Also: `data/processed/finllm_failure_taxonomy_detailed.csv` (every instance with cluster assignment + Stream B mapping)

---

## File summary

| File | Purpose |
|------|---------|
| `src/parse_papers.py` | PDF → section-tagged chunks |
| `src/build_vectorstore.py` | Chunks → ChromaDB (reusable for all future queries) |
| `src/llm_interface.py` | Swappable LLM wrapper (Ollama/Gemini/Claude) |
| `src/extract_failure_modes.py` | Stream A: identify + extract failure modes via RAG |
| `src/extract_task_properties.py` | Stream B: extract inherent task difficulty properties |
| `src/build_taxonomy.py` | Cluster failures, map to task properties, assemble taxonomy |

## Dependencies to add

```
pymupdf
chromadb
ollama
```

**Ollama setup:** `ollama pull qwen2.5:14b-instruct`

## Pipeline

```bash
# 0. Download papers (existing, both streams)
python src/download_papers.py --stream A --email <email>
python src/download_papers.py --stream B --email <email>

# 1. Parse PDFs
python src/parse_papers.py --stream A
python src/parse_papers.py --stream B

# 2. Build vector store
python src/build_vectorstore.py --stream A
python src/build_vectorstore.py --stream B

# 3. Extract failure modes from Stream A
python src/extract_failure_modes.py --stream A --model qwen2.5:14b --sample 10  # test
python src/extract_failure_modes.py --stream A --model qwen2.5:14b              # full

# 4. Extract task properties from Stream B
python src/extract_task_properties.py --stream B --model qwen2.5:14b --sample 10
python src/extract_task_properties.py --stream B --model qwen2.5:14b

# 5. Build taxonomy (combines both streams)
python src/build_taxonomy.py --model qwen2.5:14b
```

## Cost

- Ollama default: **$0**
- Vector store + embeddings: ~2 minutes on CPU for 331 papers
- LLM extraction: depends on model speed, ~1-3 min per paper on qwen2.5:14b

## Verification

1. After step 1: spot-check 5 parsed papers per stream
2. After step 2: test queries against ChromaDB — do relevant chunks come back?
3. After step 3 with `--sample 10`: review extraction JSON quality
4. After step 4 with `--sample 10`: review task property extractions
5. After step 5: review taxonomy — do clusters make sense? Are unmapped entries genuine gaps?
