# Literature Search Query Documentation

## Project

Taxonomy of LLM failure modes in finance and accounting.

## Search Strategy

Two independent literature streams, kept separate throughout the pipeline:

- **Stream A** — Documented LLM failures in finance (what the literature already reports)
- **Stream B** — Inherent task difficulty in accounting/finance (latent failure surfaces where LLMs will likely fail because even human experts struggle or disagree)

Streams are not merged. Cross-stream comparison is done at the taxonomy level, not the dataset level.

## Source

All queries executed via **Lens.org Scholarly API**. No web-export CSVs used (prior exports from earlier query designs were discarded when queries were redesigned).

---

## Stream A: LLM + Finance Failures

**Date range:** 2023–2026
**Publication types:** journal article; preprint; conference proceedings article; dataset
**Open access filter:** none (all papers retrieved; download step filters to OA only)

### Q1 — LLM failure modes in finance (direct)

**Rationale:** Directly targets papers that document hallucinations, errors, biases, and other failure modes of LLMs when applied to financial or accounting tasks.

```
("large language model" OR LLM OR GPT OR "foundation model" OR "generative AI")
AND (finance OR financial OR accounting OR audit OR "SEC filing" OR "earnings call" OR "10-K" OR GAAP OR IFRS)
AND (hallucination OR "factual error" OR bias OR robustness OR limitation OR failure OR misinterpretation OR "numerical reasoning" OR confabulation OR "error analysis" OR reliability)
```

### Q3 — Named financial benchmarks + LLM

**Rationale:** Captures papers evaluating LLMs on established financial NLP benchmarks (FinQA, FinanceBench, etc.). These often contain detailed error analysis even when the paper's primary framing is evaluation rather than failure.

```
("large language model" OR LLM OR GPT OR "foundation model")
AND (FinQA OR FinanceBench OR "Financial PhraseBank" OR "EDGAR QA" OR ConvFinQA OR TAT-QA OR "earnings call" OR "10-K")
AND (error OR limitation OR benchmark OR evaluation)
```

### Q4 — LLM + finance + evaluation (broad)

**Rationale:** Broader net for LLM evaluation in finance. Overlaps with Q1 but uses different outcome terms (e.g., "failure mode", "benchmark") to catch papers that frame findings differently. Deduplication handles overlap with Q1.

```
("large language model" OR LLM OR GPT OR "foundation model")
AND (finance OR financial OR fintech OR "financial reports" OR "earnings call" OR "SEC filing" OR accounting OR audit)
AND (evaluation OR benchmark OR limitation OR "error analysis" OR robustness OR "failure mode" OR hallucination OR bias)
```

---

## Stream B: Inherent Task Difficulty

**Date range:** 2015–2026 (wider window — this literature predates LLMs)
**Publication types:** journal article; preprint; conference proceedings article; dataset
**Open access filter:** none

### Q2 — Professional judgment and estimation uncertainty

**Rationale:** Identifies tasks in accounting and financial reporting where the correct answer is underdetermined by the data — requiring professional judgment, estimation, or interpretation. These represent latent failure surfaces for LLMs: areas where the model will inherit or amplify human-level ambiguity. Papers in this stream typically do not mention LLMs at all.

```
(accounting OR auditing OR "financial reporting" OR "financial analysis")
AND ("professional judgment" OR "estimation uncertainty" OR "fair value" OR "going concern" OR "materiality" OR "restatement" OR "earnings management" OR "contingent liability" OR "related party" OR "off-balance sheet" OR "revenue recognition")
AND (error OR failure OR disagreement OR inconsistency OR bias OR "judgment variability" OR challenge)
```

---

## Query Numbering

Q1, Q3, Q4 are used (not Q1, Q2, Q3) because the numbering reflects the order of design discussion. Q2 belongs to Stream B. This numbering is preserved in the `query_block` column throughout the pipeline.

## Prefiltering (post-search)

Each stream has its own keyword gates and embedding-similarity anchors applied in `lit_review_classify.py`. See that script for details.
