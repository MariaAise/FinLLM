"""Extract LLM failure modes from Stream A papers using RAG.

Phase 1: Query ChromaDB with seed queries to find failure-related chunks.
Phase 2: Send candidate chunks to LLM for structured extraction.
Phase 3: Cluster extracted failure modes into categories.

Outputs:
  - data/processed/failure_extractions/{lens_id}.json  (per-paper)
  - data/processed/stream_a_failure_modes.csv           (aggregate)
  - data/processed/stream_a_failure_clusters.csv        (clustered)

v2 changes (2026-03-20):
  - Tighter system prompt with explicit exclusion list
  - Added evidence_type, trigger_or_condition, confidence fields
  - Removed priority gate from Phase 1 (use as ranking signal only)
  - Replaced queries with focused failure-evidence + finance-specific set
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import chromadb
import pandas as pd
from tqdm import tqdm

from llm_interface import get_llm

VECTORSTORE_DIR = "data/vectorstore"
EXTRACTIONS_DIR = "data/processed/failure_extractions"
COLLECTION_NAME = "stream_a_papers"

# Seed queries — focused on explicit failure evidence + finance-specific patterns
SEED_QUERIES = [
    # Explicit failure evidence
    "error analysis financial task model failed",
    "failure cases financial question answering",
    "limitations and failure analysis in financial LLM evaluation",
    "hallucination factual error financial documents",
    "numerical reasoning errors financial QA",
    "wrong extraction from financial tables or filings",
    "performance breakdown showing failure patterns",
    "qualitative examples of incorrect model outputs in finance",
    # Finance-specific failure patterns
    "temporal confusion fiscal year quarter financial reports",
    "entity confusion across companies subsidiaries financial documents",
    "hallucinated numbers or facts in SEC filing analysis",
]

EXTRACTION_SYSTEM = """You are extracting documented failure evidence from research papers on language models in finance.

Your task is to identify failures, errors, weaknesses, or limitations that are supported by evidence in the text. Valid evidence includes:
- quantitative results showing poor or degraded performance
- error case studies or qualitative examples of wrong model outputs
- benchmark comparisons where a model underperforms
- ablation results revealing brittleness or weakness
- author analysis explaining why or where a model fails
- confusion matrices, error breakdowns, or per-category performance gaps

Do not extract:
- generic future work suggestions
- unsupported speculation
- broad statements that the task is "challenging" without specific evidence
- purely methodological descriptions that do not reveal a model weakness
- claims not tied to evidence in the provided text

Prefer concrete, finance-specific failures over generic LLM weaknesses.
When in doubt about whether something counts as failure evidence, extract it with confidence "medium"."""

EXTRACTION_PROMPT = """Below are excerpts from a research paper about language models applied to financial or accounting tasks.

Paper: {title}
DOI: {doi}

--- EXCERPTS ---
{chunks_text}
--- END EXCERPTS ---

Extract only failure evidence that is explicitly documented in these excerpts.

A valid extraction must satisfy at least one of the following:
- reports an incorrect model behavior or a concrete error case study
- reports poor or weaker performance on a specific task, input type, or data condition
- reports hallucination, reasoning error, extraction error, temporal confusion, entity confusion, semantic ambiguity, or similar failure
- shows a confusion between similar concepts, categories, or entities
- reports instability, brittleness, or robustness weakness
- reports a limitation clearly tied to results or examples
- presents a qualitative example of a wrong model output (even a single example counts)

For each extracted item, return:

1. failure_category: short normalized label (e.g., "numerical reasoning error", "table structure misinterpretation", "temporal confusion")
2. description: 1-2 sentences explaining what goes wrong
3. evidence_type: one of ["quantitative_result", "qualitative_example", "author_interpretation", "benchmark_comparison", "ablation_result"]
4. evidence: concise direct quote or specific result from the paper
5. task_type: financial task involved (e.g., "financial QA", "sentiment analysis", "table extraction", "risk assessment")
6. trigger_or_condition: what condition exposed the failure, if stated (e.g., "long documents", "multi-step calculations", "cross-entity comparisons")
7. models_tested: models exhibiting the failure, if stated
8. severity: exact author wording if available, otherwise "not stated"
9. confidence: "high" if explicit and direct, "medium" if clearly supported but indirect

Return a JSON array of objects. If no documented failure evidence is present, return [].

Important rules:
- do not merge distinct failures into one item
- do not infer failures beyond the excerpt
- do not extract generic statements like "the model has limitations" without evidence"""


def get_candidate_papers(collection) -> dict[str, list]:
    """Phase 1: Query vectorstore with seed queries to find failure-related chunks.

    Queries all chunks (no priority gate). Priority used as ranking signal.
    Returns dict mapping paper_id to list of relevant chunks.
    """
    paper_chunks = defaultdict(list)
    seen_chunk_ids = set()

    for query in SEED_QUERIES:
        # Query without priority filter — all chunks are candidates
        results = collection.query(
            query_texts=[query],
            n_results=40,
        )

        for doc, meta, dist, chunk_id in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
        ):
            if chunk_id not in seen_chunk_ids and dist < 0.8:
                seen_chunk_ids.add(chunk_id)
                paper_chunks[meta["paper_id"]].append({
                    "text": doc,
                    "metadata": meta,
                    "distance": dist,
                    "query": query,
                })

    # Also get chunks from error_analysis and limitations sections (any distance)
    results = collection.query(
        query_texts=["error analysis failure limitation"],
        n_results=200,
        where={"section_type": {"$in": ["error_analysis", "limitations"]}},
    )

    for doc, meta, dist, chunk_id in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        results["ids"][0],
    ):
        if chunk_id not in seen_chunk_ids:
            seen_chunk_ids.add(chunk_id)
            paper_chunks[meta["paper_id"]].append({
                "text": doc,
                "metadata": meta,
                "distance": dist,
                "query": "section_type_filter",
            })

    return dict(paper_chunks)


def extract_from_paper(llm, paper_id: str, chunks: list, output_dir: str) -> list[dict]:
    """Phase 2: Extract failure modes from one paper's chunks using LLM."""
    output_path = os.path.join(output_dir, f"{paper_id}.json")

    # Skip if already extracted
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)

    # Sort chunks: prefer high-priority sections, then by distance
    def chunk_sort_key(c):
        priority_boost = 0 if c["metadata"].get("priority") == "high" else 0.1
        return c["distance"] + priority_boost

    chunks_sorted = sorted(chunks, key=chunk_sort_key)[:12]

    # Build context
    title = chunks_sorted[0]["metadata"].get("title", "Unknown")
    doi = chunks_sorted[0]["metadata"].get("doi", "")

    chunks_text = ""
    for chunk in chunks_sorted:
        section = chunk["metadata"].get("section_type", "unknown")
        heading = chunk["metadata"].get("heading", "")
        chunks_text += f"\n[Section: {section} | {heading}]\n{chunk['text']}\n"

    prompt = EXTRACTION_PROMPT.format(
        title=title,
        doi=doi,
        chunks_text=chunks_text,
    )

    # Extract
    failure_modes = llm.extract_json(prompt, system=EXTRACTION_SYSTEM)

    # Add paper metadata to each failure mode
    for fm in failure_modes:
        fm["paper_id"] = paper_id
        fm["doi"] = doi
        fm["paper_title"] = title
        fm["stream"] = chunks_sorted[0]["metadata"].get("stream", "A")

    # Save per-paper extraction
    with open(output_path, "w") as f:
        json.dump(failure_modes, f, indent=2, ensure_ascii=False)

    return failure_modes


def cluster_failure_modes(failure_modes: list[dict], output_path: str):
    """Phase 3: Cluster failure categories using embedding similarity."""
    from sklearn.cluster import AgglomerativeClustering
    from sentence_transformers import SentenceTransformer

    # Get unique category labels
    categories = list({fm["failure_category"] for fm in failure_modes if fm.get("failure_category")})
    if len(categories) < 2:
        print("  Too few categories to cluster")
        return

    # Embed categories
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(categories)

    # Cluster
    n_clusters = min(max(5, len(categories) // 4), 25)
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings)

    # Build cluster mapping
    category_to_cluster = dict(zip(categories, labels))

    # Name clusters by most common category in each
    cluster_names = {}
    for cluster_id in range(n_clusters):
        cluster_cats = [c for c, l in zip(categories, labels) if l == cluster_id]
        # Count how many failure modes each category has
        cat_counts = {}
        for fm in failure_modes:
            cat = fm.get("failure_category", "")
            if cat in cluster_cats:
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
        if cat_counts:
            cluster_names[cluster_id] = max(cat_counts, key=cat_counts.get)
        else:
            cluster_names[cluster_id] = cluster_cats[0] if cluster_cats else f"cluster_{cluster_id}"

    # Add cluster info to failure modes
    rows = []
    for fm in failure_modes:
        cat = fm.get("failure_category", "")
        cluster_id = category_to_cluster.get(cat, -1)
        rows.append({
            **fm,
            "cluster_id": cluster_id,
            "cluster_name": cluster_names.get(cluster_id, ""),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nClustered {len(failure_modes)} failure modes into {n_clusters} clusters")
    print(f"Saved to {output_path}")

    # Print cluster summary
    print("\nCluster summary:")
    for cluster_id in sorted(cluster_names.keys()):
        count = sum(1 for r in rows if r["cluster_id"] == cluster_id)
        papers = len({r["paper_id"] for r in rows if r["cluster_id"] == cluster_id})
        print(f"  [{cluster_id}] {cluster_names[cluster_id]} — {count} instances, {papers} papers")


def main():
    parser = argparse.ArgumentParser(description="Extract failure modes from Stream A papers")
    parser.add_argument("--model", default="qwen2.5:14b-instruct",
                        help="LLM model to use (default: qwen2.5:14b-instruct)")
    parser.add_argument("--sample", type=int, default=0,
                        help="Process only N papers (for testing)")
    parser.add_argument("--skip-clustering", action="store_true",
                        help="Skip the clustering phase")
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    # Connect to vectorstore
    client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        print(f"Error: collection '{COLLECTION_NAME}' not found. Run build_vectorstore.py first.")
        sys.exit(1)

    print(f"Collection '{COLLECTION_NAME}' has {collection.count()} documents")

    # Phase 1: Find candidate papers
    print("\n=== Phase 1: Candidate identification ===")
    paper_chunks = get_candidate_papers(collection)
    print(f"Found {len(paper_chunks)} candidate papers with relevant chunks")

    # Sort by number of relevant chunks (most relevant first)
    papers_sorted = sorted(paper_chunks.items(), key=lambda x: len(x[1]), reverse=True)

    if args.sample > 0:
        papers_sorted = papers_sorted[:args.sample]
        print(f"Sampling {len(papers_sorted)} papers")

    # Phase 2: Extract failure modes
    print(f"\n=== Phase 2: Structured extraction ({args.model}) ===")
    os.makedirs(EXTRACTIONS_DIR, exist_ok=True)

    llm = get_llm(model=args.model, temperature=args.temperature)
    all_failure_modes = []
    json_errors = 0

    for paper_id, chunks in tqdm(papers_sorted, desc="Extracting"):
        try:
            modes = extract_from_paper(llm, paper_id, chunks, EXTRACTIONS_DIR)
            all_failure_modes.extend(modes)
            time.sleep(0.5)
        except Exception as e:
            json_errors += 1
            print(f"\n  Error on {paper_id}: {e}")
            continue

    print(f"\nExtracted {len(all_failure_modes)} failure modes from {len(papers_sorted)} papers")
    if json_errors:
        print(f"  JSON/extraction errors: {json_errors} ({json_errors/len(papers_sorted)*100:.0f}%)")

    # Save aggregate CSV
    if all_failure_modes:
        aggregate_path = "data/processed/stream_a_failure_modes.csv"
        df = pd.DataFrame(all_failure_modes)
        df.to_csv(aggregate_path, index=False)
        print(f"Saved aggregate to {aggregate_path}")

        # Print summary
        print(f"\nTop failure categories:")
        cat_counts = df["failure_category"].value_counts().head(15)
        for cat, count in cat_counts.items():
            print(f"  {cat}: {count}")

        # Phase 3: Cluster
        if not args.skip_clustering and len(all_failure_modes) >= 5:
            print(f"\n=== Phase 3: Clustering ===")
            cluster_path = "data/processed/stream_a_failure_clusters.csv"
            cluster_failure_modes(all_failure_modes, cluster_path)
    else:
        print("No failure modes extracted.")


if __name__ == "__main__":
    main()
