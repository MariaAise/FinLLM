"""Extract LLM failure modes from Stream A papers using RAG.

Phase 1: Query ChromaDB with seed queries to find failure-related chunks.
Phase 2: Send candidate chunks to LLM for structured extraction.
Phase 3: Cluster extracted failure modes into categories.

Outputs:
  - data/processed/failure_extractions/{lens_id}.json  (per-paper)
  - data/processed/stream_a_failure_modes.csv           (aggregate)
  - data/processed/stream_a_failure_clusters.csv        (clustered)
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

# Seed queries targeting failure content in financial LLM papers
SEED_QUERIES = [
    "error analysis showing where the model failed on financial tasks",
    "numerical reasoning mistakes in financial question answering",
    "hallucination or factual errors in financial document analysis",
    "model limitations on accounting and auditing tasks",
    "performance breakdown by task type showing failure categories",
    "incorrect extraction from financial tables or SEC filings",
    "robustness failures when prompt or input format changes",
    "bias in financial sentiment or risk assessment",
    "wrong predictions on financial text classification",
    "model struggles with multi-step financial reasoning",
]

EXTRACTION_SYSTEM = """You are an expert at analyzing NLP/AI research papers about financial applications.
Your task is to extract every failure mode, error pattern, or limitation reported in the provided text.
Be specific and evidence-based. Only extract failures that are explicitly documented with evidence."""

EXTRACTION_PROMPT = """Below are excerpts from a research paper about LLMs applied to financial/accounting tasks.

Paper: {title}
DOI: {doi}

--- EXCERPTS ---
{chunks_text}
--- END EXCERPTS ---

Extract every failure mode, error pattern, or limitation reported in these excerpts.
For each one, provide:

1. failure_category: a short descriptive label (e.g., "numerical reasoning error", "table structure misinterpretation", "temporal confusion")
2. description: 1-2 sentences explaining what goes wrong
3. evidence: a direct quote or specific result from the paper (keep it concise)
4. task_type: what financial task was being attempted (e.g., "financial QA", "sentiment analysis", "table extraction", "risk assessment")
5. models_tested: which model(s) exhibited this failure (if stated)
6. severity: how the authors characterize the severity (e.g., "major", "minor", "systematic", or quote their characterization)

Return a JSON array of objects. If no failure modes are found, return an empty array [].
Only include failures with clear evidence — do not speculate or infer failures not documented in the text."""


def get_candidate_papers(collection) -> dict[str, list]:
    """Phase 1: Query vectorstore with seed queries to find failure-related chunks.

    Returns dict mapping paper_id to list of relevant chunks.
    """
    paper_chunks = defaultdict(list)
    seen_chunk_ids = set()

    for query in SEED_QUERIES:
        results = collection.query(
            query_texts=[query],
            n_results=30,
            where={"priority": "high"},
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

    # Also get high-priority chunks from all papers (error_analysis, limitations sections)
    results = collection.query(
        query_texts=["error analysis failure limitation"],
        n_results=100,
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

    # Sort chunks by relevance (distance), take top chunks
    chunks_sorted = sorted(chunks, key=lambda c: c["distance"])[:10]

    # Build context
    title = chunks_sorted[0]["metadata"].get("title", "Unknown")
    doi = chunks_sorted[0]["metadata"].get("doi", "")

    chunks_text = ""
    for i, chunk in enumerate(chunks_sorted):
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

    for paper_id, chunks in tqdm(papers_sorted, desc="Extracting"):
        try:
            modes = extract_from_paper(llm, paper_id, chunks, EXTRACTIONS_DIR)
            all_failure_modes.extend(modes)
            # Brief pause to avoid overwhelming Ollama
            time.sleep(0.5)
        except Exception as e:
            print(f"\n  Error on {paper_id}: {e}")
            continue

    print(f"\nExtracted {len(all_failure_modes)} failure modes from {len(papers_sorted)} papers")

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
