"""Compare extraction quality across LLM backends on the same papers.

Runs the same extraction prompt on a fixed set of test papers using
different models and saves results for side-by-side comparison.
"""

import json
import os
import sys
import time

import chromadb
from dotenv import load_dotenv

load_dotenv()

from llm_interface import get_llm
from extract_failure_modes import (
    EXTRACTION_SYSTEM,
    EXTRACTION_PROMPT,
    SEED_QUERIES,
    VECTORSTORE_DIR,
    COLLECTION_NAME,
)

# Papers to test — chosen to cover different difficulty levels
TEST_PAPERS = {
    # Empty in v2 — should have extracted (false negative)
    "074-042-452-083-012": "FinTagging (error case in noisy chunk)",
    # Empty in v2 — correct reject
    "107-232-497-691-567": "FinDABench (generic limitations only)",
    # Minimal extraction (2 duplicates) — under-extracted
    "167-398-185-509-339": "FinanceReasoning (has 4-type error taxonomy)",
    # Max extraction (5 same-category) — under-diverse
    "196-068-497-575-765": "FinMaster (has multi-type error taxonomy)",
}

MODELS = [
    ("qwen2.5:14b-instruct", "Qwen 2.5 14B (Ollama, local)"),
    ("gemini-2.0-flash", "Gemini 2.0 Flash (API, free tier)"),
]

OUTPUT_DIR = "data/processed/model_comparison"


def get_chunks_for_paper(collection, paper_id: str) -> list[dict]:
    """Retrieve relevant chunks for a paper (same logic as extract_failure_modes)."""
    from collections import defaultdict

    paper_chunks = []
    seen = set()

    for query in SEED_QUERIES:
        results = collection.query(query_texts=[query], n_results=40)
        for doc, meta, dist, cid in zip(
            results["documents"][0], results["metadatas"][0],
            results["distances"][0], results["ids"][0],
        ):
            if meta["paper_id"] == paper_id and cid not in seen and dist < 0.8:
                seen.add(cid)
                paper_chunks.append({"text": doc, "metadata": meta, "distance": dist})

    # Section filter
    results = collection.query(
        query_texts=["error analysis failure limitation"],
        n_results=200,
        where={"section_type": {"$in": ["error_analysis", "limitations"]}},
    )
    for doc, meta, dist, cid in zip(
        results["documents"][0], results["metadatas"][0],
        results["distances"][0], results["ids"][0],
    ):
        if meta["paper_id"] == paper_id and cid not in seen:
            seen.add(cid)
            paper_chunks.append({"text": doc, "metadata": meta, "distance": dist})

    return paper_chunks


def run_extraction(llm, chunks: list, paper_id: str) -> tuple[list[dict], float]:
    """Run extraction and return (results, elapsed_seconds)."""
    def chunk_sort_key(c):
        priority_boost = 0 if c["metadata"].get("priority") == "high" else 0.1
        return c["distance"] + priority_boost

    chunks_sorted = sorted(chunks, key=chunk_sort_key)[:12]

    title = chunks_sorted[0]["metadata"].get("title", "Unknown")
    doi = chunks_sorted[0]["metadata"].get("doi", "")

    chunks_text = ""
    for chunk in chunks_sorted:
        section = chunk["metadata"].get("section_type", "unknown")
        heading = chunk["metadata"].get("heading", "")
        chunks_text += f"\n[Section: {section} | {heading}]\n{chunk['text']}\n"

    prompt = EXTRACTION_PROMPT.format(title=title, doi=doi, chunks_text=chunks_text)

    start = time.time()
    failure_modes = llm.extract_json(prompt, system=EXTRACTION_SYSTEM)
    elapsed = time.time() - start

    for fm in failure_modes:
        fm["paper_id"] = paper_id
        fm["doi"] = doi
        fm["paper_title"] = title

    return failure_modes, elapsed


def main():
    client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = {}

    for model_id, model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        try:
            llm = get_llm(model=model_id, temperature=0.1)
        except Exception as e:
            print(f"  Failed to initialize: {e}")
            continue

        model_results = {}

        for paper_id, description in TEST_PAPERS.items():
            print(f"\n  {description}...")
            chunks = get_chunks_for_paper(collection, paper_id)

            if not chunks:
                print(f"    No chunks found")
                model_results[paper_id] = {"modes": [], "time": 0, "error": "no chunks"}
                continue

            try:
                modes, elapsed = run_extraction(llm, chunks, paper_id)
                model_results[paper_id] = {
                    "modes": modes,
                    "time": round(elapsed, 1),
                    "count": len(modes),
                }
                print(f"    {len(modes)} modes in {elapsed:.1f}s")
                for fm in modes:
                    cat = fm.get("failure_category", "?")
                    conf = fm.get("confidence", "?")
                    etype = fm.get("evidence_type", "?")
                    print(f"      [{conf}] [{etype}] {cat}")
            except Exception as e:
                print(f"    Error: {e}")
                model_results[paper_id] = {"modes": [], "time": 0, "error": str(e)}

            time.sleep(1)

        all_results[model_id] = model_results

        # Save per-model results
        output_path = os.path.join(OUTPUT_DIR, f"{model_id.replace(':', '_').replace('.', '_')}.json")
        with open(output_path, "w") as f:
            json.dump(model_results, f, indent=2, ensure_ascii=False)

    # Print comparison summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")

    for paper_id, description in TEST_PAPERS.items():
        print(f"\n{description}")
        for model_id, model_name in MODELS:
            if model_id in all_results:
                r = all_results[model_id].get(paper_id, {})
                count = r.get("count", 0)
                elapsed = r.get("time", 0)
                categories = set()
                for m in r.get("modes", []):
                    categories.add(m.get("failure_category", ""))
                cats_str = ", ".join(sorted(categories)) if categories else "none"
                print(f"  {model_name}: {count} modes ({elapsed}s) — {cats_str}")

    # Save full comparison
    with open(os.path.join(OUTPUT_DIR, "comparison_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
