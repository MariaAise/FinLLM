"""Build ChromaDB vector store from parsed paper chunks.

Embeds chunks using all-MiniLM-L6-v2 and stores them in a persistent ChromaDB
collection. The vector store is reusable for:
- Failure mode extraction (targeted seed queries)
- Ad-hoc queries during paper writing
- Cross-paper analysis

Each chunk's metadata (paper_id, section_type, priority, stream, query_block)
is stored and filterable.
"""

import argparse
import json
import os
import sys

import chromadb
from tqdm import tqdm

VECTORSTORE_DIR = "data/vectorstore"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def main():
    parser = argparse.ArgumentParser(description="Build ChromaDB vector store from chunks")
    parser.add_argument("--stream", required=True, choices=["A", "B"],
                        help="Stream to index")
    parser.add_argument("--rebuild", action="store_true",
                        help="Delete existing collection and rebuild from scratch")
    args = parser.parse_args()

    stream = args.stream.lower()
    chunks_path = f"data/processed/stream_{stream}_chunks.jsonl"
    collection_name = f"stream_{stream}_papers"

    if not os.path.exists(chunks_path):
        print(f"Error: chunks file not found: {chunks_path}")
        print("Run parse_papers.py first.")
        sys.exit(1)

    # Load chunks
    print(f"Loading chunks from {chunks_path}...")
    chunks = []
    with open(chunks_path) as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"  {len(chunks)} chunks loaded")

    # Init ChromaDB
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=VECTORSTORE_DIR)

    if args.rebuild:
        try:
            client.delete_collection(collection_name)
            print(f"  Deleted existing collection '{collection_name}'")
        except (ValueError, chromadb.errors.NotFoundError):
            pass

    # Create collection with embedding function
    # ChromaDB's default embedding function uses all-MiniLM-L6-v2
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    existing = collection.count()
    if existing > 0 and not args.rebuild:
        print(f"  Collection already has {existing} documents. Use --rebuild to recreate.")
        return

    # Add chunks in batches
    batch_size = 200
    print(f"Embedding and indexing {len(chunks)} chunks...")

    for i in tqdm(range(0, len(chunks), batch_size), desc="Indexing"):
        batch = chunks[i:i + batch_size]

        ids = [f"{c['paper_id']}_{c['chunk_index']}" for c in batch]
        documents = [c["text"] for c in batch]
        metadatas = [
            {
                "paper_id": c["paper_id"],
                "doi": c.get("doi", ""),
                "title": c.get("title", "")[:200],  # ChromaDB metadata size limit
                "stream": c.get("stream", ""),
                "query_block": c.get("query_block", ""),
                "section_type": c.get("section_type", ""),
                "heading": c.get("heading", "")[:100],
                "priority": c.get("priority", ""),
                "chunk_index": c.get("chunk_index", 0),
                "total_chunks": c.get("total_chunks", 0),
                "parse_quality": c.get("parse_quality", ""),
            }
            for c in batch
        ]

        collection.add(ids=ids, documents=documents, metadatas=metadatas)

    final_count = collection.count()
    print(f"\nDone. Collection '{collection_name}' has {final_count} documents.")

    # Quick test query
    print("\nTest query: 'error analysis model failure financial'")
    results = collection.query(
        query_texts=["error analysis model failure financial"],
        n_results=3,
        where={"priority": "high"},
    )
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        print(f"  [{dist:.3f}] [{meta['section_type']}] {meta['title'][:60]}")
        print(f"           {doc[:100]}...")


if __name__ == "__main__":
    main()
