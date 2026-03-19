"""Prepare a deduplicated, stream-tagged dataset from Lens API search results.

Design decisions:
- Prior web-export CSVs (from external drive) are discarded. All data now comes
  from the Lens API via lit_search_lens.py, which already deduplicates within
  each run and tags rows with query_block.
- This script adds a `stream` column (A or B) and performs a final dedup on
  lens_id as a safety net.
- Cross-stream deduplication is NOT done. Streams A and B serve different
  analytical purposes:
    Stream A = documented LLM failures in finance (Q1, Q3, Q4)
    Stream B = inherent accounting/finance task difficulty (Q2)
  A paper appearing in both streams is meaningful, not a duplicate.

See data/query_documentation.md for full query rationale.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

STREAM_CONFIG = {
    "A": {
        "input": "data/processed/lens_stream_a_list.csv",
        "output": "data/raw/lit_review_stream_a_merged.csv",
        "description": "LLM failures in finance (Q1, Q3, Q4)",
    },
    "B": {
        "input": "data/processed/lens_stream_b_list.csv",
        "output": "data/raw/lit_review_stream_b_merged.csv",
        "description": "Inherent accounting/finance task difficulty (Q2)",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Merge and tag stream dataset")
    parser.add_argument(
        "--stream",
        required=True,
        choices=["A", "B"],
        help="Stream to process (A = LLM failures, B = inherent task difficulty)",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Override input CSV path (default: per-stream config)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override output CSV path (default: per-stream config)",
    )
    args = parser.parse_args()

    config = STREAM_CONFIG[args.stream]
    input_path = args.input or config["input"]
    output_path = args.output or config["output"]

    print(f"Stream {args.stream}: {config['description']}")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")

    # Load
    if not Path(input_path).exists():
        print(f"Error: input file not found: {input_path}")
        print("Run lit_search_lens.py first to generate search results.")
        sys.exit(1)

    df = pd.read_csv(input_path, low_memory=False)
    print(f"  Loaded {len(df)} rows")

    # Verify query_block column exists (set by lit_search_lens.py)
    if "query_block" not in df.columns:
        print("Warning: 'query_block' column not found. Adding placeholder.")
        df["query_block"] = "unknown"

    # Add stream tag
    df["stream"] = args.stream

    # Safety dedup on lens_id (should already be deduped by lit_search_lens.py)
    before = len(df)
    df = df.drop_duplicates(subset="lens_id", keep="first")
    after = len(df)
    if before != after:
        print(f"  Dedup: {before} → {after} ({before - after} duplicates removed)")
    else:
        print(f"  No duplicates found (already clean)")

    # Query block breakdown
    print(f"\n  Query block breakdown:")
    for qb, count in df["query_block"].value_counts().sort_index().items():
        print(f"    {qb}: {count} papers")

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n  Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
