"""Score and filter papers by keyword gates and embedding similarity.

Design decisions:
- Two streams with different prefiltering logic:

  Stream A (LLM failures in finance):
    Gates: LLM_GATE AND FINANCE_GATE — both must match in title+abstract.
    Anchors: 3 phrases focused on LLM evaluation, failure modes, and limitations.
    Threshold: 0.75 (tight — LLM papers use consistent vocabulary).

  Stream B (Inherent accounting/finance task difficulty):
    Gates: ACCOUNTING_GATE only — no LLM gate, because these papers predate or
           are independent of LLM literature.
    Anchors: 3 phrases focused on professional judgment, estimation uncertainty,
             and accounting standards ambiguity.
    Threshold: 0.65 (looser — accounting abstracts vary more in phrasing).

- Gates differ per stream because Stream B papers will never mention LLMs.
  Applying an LLM gate to Stream B would drop the entire dataset.
- Thresholds differ because accounting literature uses more varied language
  than ML/NLP papers, so embedding similarity scores are naturally lower.

See data/query_documentation.md for full query and stream rationale.

Pipelines to run:

# Stream A
python src/lit_search_lens.py --config data/input_stream_a.csv
python src/lit_review_merge.py --stream A
python src/lit_review_classify.py --stream A

# Stream B
python src/lit_search_lens.py --config data/input_stream_b.csv
python src/lit_review_merge.py --stream B
python src/lit_review_classify.py --stream B

# Downloads (after reviewing filtered results)
python src/download_papers.py --stream A --email <your-email>
python src/download_papers.py --stream B --email <your-email>

"""

import argparse
import re
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------------------------------------
# Stream A: LLM failures in finance
# ---------------------------------------------------------------------------

LLM_GATE = re.compile(
    r"\b(?:large language model|llms?|gpt[-\s]?\d*|language model|transformers?"
    r"|bert|finbert|bloomberggpt|fingpt|llama|mistral|qwen"
    r"|generative ai|foundation model|chatgpt|gemini|claude"
    r"|copilot)\b",
    re.IGNORECASE,
)

FINANCE_GATE = re.compile(
    r"\b(?:financ\w*|fintech|sec filings?|10-k|annual reports?"
    r"|earnings calls?|financial reports?|market|trading|portfolio"
    r"|credit|bank\w*|risk|fraud|accounting|audit\w*"
    r"|gaap|ifrs|revenue recognition|fair value"
    r"|materiality|restatement|going concern)\b",
    re.IGNORECASE,
)

ANCHORS_A = [
    "evaluation and benchmarking of large language models on financial question answering and numerical reasoning",
    "hallucination, factual errors, and failure modes of language models in financial document analysis",
    "robustness, bias, and limitations of LLMs applied to accounting, auditing, and SEC filings",
]

THRESHOLD_A = 0.50

# ---------------------------------------------------------------------------
# Stream B: Inherent accounting/finance task difficulty
# ---------------------------------------------------------------------------

ACCOUNTING_GATE = re.compile(
    r"\b(?:accounting|audit\w*|financial reporting|financial statement"
    r"|gaap|ifrs|revenue recognition|fair value|materiality"
    r"|restatement|going concern|contingent|related.party"
    r"|impairment|goodwill|lease accounting|tax provision"
    r"|internal control|sox|pcaob)\b",
    re.IGNORECASE,
)

ANCHORS_B = [
    "professional judgment and estimation uncertainty in financial reporting and auditing",
    "disagreement, error, and inconsistency in accounting estimates and materiality assessment",
    "ambiguity and complexity in applying accounting standards to revenue recognition, fair value, and contingencies",
]

THRESHOLD_B = 0.30

# ---------------------------------------------------------------------------
# Stream configs
# ---------------------------------------------------------------------------

STREAM_CONFIG = {
    "A": {
        "input": "data/raw/lit_review_stream_a_merged.csv",
        "output_scored": "data/processed/lit_review_stream_a_similarity.csv",
        "output_filtered": "data/processed/lit_review_stream_a_filtered.csv",
        "gates": [LLM_GATE, FINANCE_GATE],
        "gate_names": ["LLM_GATE", "FINANCE_GATE"],
        "anchors": ANCHORS_A,
        "threshold": THRESHOLD_A,
    },
    "B": {
        "input": "data/raw/lit_review_stream_b_merged.csv",
        "output_scored": "data/processed/lit_review_stream_b_similarity.csv",
        "output_filtered": "data/processed/lit_review_stream_b_filtered.csv",
        "gates": [ACCOUNTING_GATE],
        "gate_names": ["ACCOUNTING_GATE"],
        "anchors": ANCHORS_B,
        "threshold": THRESHOLD_B,
    },
}

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    parser = argparse.ArgumentParser(description="Score and filter papers by stream")
    parser.add_argument(
        "--stream",
        required=True,
        choices=["A", "B"],
        help="Stream to classify (A = LLM failures, B = inherent task difficulty)",
    )
    parser.add_argument("--input", default=None, help="Override input CSV path")
    parser.add_argument("--sample", type=int, default=None, help="Run on N random records only")
    args = parser.parse_args()

    config = STREAM_CONFIG[args.stream]
    input_path = args.input or config["input"]

    print(f"Stream {args.stream}")
    print(f"  Input: {input_path}")
    print(f"  Gates: {config['gate_names']}")
    print(f"  Anchors: {len(config['anchors'])}")
    print(f"  Threshold: {config['threshold']}")

    # Load data
    print(f"\nLoading {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  {len(df)} papers loaded")

    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=42)
        print(f"  Sampled {len(df)} papers")

    # Keyword gate filtering
    df = df.copy()
    combined = df["title"].fillna("") + " " + df["abstract"].fillna("")

    for gate, gate_name in zip(config["gates"], config["gate_names"]):
        match = combined.str.contains(gate, regex=True)
        before = len(df)
        df = df[match]
        combined = combined[match]
        print(f"  {gate_name}: {before} → {len(df)} papers")

    if len(df) == 0:
        print("No papers passed keyword gates. Check your gates or input data.")
        return

    # Combine title + abstract for embedding
    texts = (df["title"].fillna("") + " " + df["abstract"].fillna("")).tolist()

    # Encode
    print(f"\nLoading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding paper texts...")
    paper_embs = model.encode(texts, show_progress_bar=True, convert_to_tensor=True)

    anchors = config["anchors"]
    anchor_embs = model.encode(anchors, convert_to_tensor=True)

    # Cosine similarity
    sims = util.cos_sim(paper_embs, anchor_embs)
    for i in range(len(anchors)):
        df[f"sim{i+1}"] = sims[:, i].cpu().numpy()
    df["score"] = df[[f"sim{i+1}" for i in range(len(anchors))]].max(axis=1)

    # Save all scored papers
    output_scored = Path(config["output_scored"])
    output_scored.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_scored, index=False)
    print(f"\nSaved {len(df)} scored papers to {output_scored}")

    # Filter by threshold
    threshold = config["threshold"]
    filtered = df[df["score"] >= threshold]
    output_filtered = Path(config["output_filtered"])
    filtered.to_csv(output_filtered, index=False)
    print(f"Saved {len(filtered)} papers with score >= {threshold} to {output_filtered}")

    # Score distribution
    print(f"\nScore distribution:")
    print(df["score"].describe().to_string())

    # Histogram buckets
    print(f"\nScore buckets:")
    for lo, hi in [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]:
        count = len(df[(df["score"] >= lo) & (df["score"] < hi)])
        marker = " ← threshold" if lo <= threshold < hi else ""
        print(f"  [{lo:.1f}, {hi:.1f}): {count}{marker}")

    # Top papers
    sim_cols = [f"sim{i+1}" for i in range(len(anchors))]
    print(f"\nTop 5 papers by score:")
    top = df.nlargest(5, "score")[["title"] + sim_cols + ["score"]]
    for _, row in top.iterrows():
        print(f"  [{row['score']:.3f}] {row['title'][:80]}")


if __name__ == "__main__":
    main()
