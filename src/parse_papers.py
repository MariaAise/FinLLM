"""Parse downloaded PDFs into section-tagged text chunks.

Extracts text from PDFs using pymupdf, detects section headings, and outputs
one JSONL file with one record per chunk. Each chunk carries metadata (paper_id,
section_type, priority) used downstream by the vector store and extraction pipeline.

Section detection uses regex patterns common in academic papers (numbered or
unnumbered headings). When detection fails, falls back to fixed-size chunking.

Priority tagging:
  high — Results, Experiments, Error Analysis, Limitations, Discussion
  low  — Introduction, Related Work, Methods, Background, Conclusion, Other

High-priority sections are where failure modes concentrate.
"""

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
import pymupdf
from tqdm import tqdm

# Section heading patterns — order matters (first match wins)
SECTION_PATTERNS = [
    # Numbered: "1 Introduction", "3.2 Results", "IV. DISCUSSION"
    (r"^(?:\d+\.?\d*\.?\s+|[IVX]+\.?\s+)", None),
]

# Keywords to classify detected headings into section types
SECTION_KEYWORDS = {
    "abstract": ["abstract"],
    "introduction": ["introduction"],
    "related_work": ["related work", "literature review", "prior work", "background"],
    "methods": ["method", "methodology", "approach", "framework", "model", "system design", "experimental setup", "setup"],
    "results": ["result", "experiment", "evaluation", "finding", "performance", "empirical"],
    "error_analysis": ["error analysis", "failure analysis", "error case", "case study", "qualitative analysis"],
    "discussion": ["discussion", "analysis", "implications"],
    "limitations": ["limitation", "threat", "shortcoming", "future work"],
    "conclusion": ["conclusion", "summary", "concluding"],
}

HIGH_PRIORITY_SECTIONS = {"results", "error_analysis", "discussion", "limitations"}
LOW_PRIORITY_SECTIONS = {"abstract", "introduction", "related_work", "methods", "conclusion", "other"}

MAX_CHUNK_TOKENS = 1500  # approximate — using word count / 0.75
FALLBACK_CHUNK_SIZE = 512
FALLBACK_OVERLAP = 64


def extract_text(pdf_path: str) -> str:
    """Extract full text from PDF."""
    try:
        doc = pymupdf.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        return ""


def classify_heading(heading: str) -> str:
    """Map a heading string to a section type."""
    h = heading.lower().strip()
    # Remove numbering prefix
    h = re.sub(r"^(?:\d+\.?\d*\.?\s*|[ivx]+\.?\s*)", "", h).strip()

    for section_type, keywords in SECTION_KEYWORDS.items():
        for kw in keywords:
            if kw in h:
                return section_type
    return "other"


def is_heading(line: str) -> bool:
    """Heuristic: is this line likely a section heading?"""
    s = line.strip()
    if not s or len(s) > 80 or len(s) < 3:
        return False

    # Reject reference lines: "10. X. D. Yu, et al." or "[12] Smith..."
    if re.match(r"^\d+\.\s+[A-Z]\.\s", s):  # "10. X. D. Yu"
        return False
    if re.match(r"^\[\d+\]", s):  # "[12] Smith"
        return False
    if "et al" in s.lower():
        return False

    # Reject lines with URLs, DOIs, emails
    if any(x in s.lower() for x in ["http", "doi", "@", ".com", ".org"]):
        return False

    # Numbered heading: "1 Introduction", "3.2 Results and Discussion"
    # Must have a known section keyword or be short (< 8 words)
    if re.match(r"^\d+\.?\d*\.?\s+[A-Z]", s):
        words = s.split()
        if len(words) <= 8:
            return True

    # Roman numeral heading: "IV. DISCUSSION"
    if re.match(r"^[IVX]+\.?\s+[A-Z]", s):
        return True

    # All caps short line matching a known keyword
    if s.isupper() and len(s.split()) <= 6:
        for keywords in SECTION_KEYWORDS.values():
            for kw in keywords:
                if kw in s.lower():
                    return True

    # Standalone keyword line: "Abstract", "Limitations", "Discussion"
    # Must be very short (1-4 words) and start with a capital letter
    if len(s.split()) <= 4 and s[0].isupper():
        s_lower = s.lower().rstrip(".:").strip()
        for keywords in SECTION_KEYWORDS.values():
            for kw in keywords:
                if s_lower == kw or s_lower.startswith(kw):
                    return True

    return False


def split_into_sections(text: str) -> list[dict]:
    """Split text into sections based on heading detection.

    Returns list of {section_type, heading, text} dicts.
    """
    lines = text.split("\n")
    sections = []
    current_heading = "preamble"
    current_type = "other"
    current_lines = []

    for line in lines:
        if is_heading(line):
            # Save previous section
            if current_lines:
                content = "\n".join(current_lines).strip()
                if content:
                    sections.append({
                        "section_type": current_type,
                        "heading": current_heading,
                        "text": content,
                    })
            current_heading = line.strip()
            current_type = classify_heading(current_heading)
            current_lines = []
        else:
            current_lines.append(line)

    # Save last section
    if current_lines:
        content = "\n".join(current_lines).strip()
        if content:
            sections.append({
                "section_type": current_type,
                "heading": current_heading,
                "text": content,
            })

    return sections


def approx_tokens(text: str) -> int:
    """Rough token count (words / 0.75)."""
    return int(len(text.split()) / 0.75)


def chunk_text(text: str, max_tokens: int = FALLBACK_CHUNK_SIZE, overlap: int = FALLBACK_OVERLAP) -> list[str]:
    """Split text into fixed-size chunks with overlap (by word count)."""
    words = text.split()
    max_words = int(max_tokens * 0.75)
    overlap_words = int(overlap * 0.75)
    chunks = []
    start = 0

    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap_words
        if start >= len(words):
            break

    return chunks


def process_paper(pdf_path: str, metadata: dict) -> list[dict]:
    """Parse one PDF into chunks with metadata.

    Returns list of chunk dicts ready for JSONL output.
    """
    text = extract_text(pdf_path)
    if not text.strip():
        return []

    sections = split_into_sections(text)

    # Determine parse quality
    section_types_found = {s["section_type"] for s in sections}
    has_key_sections = bool(section_types_found & {"results", "discussion", "limitations", "introduction"})

    if has_key_sections and len(sections) >= 3:
        parse_quality = "good"
    elif len(sections) >= 2:
        parse_quality = "partial"
    else:
        parse_quality = "failed"

    chunks = []
    chunk_index = 0

    if parse_quality == "failed":
        # Fallback: fixed-size chunking
        raw_chunks = chunk_text(text, FALLBACK_CHUNK_SIZE, FALLBACK_OVERLAP)
        for chunk_text_str in raw_chunks:
            if chunk_text_str.strip():
                chunks.append({
                    **metadata,
                    "section_type": "other",
                    "heading": "",
                    "priority": "low",
                    "chunk_index": chunk_index,
                    "text": chunk_text_str.strip(),
                    "parse_quality": parse_quality,
                })
                chunk_index += 1
    else:
        # Section-based chunking
        for section in sections:
            section_type = section["section_type"]
            priority = "high" if section_type in HIGH_PRIORITY_SECTIONS else "low"
            section_text = section["text"]

            # Split large sections
            if approx_tokens(section_text) > MAX_CHUNK_TOKENS:
                sub_chunks = chunk_text(section_text, MAX_CHUNK_TOKENS, FALLBACK_OVERLAP)
            else:
                sub_chunks = [section_text]

            for sub in sub_chunks:
                if sub.strip():
                    chunks.append({
                        **metadata,
                        "section_type": section_type,
                        "heading": section["heading"],
                        "priority": priority,
                        "chunk_index": chunk_index,
                        "text": sub.strip(),
                        "parse_quality": parse_quality,
                    })
                    chunk_index += 1

    # Add total_chunks to each
    for c in chunks:
        c["total_chunks"] = len(chunks)

    return chunks


def main():
    parser = argparse.ArgumentParser(description="Parse PDFs into section-tagged chunks")
    parser.add_argument("--stream", required=True, choices=["A", "B"],
                        help="Stream to parse")
    parser.add_argument("--papers-dir", default=None,
                        help="Override PDF directory")
    parser.add_argument("--metadata-csv", default=None,
                        help="Override metadata CSV path")
    parser.add_argument("--sample", type=int, default=0,
                        help="Process only N papers (for testing)")
    args = parser.parse_args()

    stream = args.stream.lower()
    papers_dir = args.papers_dir or f"data/papers/stream_{stream}"
    metadata_csv = args.metadata_csv or f"data/processed/lit_review_stream_{stream}_filtered.csv"
    output_jsonl = f"data/processed/stream_{stream}_chunks.jsonl"
    manifest_csv = f"data/processed/stream_{stream}_parsed_manifest.csv"

    # Load metadata
    print(f"Loading metadata from {metadata_csv}...")
    df = pd.read_csv(metadata_csv, low_memory=False)
    print(f"  {len(df)} papers in metadata")

    # Build DOI-to-metadata lookup
    # download_papers.py creates filenames from DOIs: / and : replaced with _
    doi_to_meta = {}
    for _, row in df.iterrows():
        doi = row.get("doi", "")
        lens_id = row.get("lens_id", "")
        if pd.notna(doi) and doi:
            filename = doi.replace("/", "_").replace(":", "_") + ".pdf"
            doi_to_meta[filename] = {
                "paper_id": str(lens_id),
                "doi": str(doi),
                "title": str(row.get("title", "")),
                "stream": args.stream,
                "query_block": str(row.get("query_block", "")),
            }

    # Find PDFs
    if not os.path.isdir(papers_dir):
        print(f"Error: papers directory not found: {papers_dir}")
        return

    pdf_files = sorted([f for f in os.listdir(papers_dir) if f.endswith(".pdf")])
    print(f"  {len(pdf_files)} PDFs found in {papers_dir}")

    if args.sample > 0:
        pdf_files = pdf_files[:args.sample]
        print(f"  Sampling {len(pdf_files)} papers")

    # Process
    all_chunks = []
    manifest_rows = []

    for pdf_file in tqdm(pdf_files, desc="Parsing PDFs"):
        pdf_path = os.path.join(papers_dir, pdf_file)

        # Get metadata
        meta = doi_to_meta.get(pdf_file, {
            "paper_id": pdf_file.replace(".pdf", ""),
            "doi": "",
            "title": "",
            "stream": args.stream,
            "query_block": "",
        })

        chunks = process_paper(pdf_path, meta)

        # Manifest entry
        section_types = list({c["section_type"] for c in chunks})
        manifest_rows.append({
            "paper_id": meta["paper_id"],
            "doi": meta["doi"],
            "title": meta["title"],
            "pdf_file": pdf_file,
            "parse_quality": chunks[0]["parse_quality"] if chunks else "failed",
            "num_chunks": len(chunks),
            "sections_found": "; ".join(sorted(section_types)),
            "high_priority_chunks": sum(1 for c in chunks if c["priority"] == "high"),
        })

        all_chunks.extend(chunks)

    # Write chunks JSONL
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    with open(output_jsonl, "w") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(all_chunks)} chunks to {output_jsonl}")

    # Write manifest
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"Wrote manifest ({len(manifest_rows)} papers) to {manifest_csv}")

    # Stats
    print(f"\nParse quality:")
    print(manifest_df["parse_quality"].value_counts().to_string())
    print(f"\nPriority breakdown:")
    high = sum(1 for c in all_chunks if c["priority"] == "high")
    low = sum(1 for c in all_chunks if c["priority"] == "low")
    print(f"  high: {high} chunks")
    print(f"  low:  {low} chunks")


if __name__ == "__main__":
    main()
