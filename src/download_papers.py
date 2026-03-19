"""Download PDFs for filtered literature review papers.

Priority order:
1. arXiv papers — direct PDF download
2. Other OA papers — Unpaywall API
3. Paywalled — logged as not downloaded

Supports --stream flag for stream-aware default paths.
Only downloads open-access papers (search retrieves all papers regardless of
access status so we can quantify the paywalled fraction, but this script
skips non-OA papers at download time).
Column names use snake_case (lens_id, doi, source_urls, is_open_access).
"""

import argparse
import re
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}.pdf"
UNPAYWALL_API_URL = "https://api.unpaywall.org/v2/{doi}"
REQUEST_TIMEOUT = 30
RATE_LIMIT_DELAY = 1.0


def extract_arxiv_id(doi: str, source_urls: str) -> str | None:
    """Extract arXiv ID from DOI or Source URLs.

    arXiv DOIs look like: 10.48550/arxiv.2301.12345
    Source URLs may contain: arxiv.org/abs/2301.12345
    """
    if pd.notna(doi):
        m = re.search(r"10\.48550/arxiv\.(.+)", doi, re.IGNORECASE)
        if m:
            return m.group(1)

    if pd.notna(source_urls):
        m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+(?:v\d+)?)", source_urls)
        if m:
            return m.group(1)

    return None


def doi_to_filename(doi: str) -> str:
    """Convert DOI to a safe filename."""
    return doi.replace("/", "_").replace(":", "_") + ".pdf"


def download_file(url: str, dest: Path) -> bool:
    """Download a file from URL to dest. Returns True on success."""
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, stream=True,
                            headers={"User-Agent": "FinLLM-LitReview/1.0"})
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if "pdf" not in content_type and "octet-stream" not in content_type:
            return False

        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        # Verify non-empty
        if dest.stat().st_size < 1000:
            dest.unlink()
            return False

        return True
    except (requests.RequestException, OSError):
        if dest.exists():
            dest.unlink()
        return False


def download_arxiv(arxiv_id: str, dest: Path) -> bool:
    """Download PDF from arXiv."""
    url = ARXIV_PDF_URL.format(arxiv_id=arxiv_id)
    return download_file(url, dest)


def download_via_unpaywall(doi: str, email: str, dest: Path) -> tuple[bool, str]:
    """Query Unpaywall for OA PDF link, then download.

    Returns (success, reason).
    """
    try:
        api_url = UNPAYWALL_API_URL.format(doi=doi)
        resp = requests.get(api_url, params={"email": email},
                            timeout=REQUEST_TIMEOUT,
                            headers={"User-Agent": "FinLLM-LitReview/1.0"})
        if resp.status_code == 404:
            return False, "unpaywall_not_found"
        if resp.status_code == 422:
            return False, "unpaywall_bad_email"
        resp.raise_for_status()

        data = resp.json()
        best = data.get("best_oa_location") or {}
        pdf_url = best.get("url_for_pdf")

        if not pdf_url:
            # Try other OA locations
            for loc in data.get("oa_locations", []):
                pdf_url = loc.get("url_for_pdf")
                if pdf_url:
                    break

        if not pdf_url:
            return False, "no_pdf_url"

        if download_file(pdf_url, dest):
            return True, "unpaywall"
        return False, "download_failed"

    except requests.RequestException as e:
        return False, f"unpaywall_error:{type(e).__name__}"


STREAM_DEFAULTS = {
    "A": {
        "input": "data/processed/lit_review_stream_a_filtered.csv",
        "output_dir": "data/papers/stream_a",
        "manifest": "data/processed/paper_downloads_stream_a.csv",
    },
    "B": {
        "input": "data/processed/lit_review_stream_b_filtered.csv",
        "output_dir": "data/papers/stream_b",
        "manifest": "data/processed/paper_downloads_stream_b.csv",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Download paper PDFs")
    parser.add_argument("--stream", choices=["A", "B"], default=None,
                        help="Stream (sets default input/output paths)")
    parser.add_argument("--input", default=None,
                        help="Input CSV with paper metadata")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save PDFs")
    parser.add_argument("--manifest", default=None,
                        help="Output manifest CSV")
    parser.add_argument("--email", required=True,
                        help="Email for Unpaywall API (required)")
    parser.add_argument("--sample", type=int, default=0,
                        help="Download only N papers (for testing)")
    args = parser.parse_args()

    # Resolve defaults from stream if provided
    defaults = STREAM_DEFAULTS.get(args.stream, {})
    input_path = args.input or defaults.get("input", "data/processed/lit_review_filtered.csv")
    output_dir_path = args.output_dir or defaults.get("output_dir", "data/papers")
    manifest_path = args.manifest or defaults.get("manifest", "data/processed/paper_downloads.csv")

    df = pd.read_csv(input_path)
    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.sample > 0:
        df = df.head(args.sample)

    print(f"Processing {len(df)} papers from {input_path}...")

    results = []
    stats = {"arxiv": 0, "unpaywall": 0, "skipped_existing": 0, "failed": 0}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading"):
        doi = row.get("doi", "")
        source_urls = row.get("source_urls", "")

        # Determine destination filename
        if pd.notna(doi) and doi:
            filename = doi_to_filename(doi)
        else:
            filename = str(row.get("lens_id", "unknown")) + ".pdf"

        dest = output_dir / filename

        # Resume support: skip if already downloaded
        if dest.exists() and dest.stat().st_size > 1000:
            results.append({"pdf_path": str(dest), "download_status": "skipped_existing"})
            stats["skipped_existing"] += 1
            continue

        downloaded = False
        status = "failed"
        reason = ""

        # 1. Try arXiv
        arxiv_id = extract_arxiv_id(doi, source_urls)
        if arxiv_id:
            if download_arxiv(arxiv_id, dest):
                downloaded = True
                status = "arxiv"
                stats["arxiv"] += 1

        # 2. Try Unpaywall (for OA papers without arXiv)
        if not downloaded and pd.notna(doi) and doi:
            time.sleep(RATE_LIMIT_DELAY)
            success, reason = download_via_unpaywall(doi, args.email, dest)
            if success:
                downloaded = True
                status = "unpaywall"
                stats["unpaywall"] += 1

        if not downloaded:
            status = f"failed:{reason}" if reason else "failed:no_source"
            stats["failed"] += 1

        if downloaded:
            time.sleep(RATE_LIMIT_DELAY)

        results.append({"pdf_path": str(dest) if downloaded else "", "download_status": status})

    # Build manifest
    result_df = pd.DataFrame(results)
    manifest = pd.concat([df.reset_index(drop=True), result_df], axis=1)
    manifest.to_csv(manifest_path, index=False)

    print(f"\nDone! Stats:")
    print(f"  arXiv:           {stats['arxiv']}")
    print(f"  Unpaywall:       {stats['unpaywall']}")
    print(f"  Skipped (exist): {stats['skipped_existing']}")
    print(f"  Failed:          {stats['failed']}")
    print(f"  Manifest saved:  {manifest_path}")


if __name__ == "__main__":
    main()
