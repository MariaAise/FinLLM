"""Search Lens.org scholarly API for literature using a multi-query config CSV.

Design decisions:
- Each config CSV contains one or more query blocks (rows), each with its own
  keywords, date range, and publication type filters.
- Query blocks are executed sequentially against the API. Each result row is
  tagged with a `query_block` column (e.g. Q1, Q3, Q4) so downstream analysis
  can trace which query surfaced each paper.
- Results are deduplicated on `lens_id` within a single run. The first
  occurrence is kept, preserving its `query_block` tag.
- No open_access filter is applied at search time — all papers are retrieved so
  we can quantify the paywalled fraction. The download step filters to OA only.
- Cross-stream deduplication is NOT done here. Streams A and B serve different
  analytical purposes (documented LLM failures vs latent accounting task
  difficulty) and are kept separate.

See data/query_documentation.md for full query rationale.
"""

import argparse
import os
import sys
import time

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

API_URL = "https://api.lens.org/scholarly/search"

VALID_PUB_TYPES = {
    "journal article",
    "conference proceedings",
    "conference proceedings article",
    "preprint",
    "book",
    "book chapter",
    "dataset",
    "dissertation",
    "report",
    "working paper",
    "review",
    "component",
    "reference entry",
    "libguide",
    "other",
    "unknown",
}


def load_config(path: str) -> list[dict]:
    """Read multi-query config CSV. Each row is one query block.

    Expected columns: query_id, keywords, year_start, year_end, publication_type
    Optional columns: open_access
    """
    df = pd.read_csv(path, dtype=str).fillna("")
    required_cols = {"query_id", "keywords"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Config CSV missing required columns: {missing}")
        sys.exit(1)

    query_blocks = []
    errors = []

    for idx, row in df.iterrows():
        query_id = row["query_id"].strip()
        if not query_id:
            errors.append(f"Row {idx}: 'query_id' is required.")
            continue

        params = {"query_id": query_id}

        # keywords — required
        keywords = row["keywords"].strip()
        if not keywords:
            errors.append(f"Row {idx} ({query_id}): 'keywords' is required.")
            continue

        # Check balanced parentheses
        depth = 0
        for ch in keywords:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth < 0:
                break
        if depth != 0:
            errors.append(f"Row {idx} ({query_id}): unbalanced parentheses in keywords.")
            continue
        params["keywords"] = keywords

        # year_start — optional
        year_start_str = row.get("year_start", "").strip()
        if year_start_str:
            if not year_start_str.isdigit() or len(year_start_str) != 4:
                errors.append(f"Row {idx} ({query_id}): 'year_start' must be 4-digit integer, got: {year_start_str}")
            else:
                year_start = int(year_start_str)
                if not (1900 <= year_start <= 2030):
                    errors.append(f"Row {idx} ({query_id}): 'year_start' out of range: {year_start}")
                else:
                    params["year_start"] = year_start

        # year_end — optional
        year_end_str = row.get("year_end", "").strip()
        if year_end_str:
            if not year_end_str.isdigit() or len(year_end_str) != 4:
                errors.append(f"Row {idx} ({query_id}): 'year_end' must be 4-digit integer, got: {year_end_str}")
            else:
                year_end = int(year_end_str)
                if not (1900 <= year_end <= 2030):
                    errors.append(f"Row {idx} ({query_id}): 'year_end' out of range: {year_end}")
                else:
                    params["year_end"] = year_end

        # Cross-validate year range
        if "year_start" in params and "year_end" in params:
            if params["year_end"] < params["year_start"]:
                errors.append(f"Row {idx} ({query_id}): year_end < year_start.")

        # publication_type — optional, semicolon-separated
        pub_type_str = row.get("publication_type", "").strip()
        if pub_type_str:
            pub_types = [t.strip().lower() for t in pub_type_str.split(";") if t.strip()]
            invalid = [t for t in pub_types if t not in VALID_PUB_TYPES]
            if invalid:
                errors.append(f"Row {idx} ({query_id}): invalid publication_type(s): {invalid}")
            else:
                params["publication_type"] = pub_types

        # open_access — optional
        oa_str = row.get("open_access", "").strip().lower()
        if oa_str:
            if oa_str not in ("true", "false"):
                errors.append(f"Row {idx} ({query_id}): 'open_access' must be true/false, got: {oa_str}")
            else:
                params["open_access"] = oa_str == "true"

        query_blocks.append(params)

    if errors:
        print("Config validation errors:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    return query_blocks


def build_query(params: dict) -> dict:
    """Construct the Elasticsearch-style API request body."""
    query = {"bool": {"must": [], "filter": []}}

    # Keywords → query_string on title + abstract
    query["bool"]["must"].append(
        {
            "query_string": {
                "query": params["keywords"],
                "fields": ["title", "abstract"],
                "default_operator": "AND",
            }
        }
    )

    # Year range → range filter
    if "year_start" in params or "year_end" in params:
        range_filter = {}
        if "year_start" in params:
            range_filter["gte"] = params["year_start"]
        if "year_end" in params:
            range_filter["lte"] = params["year_end"]
        query["bool"]["filter"].append({"range": {"year_published": range_filter}})

    # Publication type → terms filter
    if "publication_type" in params:
        query["bool"]["filter"].append(
            {"terms": {"publication_type": params["publication_type"]}}
        )

    # Open access → term filter
    if "open_access" in params:
        query["bool"]["filter"].append(
            {"term": {"is_open_access": params["open_access"]}}
        )

    return {"query": query}


def search_lens(query_body: dict, token: str, query_id: str = "") -> list[dict]:
    """Execute search with cursor-based pagination, return all results."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    all_results = []
    scroll_id = None
    request_body = {**query_body, "size": 1000, "scroll": "1m"}

    backoff = 8
    max_retries = 5
    desc = f"Fetching {query_id}" if query_id else "Fetching results"
    pbar = None

    while True:
        if scroll_id:
            payload = {"scroll_id": scroll_id, "scroll": "1m"}
        else:
            payload = request_body

        retries = 0
        while retries < max_retries:
            resp = requests.post(API_URL, json=payload, headers=headers)

            if resp.status_code == 429:
                retries += 1
                wait = min(backoff * (2 ** (retries - 1)), 60)
                remaining = resp.headers.get("x-rate-limit-remaining-request-per-minute", "?")
                print(f"Rate limited (remaining: {remaining}). Retrying in {wait}s... ({retries}/{max_retries})")
                time.sleep(wait)
                continue

            if resp.status_code == 204:
                if pbar:
                    pbar.close()
                return all_results

            if resp.status_code not in (200,):
                print(f"API error {resp.status_code}: {resp.text[:500]}")
                sys.exit(1)

            break
        else:
            print(f"Max retries ({max_retries}) exceeded due to rate limiting.")
            sys.exit(1)

        data = resp.json()
        total = data.get("total", 0)
        results = data.get("data", [])

        if not results:
            break

        all_results.extend(results)

        if pbar is None and total > 0:
            pbar = tqdm(total=total, initial=len(all_results), desc=desc)
        elif pbar is not None:
            pbar.update(len(results))

        scroll_id = data.get("scroll_id")
        if not scroll_id or len(all_results) >= total:
            break

    if pbar:
        pbar.close()

    print(f"  {query_id}: fetched {len(all_results)} / {total} results.")
    return all_results


def extract_record(item: dict) -> dict:
    """Flatten one API result into a CSV row."""
    # Authors
    authors_raw = item.get("authors") or []
    author_names = []
    for a in authors_raw:
        first = a.get("first_name") or ""
        last = a.get("last_name") or ""
        name = f"{first} {last}".strip()
        if name:
            author_names.append(name)

    # External IDs
    ext_ids = item.get("external_ids") or []
    ext_id_strs = [f"{e.get('type', '')}:{e.get('value', '')}" for e in ext_ids]

    # Source URLs
    source_urls = item.get("source_urls") or []
    url_strs = [u.get("url", "") for u in source_urls if u.get("url")]

    # Open access
    oa = item.get("open_access") or {}

    # Keywords
    keywords = item.get("keywords") or []

    # Fields of study
    fields = item.get("fields_of_study") or []

    return {
        "lens_id": item.get("lens_id", ""),
        "title": item.get("title", ""),
        "date_published": item.get("date_published", ""),
        "year_published": item.get("year_published", ""),
        "publication_type": item.get("publication_type", ""),
        "source_title": (item.get("source") or {}).get("title", ""),
        "authors": "; ".join(author_names),
        "abstract": (item.get("abstract") or "").replace("\n", " "),
        "doi": next(
            (e.get("value", "") for e in ext_ids if e.get("type") == "doi"), ""
        ),
        "external_ids": "; ".join(ext_id_strs),
        "scholarly_citations_count": item.get("scholarly_citations_count", 0),
        "references_count": len(item.get("references") or []),
        "is_open_access": oa.get("is_open_access", ""),
        "open_access_colour": oa.get("colour", ""),
        "open_access_license": oa.get("license", ""),
        "source_urls": "; ".join(url_strs),
        "keywords": "; ".join(keywords),
        "fields_of_study": "; ".join(fields),
    }


def compute_stats(records: list[dict], query_blocks: list[dict]) -> list[dict]:
    """Compute summary statistics from extracted records."""
    stats = []
    stats.append({"metric": "total_results_after_dedup", "value": len(records)})

    # Per-query-block counts
    block_counts: dict[str, int] = {}
    for r in records:
        qb = r.get("query_block", "")
        block_counts[qb] = block_counts.get(qb, 0) + 1
    for qb in sorted(block_counts):
        stats.append({"metric": f"query_block_{qb}", "value": block_counts[qb]})

    # Queries used
    for qb in query_blocks:
        stats.append({"metric": f"query_{qb['query_id']}", "value": qb["keywords"][:200]})

    # Per-year breakdown
    year_counts: dict[str, int] = {}
    for r in records:
        y = r.get("year_published", "")
        if y:
            key = str(y)
            year_counts[key] = year_counts.get(key, 0) + 1
    for year in sorted(year_counts):
        stats.append({"metric": f"year_{year}", "value": year_counts[year]})

    # Per-type breakdown
    type_counts: dict[str, int] = {}
    for r in records:
        t = r.get("publication_type", "")
        if t:
            type_counts[t] = type_counts.get(t, 0) + 1
    for pub_type in sorted(type_counts):
        stats.append({"metric": f"type_{pub_type}", "value": type_counts[pub_type]})

    # Open access breakdown
    oa_counts: dict[str, int] = {}
    for r in records:
        oa_val = str(r.get("is_open_access", ""))
        oa_counts[oa_val] = oa_counts.get(oa_val, 0) + 1
    for oa_val in sorted(oa_counts):
        stats.append({"metric": f"open_access_{oa_val}", "value": oa_counts[oa_val]})

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Search Lens.org scholarly API using a multi-query config CSV."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config CSV (e.g. data/input_stream_a.csv)",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output file prefix (e.g. data/processed/lens_stream_a). "
             "Defaults to config filename without extension.",
    )
    args = parser.parse_args()

    load_dotenv()
    token = os.getenv("LENS_API_TOKEN")
    if not token:
        print("Error: LENS_API_TOKEN not found in .env file.")
        sys.exit(1)

    # Determine output paths
    if args.output_prefix:
        prefix = args.output_prefix
    else:
        # Derive from config filename: data/input_stream_a.csv → data/processed/lens_stream_a
        config_stem = os.path.splitext(os.path.basename(args.config))[0]
        # Replace "input_" prefix with "lens_" for output
        output_stem = config_stem.replace("input_", "lens_", 1)
        prefix = os.path.join("data", "processed", output_stem)

    output_list_path = f"{prefix}_list.csv"
    output_stats_path = f"{prefix}_stats.csv"

    print(f"Loading config: {args.config}")
    query_blocks = load_config(args.config)
    print(f"  {len(query_blocks)} query block(s) loaded.\n")

    # Execute each query block and collect all records
    all_records = []
    for qb in query_blocks:
        query_id = qb["query_id"]
        print(f"Running {query_id}: {qb['keywords'][:80]}...")
        if "year_start" in qb or "year_end" in qb:
            print(f"  Year range: {qb.get('year_start', '?')}-{qb.get('year_end', '?')}")
        if "publication_type" in qb:
            print(f"  Publication types: {qb['publication_type']}")

        query_body = build_query(qb)
        results = search_lens(query_body, token, query_id=query_id)

        if not results:
            print(f"  {query_id}: no results found.\n")
            continue

        records = [extract_record(item) for item in results]
        # Tag each record with the query block that produced it
        for r in records:
            r["query_block"] = query_id

        all_records.extend(records)
        print()

    if not all_records:
        print("No results found across all query blocks.")
        return

    # Deduplicate on lens_id within this run.
    # Keep first occurrence — its query_block tag reflects the first query that found it.
    df = pd.DataFrame(all_records)
    before_dedup = len(df)
    df = df.drop_duplicates(subset="lens_id", keep="first")
    after_dedup = len(df)
    print(f"Deduplication: {before_dedup} → {after_dedup} ({before_dedup - after_dedup} duplicates removed)")

    # Write results
    os.makedirs(os.path.dirname(output_list_path), exist_ok=True)
    df.to_csv(output_list_path, index=False)
    print(f"Wrote {len(df)} records to {output_list_path}")

    # Write stats
    stats = compute_stats(df.to_dict("records"), query_blocks)
    df_stats = pd.DataFrame(stats)
    df_stats.to_csv(output_stats_path, index=False)
    print(f"Wrote summary stats to {output_stats_path}")


if __name__ == "__main__":
    main()
