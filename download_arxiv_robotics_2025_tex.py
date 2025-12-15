#!/usr/bin/env python3
"""
Download arXiv source (including .tex files) for robotics (cs.RO) papers
published in a given year (default: 2025).

Strategy:
  - Query cat:<category> sorted by submittedDate DESC.
  - For each entry, read <published> year.
  - Collect IDs with published.year == target year.
  - Stop once published.year < target year (because of DESC order).

Usage:
    # Default: cs.RO in 2025
    python download_arxiv_robotics_2025_tex.py

    # Different year
    python download_arxiv_robotics_2025_tex.py --year 2024

    # Different category
    python download_arxiv_robotics_2025_tex.py --category cs.AI --year 2025

    # Limit number of papers (for testing)
    python download_arxiv_robotics_2025_tex.py --limit 5
"""

import argparse
import os
import re
import sys
import tarfile
import time
from io import BytesIO
from typing import List, Optional

import requests

try:
    import feedparser
except ImportError:
    print("[ERROR] This script requires feedparser. Install it via:")
    print("        pip install feedparser")
    sys.exit(1)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_EPRINT_URL = "https://arxiv.org/e-print/{id}"

# arXiv asks for ~1 request every 3 seconds
API_SLEEP_SECONDS = 3.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download arXiv source for robotics (cs.RO) papers in a given year."
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Publication year to filter on (default: 2025).",
    )
    parser.add_argument(
        "--category",
        default="cs.RO",
        help="arXiv category to query (default: cs.RO).",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        default=None,
        help="Directory to store extracted sources (default: arxiv_tex_<category>_<year>).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of papers to download (for testing).",
    )
    parser.add_argument(
        "--max-results-per-call",
        type=int,
        default=200,
        help="Maximum results per API call (<= 2000, default: 200).",
    )
    return parser.parse_args()


def get_arxiv_ids_for_year(
    category: str,
    year: int,
    max_results_per_call: int,
    limit: Optional[int],
) -> List[str]:
    """
    Query arXiv for a given category, sorted by submittedDate DESCENDING,
    and collect all IDs whose <published> year == target year.

    Because we sort by submittedDate DESC, once we see <published>.year < year,
    we can stop paging backward.
    """
    ids: List[str] = []
    start = 0

    search_query = f"cat:{category}"
    print(f"[INFO] Querying arXiv for category={category}, year={year} ...")
    print(f"[INFO] search_query={search_query}")

    while True:
        params = {
            "search_query": search_query,
            "start": start,
            "max_results": max_results_per_call,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        print(
            f"[INFO] API query: start={start}, max_results={max_results_per_call} "
            f"(current collected: {len(ids)})"
        )
        resp = requests.get(ARXIV_API_URL, params=params, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(
                f"ArXiv API error: HTTP {resp.status_code} (start={start})"
            )

        feed = feedparser.parse(resp.text)
        entries = feed.entries
        if not entries:
            print("[INFO] No more entries returned; done paging.")
            break

        stop_entirely = False

        for e in entries:
            published = getattr(e, "published", None)
            if not published:
                # Just skip weird entries
                continue

            try:
                pub_year = int(published[:4])
            except Exception:
                continue

            if pub_year < year:
                # Because we're sorted DESC by submittedDate, once we see
                # a year older than the target, everything after will be older too.
                stop_entirely = True
                break

            if pub_year > year:
                # Newer than target year: skip but continue
                continue

            # pub_year == year
            url = e.id  # e.g., "http://arxiv.org/abs/2501.01234v1"
            m = re.search(r"arxiv\.org/abs/([^?#]+)", url)
            if not m:
                continue
            arxiv_id = m.group(1)

            ids.append(arxiv_id)
            print(f"[INFO] Found {arxiv_id} (published {published})")

            if limit is not None and len(ids) >= limit:
                print(f"[INFO] Reached limit of {limit} IDs.")
                return ids

        if stop_entirely:
            print("[INFO] Reached entries older than target year; stopping.")
            break

        start += len(entries)
        time.sleep(API_SLEEP_SECONDS)

    print(f"[INFO] Total IDs collected for {category} in {year}: {len(ids)}")
    return ids


def download_eprint(arxiv_id: str) -> bytes:
    """
    Download the arXiv e-print tarball for a given ID.

    Returns the raw bytes of the tarball.
    """
    url = ARXIV_EPRINT_URL.format(id=arxiv_id)
    print(f"[INFO] Downloading source for {arxiv_id} from {url} ...")
    resp = requests.get(url, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Failed to download source for {arxiv_id}: HTTP {resp.status_code}"
        )
    return resp.content


def extract_tarball(data: bytes, target_dir: str, arxiv_id: str):
    """
    Extract a tarball (bytes) into `target_dir/arxiv_id/`,
    sanitize paths, and list .tex files.
    """
    folder_name = arxiv_id.replace("/", "_")
    extract_path = os.path.join(target_dir, folder_name)

    os.makedirs(extract_path, exist_ok=True)

    print(f"[INFO] Extracting source for {arxiv_id} into {extract_path} ...")

    with tarfile.open(fileobj=BytesIO(data), mode="r:*") as tar:
        safe_members = []
        for member in tar.getmembers():
            # Basic path safety
            if member.name.startswith("/") or ".." in member.name.split(os.path.sep):
                print(f"[WARN] Skipping suspicious member: {member.name}")
                continue
            safe_members.append(member)

        tar.extractall(path=extract_path, members=safe_members)

    tex_files = []
    for root, _, files in os.walk(extract_path):
        for f in files:
            if f.lower().endswith(".tex"):
                tex_files.append(os.path.relpath(os.path.join(root, f), extract_path))

    if tex_files:
        print(f"[INFO] Found {len(tex_files)} .tex file(s) for {arxiv_id}:")
        for tf in tex_files:
            print(f"       - {tf}")
    else:
        print(f"[INFO] No .tex files found for {arxiv_id}.")


def main():
    args = parse_args()
    year = args.year
    category = args.category

    if args.out_dir is None:
        out_dir = f"arxiv_tex_{category.replace('.', '_')}_{year}"
    else:
        out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)

    # 1) Get all IDs for that category/year
    try:
        ids = get_arxiv_ids_for_year(
            category=category,
            year=year,
            max_results_per_call=args.max_results_per_call,
            limit=args.limit,
        )
    except Exception as e:
        print(f"[ERROR] Failed to query arXiv: {e}")
        sys.exit(1)

    if not ids:
        print(f"[INFO] No papers found for category={category}, year={year}.")
        return

    # 2) Download + extract each
    for i, arxiv_id in enumerate(ids, start=1):
        print(f"\n[INFO] Processing {i}/{len(ids)}: {arxiv_id}")
        try:
            data = download_eprint(arxiv_id)
        except Exception as e:
            print(f"[ERROR] Download failed for {arxiv_id}: {e}")
            continue

        try:
            extract_tarball(data, out_dir, arxiv_id)
        except Exception as e:
            print(f"[ERROR] Extraction failed for {arxiv_id}: {e}")
            continue

        time.sleep(API_SLEEP_SECONDS)


if __name__ == "__main__":
    main()
