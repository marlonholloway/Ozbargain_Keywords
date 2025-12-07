"""
OzBargain backfill scraper (requires Legal/ToS approval before running).

Scrapes paginated deal listing pages to collect historical items into JSONL.
Use only if you have approval to scrape OzBargain beyond the public RSS feed.

Example (once approved):
  source .venv/bin/activate
  python resources/ozbargain_backfill.py \
    --start-page 1 \
    --max-pages 20 \
    --out data/ozbargain_backfill.jsonl \
    --sleep 2
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.ozbargain.com.au/deals"
USER_AGENT = os.getenv("OZB_USER_AGENT", "delphi-trend-radar/0.1")


def fetch_page(page: int) -> str:
    url = f"{BASE_URL}?page={page}" if page > 0 else BASE_URL
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(f"Fetch failed ({resp.status_code}) on page {page}")
    return resp.text


def parse_page(html: str) -> List[Dict]:
    """
    Parse deal cards from an OzBargain listing page.
    Structure can change; adjust selectors as needed.
    """
    soup = BeautifulSoup(html, "lxml")
    cards = soup.select("div.node-ozbdeal") or soup.select("article")
    rows: List[Dict] = []

    for card in cards:
        title_el = card.select_one("h2.title a") or card.select_one("h2 a")
        if not title_el:
            continue
        url = title_el.get("href")
        if url and url.startswith("/"):
            url = f"https://www.ozbargain.com.au{url}"
        guid = card.get("id") or url
        author_el = card.select_one(".submitted a.username")
        time_el = card.select_one("time")
        # Many pages store datetime in datetime attribute; fallback to text.
        created_iso: Optional[str] = None
        if time_el and time_el.get("datetime"):
            try:
                dt = datetime.fromisoformat(time_el["datetime"])
                created_iso = dt.astimezone(timezone.utc).isoformat()
            except Exception:  # noqa: BLE001
                created_iso = None

        rows.append(
            {
                "source": "ozbargain",
                "external_id": guid,
                "parent_external_id": None,
                "url": url,
                "author_handle": author_el.text.strip() if author_el else None,
                "author_followers": None,
                "created_at_utc": created_iso,
                "collected_at_utc": datetime.now(tz=timezone.utc).isoformat(),
                "raw_title": title_el.text.strip(),
                "raw_text": None,
                "raw_metadata": {},
            }
        )
    return rows


def write_jsonl(records: List[Dict], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OzBargain backfill scraper (requires ToS/legal approval)."
    )
    parser.add_argument("--start-page", type=int, default=1, help="Starting page (1-indexed).")
    parser.add_argument("--max-pages", type=int, default=5, help="Max pages to crawl.")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between pages.")
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("data/ozbargain_backfill.jsonl"),
        help="Output JSONL path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_rows: List[Dict] = []

    for page in range(args.start_page, args.start_page + args.max_pages):
        try:
            html = fetch_page(page)
            rows = parse_page(html)
            all_rows.extend(rows)
            time.sleep(args.sleep)
        except Exception as exc:  # noqa: BLE001
            print(f"Error on page {page}: {exc}", file=sys.stderr)
            break

    write_jsonl(all_rows, args.out)
    print(f"Wrote {len(all_rows)} items to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
