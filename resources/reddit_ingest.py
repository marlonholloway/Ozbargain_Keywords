"""
Minimal OzBargain ingestion prototype (RSS-based, no login).

Fetches the latest deals feed and writes normalized rows to a JSONL file for
downstream processing. This uses the public RSS feed to avoid brittle scraping.

Usage example:
  python resources/reddit_ingest.py \
      --feed https://www.ozbargain.com.au/deals/feed \
      --limit 50 \
      --out data/ozbargain_raw.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from datetime import datetime, timezone
from typing import Dict, List

import requests
from bs4 import BeautifulSoup


USER_AGENT = os.getenv("OZB_USER_AGENT", "delphi-trend-radar/0.1")
DEFAULT_FEED = "https://www.ozbargain.com.au/deals/feed"


def fetch_feed(feed_url: str) -> str:
    """Fetch the RSS feed XML."""
    resp = requests.get(
        feed_url,
        headers={"User-Agent": USER_AGENT},
        timeout=15,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"OzBargain fetch failed ({resp.status_code}): {resp.text}")
    return resp.text


def parse_feed(xml_text: str, limit: int) -> List[Dict]:
    """Parse RSS items into normalized rows."""
    soup = BeautifulSoup(xml_text, "xml")
    items = soup.find_all("item")[:limit]
    normalized: List[Dict] = []

    for item in items:
        link = item.link.text if item.link else None
        guid = item.guid.text if item.guid else link
        pub_date = item.pubDate.text if item.pubDate else None
        try:
            created = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %z") if pub_date else None
        except Exception:  # noqa: BLE001
            created = None

        normalized.append(
            {
                "source": "ozbargain",
                "external_id": guid,
                "parent_external_id": None,
                "url": link,
                "author_handle": item.find("dc:creator").text if item.find("dc:creator") else None,
                "author_followers": None,
                "created_at_utc": created.astimezone(timezone.utc).isoformat() if created else None,
                "collected_at_utc": datetime.now(tz=timezone.utc).isoformat(),
                "raw_title": item.title.text if item.title else None,
                "raw_text": BeautifulSoup(item.description.text, "html.parser").get_text()
                if item.description
                else None,
                "raw_metadata": {
                    "categories": [c.text for c in item.find_all("category")],
                },
            }
        )

    return normalized


def write_jsonl(records: List[Dict], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch OzBargain deals RSS to JSONL.")
    parser.add_argument("--feed", default=DEFAULT_FEED, help="RSS feed URL.")
    parser.add_argument("--limit", type=int, default=100, help="Max items to fetch.")
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("data/ozbargain_raw.jsonl"),
        help="Output JSONL path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        xml_text = fetch_feed(args.feed)
        rows = parse_feed(xml_text, args.limit)
        write_jsonl(rows, args.out)
        print(f"Wrote {len(rows)} items to {args.out}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
