import argparse
import csv
import json
import os
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trend_mvp.score_trend import run_pipeline


def load_trends(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def make_args(keyword: str, geo: str, start: str, end: str, timezone: int, openai_key: str, model: str, skip_gpt: bool):
    return SimpleNamespace(
        keyword=keyword,
        geo=geo,
        start=start,
        end=end,
        timezone=timezone,
        openai_key=openai_key,
        model=model,
        skip_gpt=skip_gpt,
        output=None,
    )


def run_batch(args: argparse.Namespace) -> List[Dict[str, Any]]:
    trends = load_trends(args.csv)
    results = []
    for row in trends:
        keyword = row.get("brand_or_product", "").strip()
        if not keyword:
            continue
        ns = make_args(
            keyword=keyword,
            geo=args.geo,
            start=args.start,
            end=args.end,
            timezone=args.timezone,
            openai_key=args.openai_key or os.getenv("OPENAI_API_KEY"),
            model=args.model,
            skip_gpt=args.skip_gpt,
        )
        print(f"Running: {keyword}")
        try:
            result = run_pipeline(ns)
            result["meta"] = {
                "trend_id": row.get("trend_id"),
                "category": row.get("category"),
                "first_mainstream_year": row.get("first_mainstream_year"),
                "notes": row.get("notes"),
            }
            results.append(result)
        except Exception as e:
            results.append(
                {
                    "keyword": keyword,
                    "error": str(e),
                    "meta": {
                        "trend_id": row.get("trend_id"),
                        "category": row.get("category"),
                        "first_mainstream_year": row.get("first_mainstream_year"),
                        "notes": row.get("notes"),
                    },
                }
            )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch run trend scoring over a CSV list.")
    parser.add_argument("--csv", required=True, help="Path to CSV with header: trend_id,brand_or_product,category,first_mainstream_year,notes")
    parser.add_argument("--geo", default="AU", help="Geography code (e.g., AU, US).")
    parser.add_argument("--start", default="2014-01-01", help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", default=date.today().isoformat(), help="End date YYYY-MM-DD.")
    parser.add_argument("--timezone", type=int, default=360, help="Timezone offset minutes for pytrends.")
    parser.add_argument("--openai-key", dest="openai_key", help="OpenAI API key (or set OPENAI_API_KEY).")
    parser.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model name.")
    parser.add_argument("--skip-gpt", action="store_true", help="Skip GPT call and use quantitative scores only.")
    parser.add_argument("--output", required=True, help="Path to write JSON results.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.openai_key and not os.getenv("OPENAI_API_KEY") and not args.skip_gpt:
        raise ValueError("OpenAI API key not provided. Use --openai-key or set OPENAI_API_KEY.")
    results = run_batch(args)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
