"""
Keyword extractor for OzBargain JSONL data.

Reads JSONL records (from rss/backfill ingestors), cleans text, extracts n-grams,
and writes a CSV of keywords with frequencies.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple

import spacy


# Load spaCy English model (requires: pip install spacy && python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")

# Base stopwords plus some OzBargain/deal noise words
STOPWORDS = set(nlp.Defaults.stop_words) | {
    "ozbargain",
    "ship",
    "shipping",
    "free",
    "deal",
    "bargain",
    "discount",
    "code",
    "promo",
    "voucher",
    "off",
    "sale",
    "click",
    "link",
    "expired",
    "delivered",
    "delivery",
    "coupon",
    "amazon",
}


def clean_text(text: str) -> str:
    """Lowercase, remove URLs/punct, collapse whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def iter_texts(jsonl_path: Path) -> Iterable[str]:
    """
    Yield combined text fields from the OzBargain JSONL record.
    Adjusted for rss/backfill shapes: raw_title, raw_text, plus any description/body if present.
    """
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            parts = [
                obj.get("raw_title", ""),
                obj.get("raw_text", ""),
                obj.get("title", ""),        # fallbacks if fields differ
                obj.get("description", ""),
                obj.get("body", ""),
            ]
            text = " ".join(p for p in parts if p)
            if text:
                yield text


def extract_ngrams(
    texts: List[str],
    min_freq: int = 5,
    max_ngram: int = 3,
) -> List[Tuple[str, int]]:
    """Extract unigrams/bigrams/trigrams above a frequency threshold."""
    counter = Counter()

    for raw in texts:
        cleaned = clean_text(raw)
        if not cleaned:
            continue
        doc = nlp(cleaned)
        tokens = [
            tok.text
            for tok in doc
            if not tok.is_punct and not tok.is_space and tok.text not in STOPWORDS
        ]

        # Unigrams
        for tok in tokens:
            counter[tok] += 1

        # Bigrams / trigrams
        for n in range(2, max_ngram + 1):
            for i in range(len(tokens) - n + 1):
                ngram = " ".join(tokens[i : i + n])
                counter[ngram] += 1

    items = [
        (ngram, freq)
        for ngram, freq in counter.items()
        if freq >= min_freq and len(ngram) > 1
    ]
    items.sort(key=lambda x: x[1], reverse=True)
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract keywords from OzBargain JSONL.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/ozbargain_raw.jsonl"),
        help="Path to JSONL (rss or backfill).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ozbargain_keywords.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=5,
        help="Minimum frequency to keep a keyword.",
    )
    args = parser.parse_args()

    print(f"Reading data from {args.input} ...")
    texts = list(iter_texts(args.input))
    print(f"Loaded {len(texts)} records.")

    print("Extracting n-grams...")
    ngrams = extract_ngrams(texts, min_freq=args.min_freq, max_ngram=3)
    print(f"Found {len(ngrams)} keywords with freq >= {args.min_freq}.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        f.write("keyword,frequency\n")
        for kw, freq in ngrams:
            f.write(f"\"{kw}\",{freq}\n")

    print(f"Saved keywords to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
