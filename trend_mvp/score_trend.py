import argparse
import json
import os
from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional

import numpy as np
import pandas as pd
from openai import OpenAI
from pytrends.request import TrendReq


@dataclass
class TrendInput:
    keyword: str
    geo: str
    start: str
    end: str
    tz: int = 360


def fetch_trend_series(params: TrendInput) -> pd.DataFrame:
    """Pull Google Trends interest over time for a single keyword."""
    pytrends = TrendReq(hl="en-US", tz=params.tz)
    timeframe = f"{params.start} {params.end}"
    pytrends.build_payload([params.keyword], timeframe=timeframe, geo=params.geo)
    df = pytrends.interest_over_time().reset_index()
    df = df[["date", params.keyword]].rename(columns={params.keyword: "interest"})
    df["interest"] = df["interest"].astype(float)
    return df


def compute_trend_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute base quantitative metrics from the trend series."""
    s = df["interest"]
    velocity = float(np.nan_to_num(s.diff().mean(), nan=0.0))
    acceleration = float(np.nan_to_num(s.diff().diff().mean(), nan=0.0))
    peak_interest = float(np.nan_to_num(s.max(), nan=0.0))
    mean_interest = float(np.nan_to_num(s.mean(), nan=0.0))
    std_interest = float(np.nan_to_num(s.std(), nan=0.0))
    stability = mean_interest / (std_interest + 1e-6)

    peak_idx = int(s.idxmax()) if not s.empty else 0
    pre_peak = s.iloc[: peak_idx + 1] if not s.empty else s
    peak_steepness = float(np.nan_to_num(pre_peak.diff().max(), nan=0.0))

    post_peak = s.iloc[peak_idx:] if not s.empty else s
    post_peak_stability = (
        float(
            np.nan_to_num(
                post_peak.mean() / (post_peak.std() + 1e-6),
                nan=0.0,
            )
        )
        if not post_peak.empty
        else 0.0
    )

    return {
        "mean_interest": mean_interest,
        "peak_interest": peak_interest,
        "velocity": velocity,
        "acceleration": acceleration,
        "stability": stability,
        "peak_steepness": peak_steepness,
        "post_peak_stability": post_peak_stability,
        "peak_index": peak_idx,
    }


def normalize(value: float, scale: float) -> float:
    return float(np.clip(value / scale, 0.0, 1.0))


def derive_quant_scores(metrics: Dict[str, float]) -> Dict[str, float]:
    """Map metrics to MAC(E) quant scores in 0-1 range."""
    mean_norm = normalize(metrics["mean_interest"], 100)
    stability_norm = normalize(metrics["stability"], 12)
    velocity_norm = normalize(max(metrics["velocity"], 0.0), 5)
    acceleration_norm = normalize(max(metrics["acceleration"], 0.0), 2)
    steepness_norm = normalize(max(metrics["peak_steepness"], 0.0), 20)
    post_stability_norm = normalize(metrics["post_peak_stability"], 12)

    m_quant = 0.6 * mean_norm + 0.4 * stability_norm
    c_quant = 0.55 * velocity_norm + 0.45 * acceleration_norm
    a_quant = 0.7 * steepness_norm + 0.3 * velocity_norm
    e_quant = 0.7 * post_stability_norm + 0.3 * mean_norm

    return {
        "M": round(m_quant, 3),
        "C": round(c_quant, 3),
        "A": round(a_quant, 3),
        "E": round(e_quant, 3),
    }


def summarize_signal(df: pd.DataFrame, metrics: Dict[str, float], geo: str) -> Dict[str, str]:
    """Compress the time series into a short description for GPT."""
    s = df["interest"]
    non_zero_idx = s[s > 5].index.min()
    first_seen_date = df.loc[non_zero_idx, "date"] if non_zero_idx is not None else df.iloc[0]["date"]
    peak_date = df.loc[int(metrics["peak_index"]), "date"] if not df.empty else date.today()

    volatility_ratio = metrics["stability"] ** -1 if metrics["stability"] > 0 else 0
    if volatility_ratio > 1.2:
        volatility = "high"
    elif volatility_ratio > 0.6:
        volatility = "medium"
    else:
        volatility = "low"

    if metrics["peak_steepness"] > 15 and metrics["post_peak_stability"] > metrics["stability"] * 0.8:
        growth_shape = "sharp climb then stable plateau"
    elif metrics["velocity"] > 0.5:
        growth_shape = "steady rise over time"
    elif metrics["velocity"] < -0.2:
        growth_shape = "decline after early interest"
    else:
        growth_shape = "noisy with modest movement"

    return {
        "trend": df.columns[1].replace("_", " ").title(),
        "first_seen_year": first_seen_date.year,
        "peak_year": peak_date.year,
        "growth_pattern": growth_shape,
        "volatility": volatility,
        "geo_spread": f"{geo} focus",
    }


def build_prompt(summary: Dict[str, str]) -> str:
    lines = [
        "You are a cultural analyst specialising in consumer behaviour and retail trends.",
        "",
        "I will give you a product trend and a summary of its Google Trends signal.",
        "",
        "Your task:",
        "- Analyse WHY this trend grew culturally.",
        "- Focus on human motives, lifestyle shifts, identity signaling, and media influence.",
        "- Do NOT predict future performance.",
        "- Do NOT restate the numbers.",
        "- Classify each of the following on a 0–1 scale with a short justification:",
        "",
        "M – Product Magnetism",
        "C – Social Contagion Power",
        "A – Algorithmic Amplification",
        "E – Execution & Elasticity",
        "",
        f"Trend: {summary['trend']}",
        "",
        "Observed pattern:",
        f"- First meaningful growth: {summary['first_seen_year']}",
        f"- Peak interest: {summary['peak_year']}",
        f"- Growth shape: {summary['growth_pattern']}",
        f"- Volatility: {summary['volatility']}",
        f"- Geography: {summary['geo_spread']}",
        "",
        "Return strictly in JSON:",
        "{",
        '  "M": { "score": 0.x, "analysis": "..." },',
        '  "C": { "score": 0.x, "analysis": "..." },',
        '  "A": { "score": 0.x, "analysis": "..." },',
        '  "E": { "score": 0.x, "analysis": "..." }',
        "}",
    ]
    return "\n".join(lines)


def call_gpt(prompt: str, api_key: str, model: str) -> Dict:
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You are a cultural retail analyst."},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        raise ValueError(f"GPT response was not valid JSON: {content}")


def blend_scores(quant_scores: Dict[str, float], gpt_scores: Optional[Dict]) -> Dict[str, Dict]:
    weights = {
        "M": (0.7, 0.3),
        "C": (0.7, 0.3),
        "A": (0.4, 0.6),
        "E": (0.4, 0.6),
    }
    combined = {}
    for dim in ["M", "C", "A", "E"]:
        quant = quant_scores.get(dim, 0.0)
        gpt_score = None
        gpt_analysis = None
        if gpt_scores and isinstance(gpt_scores.get(dim), dict):
            gpt_score = gpt_scores[dim].get("score")
            gpt_analysis = gpt_scores[dim].get("analysis")

        if gpt_score is None:
            final_score = quant
        else:
            q_wt, g_wt = weights[dim]
            final_score = q_wt * quant + g_wt * gpt_score

        combined[dim] = {
            "score": round(final_score, 3),
            "quant": round(quant, 3),
            "gpt": None if gpt_score is None else round(gpt_score, 3),
            "analysis": gpt_analysis or "GPT analysis not provided.",
        }
    return combined


def compute_viral_score(combined: Dict[str, Dict[str, float]]) -> float:
    product = 1.0
    for dim in ["M", "C", "A", "E"]:
        product *= combined[dim]["score"]
    return round(product, 4)


def run_pipeline(args) -> Dict:
    params = TrendInput(
        keyword=args.keyword,
        geo=args.geo,
        start=args.start,
        end=args.end,
        tz=args.timezone,
    )
    df = fetch_trend_series(params)
    metrics = compute_trend_metrics(df)
    quant_scores = derive_quant_scores(metrics)
    summary = summarize_signal(df, metrics, params.geo)
    prompt = build_prompt(summary)

    gpt_result = None
    if not args.skip_gpt:
        api_key = args.openai_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided. Use --openai-key or set OPENAI_API_KEY.")
        gpt_result = call_gpt(prompt, api_key, args.model)

    combined_scores = blend_scores(quant_scores, gpt_result)
    viral_score = compute_viral_score(combined_scores)

    return {
        "keyword": params.keyword,
        "timeframe": {"start": params.start, "end": params.end},
        "geo": params.geo,
        "metrics": metrics,
        "quant_scores": quant_scores,
        "gpt_scores": gpt_result,
        "combined_scores": combined_scores,
        "viral_score": viral_score,
        "prompt": prompt,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute MAC(E) trend scores using Google Trends and GPT.")
    parser.add_argument("--keyword", required=True, help="Keyword to track in Google Trends.")
    parser.add_argument("--geo", default="AU", help="Geography code (e.g., AU, US).")
    parser.add_argument("--start", default="2014-01-01", help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", default=date.today().isoformat(), help="End date YYYY-MM-DD.")
    parser.add_argument("--timezone", type=int, default=360, help="Timezone offset minutes for pytrends.")
    parser.add_argument("--openai-key", dest="openai_key", help="OpenAI API key (or set OPENAI_API_KEY).")
    parser.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model name.")
    parser.add_argument("--skip-gpt", action="store_true", help="Skip GPT call and use quantitative scores only.")
    parser.add_argument("--output", help="Optional path to write results as JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_pipeline(args)
    print(f"Viral score for '{result['keyword']}': {result['viral_score']}")
    for dim, values in result["combined_scores"].items():
        print(f"{dim}: {values['score']} (quant={values['quant']}, gpt={values['gpt']})")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved details to {args.output}")


if __name__ == "__main__":
    main()
