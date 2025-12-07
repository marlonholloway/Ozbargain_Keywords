"""
Simple Streamlit frontend for trend browsing/search.

Reads a CSV of trends (data/trends_batch.csv) and provides:
- Wide text search across brand/product + notes
- Category filter

Run locally:
  streamlit run trend_mvp/trend_frontend.py

Deploy free:
- Push to GitHub
- On streamlit.io (Community Cloud), point to this repo and this file as the entrypoint.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

DATA_PATH = "data/trends_batch.csv"

st.set_page_config(page_title="Trend Browser", page_icon="📈", layout="wide")
st.title("Trend Browser")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize column names expected in this MVP
    df.rename(
        columns={
            "brand_or_product": "name",
            "category": "category",
            "first_mainstream_year": "year",
            "notes": "notes",
        },
        inplace=True,
    )
    return df


df = load_data(DATA_PATH)

# Sidebar filters
st.sidebar.header("Filters")
search = st.sidebar.text_input("Search (name/notes contains)", "")
categories = sorted(df["category"].dropna().unique())
selected_cats = st.sidebar.multiselect("Category", categories, default=categories)

# Apply filters
mask = df["category"].isin(selected_cats)
if search:
    search_lower = search.lower()
    mask &= df["name"].str.lower().str.contains(search_lower) | df["notes"].str.lower().str.contains(search_lower)
filtered = df[mask]

st.caption(f"{len(filtered)} results (of {len(df)})")
st.dataframe(
    filtered[["trend_id", "name", "category", "year", "notes"]].reset_index(drop=True),
    use_container_width=True,
    hide_index=True,
)

st.markdown(
    """
    **How to deploy (free)**
    1) Commit/push this repo to GitHub.
    2) Go to https://share.streamlit.io/ → "New app".
    3) Select the repo/branch and set entrypoint to `trend_mvp/trend_frontend.py`.
    4) Set Python version to 3.11+ (matches your project), click Deploy.
    """
)
