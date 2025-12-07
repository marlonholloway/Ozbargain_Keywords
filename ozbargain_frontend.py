"""
Streamlit UI for exploring OzBargain keywords.

Reads data/ozbargain_keywords.csv (keyword,frequency) and lets you:
- Search by keyword substring
- Filter by minimum frequency

Run locally:
  streamlit run trend_mvp/ozbargain_frontend.py

Deploy free (Streamlit Community Cloud):
  1) Push repo to GitHub
  2) On share.streamlit.io, set entrypoint to trend_mvp/ozbargain_frontend.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from pathlib import Path

# Resolve data path relative to repo root (assumes data/ozbargain_keywords.csv exists in repo)
DATA_PATH = (Path(__file__).resolve().parent.parent / "data" / "ozbargain_keywords.csv").as_posix()

st.set_page_config(page_title="OzBargain Keywords", page_icon="🛒", layout="wide")
st.title("OzBargain Keywords")


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure columns exist
    if "keyword" not in df.columns or "frequency" not in df.columns:
        raise ValueError("Expected columns: keyword, frequency")
    return df


df = load_data(DATA_PATH)

# Sidebar controls
st.sidebar.header("Filters")
search = st.sidebar.text_input("Search keyword contains", "").strip().lower()
min_freq = st.sidebar.number_input(
    "Min frequency", min_value=1, max_value=int(df["frequency"].max()), value=5, step=1
)

# Apply filters
mask = df["frequency"] >= min_freq
if search:
    mask &= df["keyword"].str.lower().str.contains(search)
filtered = df[mask].sort_values("frequency", ascending=False)

st.caption(f"{len(filtered)} results (of {len(df)})")
st.dataframe(
    filtered.reset_index(drop=True),
    use_container_width=True,
    hide_index=True,
)

st.markdown(
    """
    **How to deploy (free)**
    1) Commit/push this repo to GitHub.
    2) Go to https://share.streamlit.io → "New app".
    3) Select the repo/branch; entrypoint: `trend_mvp/ozbargain_frontend.py`.
    4) Python 3.11+; deploy.
    """
)
