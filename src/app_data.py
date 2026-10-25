"""Cached loading of the precomputed results for the app."""

import json

import pandas as pd
import streamlit as st

from src.config import DEMO_RESULTS, EVAL_RESULTS, METRICS_PATH, PRODUCTS_RESULTS


@st.cache_data
def load_demo() -> pd.DataFrame:
    return pd.read_parquet(DEMO_RESULTS)


@st.cache_data
def load_eval() -> pd.DataFrame:
    return pd.read_parquet(EVAL_RESULTS)


@st.cache_data
def load_products() -> dict:
    return json.loads(PRODUCTS_RESULTS.read_text(encoding="utf-8"))


@st.cache_data
def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data(ttl=60)
def live_unavailable_reason() -> str | None:
    from src.live import unavailable_reason

    return unavailable_reason()
