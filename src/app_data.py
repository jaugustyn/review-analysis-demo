"""Cached loading of the precomputed results for the app, and the shared chart style."""

import json

import altair as alt
import pandas as pd
import streamlit as st

from src.config import DEMO_RESULTS, EVAL_RESULTS, METRICS_PATH, PRODUCTS_RESULTS
from src.formatting import INK


def readable(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
    """Larger, darker chart text than the default, so it reads well on a projector."""
    return (chart.configure_axis(labelFontSize=13, titleFontSize=13, labelColor=INK, titleColor=INK)
            .configure_legend(labelFontSize=13, labelColor=INK))


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
