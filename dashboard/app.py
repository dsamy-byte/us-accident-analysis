"""US Accidents (2016-2023) — interactive dashboard.

Reads only the small pre-aggregated Parquet files under data/processed/
(built by scripts/build_aggregates.py) — never the raw 3GB CSV, which
isn't available in the deployed environment anyway.
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

st.set_page_config(page_title="US Accidents Analysis", layout="wide")


@st.cache_data
def load_data() -> dict[str, pd.DataFrame]:
    """Load every pre-aggregated Parquet file once per session.

    Returns:
        dict mapping dataset name ("by_state", "by_weather", "by_hour")
        to its DataFrame.
    """
    return {
        "by_state": pd.read_parquet(DATA_DIR / "by_state.parquet"),
        "by_weather": pd.read_parquet(DATA_DIR / "by_weather.parquet"),
        "by_hour": pd.read_parquet(DATA_DIR / "by_hour.parquet"),
    }


def render_state_section(by_state: pd.DataFrame) -> None:
    """Render the state selector and its accident-count/severity chart."""
    st.subheader("Accidents by state")
    states = sorted(by_state["State"].dropna().unique())
    selected = st.selectbox("Focus on a state", options=["All"] + states)

    display_df = by_state if selected == "All" else by_state[by_state["State"] == selected]
    fig = px.bar(
        display_df.sort_values("accident_count", ascending=False).head(20),
        x="State",
        y="accident_count",
        color="avg_severity",
        title="Accident count by state (color = average severity)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_weather_section(by_weather: pd.DataFrame) -> None:
    """Render the weather-condition breakdown chart."""
    st.subheader("Accidents by weather condition")
    fig = px.bar(
        by_weather.sort_values("accident_count", ascending=False).head(15),
        x="Weather_Condition",
        y="accident_count",
        color="avg_severity",
        title="Accident count by weather condition (color = average severity)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_hour_section(by_hour: pd.DataFrame) -> None:
    """Render the hour-of-day trend chart."""
    st.subheader("Accidents by hour of day")
    fig = px.line(
        by_hour.sort_values("hour"),
        x="hour",
        y="avg_severity",
        markers=True,
        title="Average severity by hour of day",
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    """Entry point: load data and render every dashboard section."""
    st.title("US Accidents (2016-2023) — Statistical Analysis Dashboard")
    st.caption(
        "Data Analyst/BI portfolio project. Full statistical detail in the "
        "analysis notebook and executive summary report in this repo."
    )

    data = load_data()
    render_state_section(data["by_state"])
    render_weather_section(data["by_weather"])
    render_hour_section(data["by_hour"])


main()
