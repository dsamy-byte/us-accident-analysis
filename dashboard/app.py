"""US Accidents (2016-2023) — interactive dashboard.

Reads only the small pre-aggregated Parquet/JSON files under
data/processed/ (built by scripts/build_aggregates.py) — never the raw
3GB CSV, which isn't available in the deployed environment anyway.

Two kinds of view:
  - National rollups (state map, year trend) — computed once over the
    full 7.7M-row dataset, always shown at full population.
  - Cross-filterable views (weather, hour of day) — computed on the fly
    from the 200k-row row-level sample, filtered by the state/year
    selected above them, so a viewer can drill into e.g. "Texas, 2022"
    weather patterns interactively.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

st.set_page_config(page_title="US Accidents Analysis", layout="wide")


@st.cache_data
def load_data() -> dict:
    """Load every pre-aggregated Parquet file and the regression-results
    JSON once per session.

    Returns:
        dict with keys: by_state, by_weather, by_hour, by_year, sample
        (DataFrames) and regression_results (dict).
    """
    with open(DATA_DIR / "regression_results.json", encoding="utf-8") as f:
        regression_results = json.load(f)

    return {
        "by_state": pd.read_parquet(DATA_DIR / "by_state.parquet"),
        "by_year": pd.read_parquet(DATA_DIR / "by_year.parquet"),
        "sample": pd.read_parquet(DATA_DIR / "sample_for_stats.parquet"),
        "regression_results": regression_results,
    }


def filter_sample(sample: pd.DataFrame, selected_state: str, selected_year) -> pd.DataFrame:
    """Apply the state/year cross-filters to the row-level sample.

    Args:
        sample: the full row-level sample DataFrame.
        selected_state: a state abbreviation, or "All".
        selected_year: an int year, or "All".

    Returns:
        The filtered DataFrame (a view, not a copy of the whole sample
        when no filter is applied).
    """
    filtered = sample
    if selected_state != "All":
        filtered = filtered[filtered["State"] == selected_state]
    if selected_year != "All":
        filtered = filtered[filtered["year"] == selected_year]
    return filtered


def render_state_map(by_state: pd.DataFrame) -> None:
    """Render the US choropleth map plus a top-20 bar chart, both at the
    full-population (all 7.7M rows) level.
    """
    st.subheader("Accidents by state")
    fig_map = px.choropleth(
        by_state,
        locations="State",
        locationmode="USA-states",
        color="accident_count",
        scope="usa",
        color_continuous_scale="Reds",
        hover_data={"avg_severity": ":.2f"},
        title="Accident count by state",
    )
    st.plotly_chart(fig_map, use_container_width=True)

    fig_bar = px.bar(
        by_state.sort_values("accident_count", ascending=False).head(20),
        x="State",
        y="accident_count",
        color="avg_severity",
        title="Top 20 states by accident count (color = average severity)",
    )
    st.plotly_chart(fig_bar, use_container_width=True)


def render_year_trend(by_year: pd.DataFrame) -> None:
    """Render the year-over-year accident count and severity trend."""
    st.subheader("Accidents over time (2016-2023)")
    fig = px.line(
        by_year.sort_values("year"),
        x="year",
        y="accident_count",
        markers=True,
        title="Accident count by year",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_weather_section(sample: pd.DataFrame, selected_state: str, selected_year) -> None:
    """Render the weather-condition breakdown, cross-filtered by the
    state/year selected above.
    """
    st.subheader("Accidents by weather condition")
    filtered = filter_sample(sample, selected_state, selected_year)
    grouped = (
        filtered.groupby("Weather_Condition")
        .agg(accident_count=("Severity", "size"), avg_severity=("Severity", "mean"))
        .reset_index()
        .sort_values("accident_count", ascending=False)
        .head(15)
    )
    fig = px.bar(
        grouped,
        x="Weather_Condition",
        y="accident_count",
        color="avg_severity",
        title="Accident count by weather condition (color = average severity)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_hour_section(sample: pd.DataFrame, selected_state: str, selected_year) -> None:
    """Render the hour-of-day trend, cross-filtered by the state/year
    selected above.
    """
    st.subheader("Accidents by hour of day")
    filtered = filter_sample(sample, selected_state, selected_year)
    grouped = (
        filtered.groupby("hour")
        .agg(accident_count=("Severity", "size"), avg_severity=("Severity", "mean"))
        .reset_index()
        .sort_values("hour")
    )
    fig = px.line(
        grouped,
        x="hour",
        y="avg_severity",
        markers=True,
        title="Average severity by hour of day",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_key_drivers_panel(regression_results: dict) -> None:
    """Render the logistic-regression "key drivers" panel, surfacing the
    same findings as the analysis notebook directly in the dashboard.
    """
    st.subheader("Key drivers of high-severity accidents")
    st.caption(
        "Logistic regression results (target: Severity >= 3), computed "
        "in the full analysis notebook. Odds ratio > 1 means the feature "
        "increases the odds of a high-severity accident; < 1 means it "
        "decreases them."
    )

    df = pd.DataFrame(regression_results["features"])
    df["direction"] = df["odds_ratio"].apply(
        lambda o: "increases odds" if o > 1 else "decreases odds"
    )
    fig = px.bar(
        df,
        x="feature",
        y="odds_ratio",
        color="direction",
        error_y=df["ci_upper"] - df["odds_ratio"],
        error_y_minus=df["odds_ratio"] - df["ci_lower"],
        title="Odds ratio of high-severity accident by road feature (95% CI)",
    )
    fig.add_hline(y=1.0, line_dash="dash", annotation_text="No effect (OR = 1)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Pseudo R-squared: {regression_results['pseudo_r2']:.4f}")


def main() -> None:
    """Entry point: load data, render national views, then the
    cross-filterable weather/hour views and the key drivers panel.
    """
    st.title("US Accidents (2016-2023) — Statistical Analysis Dashboard")
    st.caption(
        "Data Analyst/BI portfolio project. Full statistical detail in the "
        "analysis notebook and executive summary report in this repo."
    )

    data = load_data()

    render_state_map(data["by_state"])
    render_year_trend(data["by_year"])

    st.divider()
    st.subheader("Cross-filter weather & time-of-day patterns")
    col1, col2 = st.columns(2)
    with col1:
        states = sorted(data["sample"]["State"].dropna().unique())
        selected_state = st.selectbox("Filter by state", options=["All"] + states)
    with col2:
        years = sorted(data["sample"]["year"].dropna().unique().tolist())
        selected_year = st.selectbox("Filter by year", options=["All"] + years)

    render_weather_section(data["sample"], selected_state, selected_year)
    render_hour_section(data["sample"], selected_state, selected_year)

    st.divider()
    render_key_drivers_panel(data["regression_results"])


main()
