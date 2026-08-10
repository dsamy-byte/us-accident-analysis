"""Tests for us_accidents.aggregate. Expected values are hand-computed
from the 12-row fixture in tests/fixtures/sample_accidents.csv so every
assertion below is an exact, reproducible number.
"""
from pathlib import Path

import pandas as pd
import pytest

from us_accidents.aggregate import (
    aggregate_by_hour,
    aggregate_by_state,
    aggregate_by_weather,
    sample_rows,
    write_parquet,
)
from us_accidents.ingest import load_raw

FIXTURE = Path(__file__).parent / "fixtures" / "sample_accidents.csv"


def test_aggregate_by_state_counts_and_avg_severity():
    con = load_raw(str(FIXTURE))
    df = aggregate_by_state(con)
    row = df[df.State == "OH"].iloc[0]
    # OH rows: A-1(3), A-2(2), A-11(3), A-12(4) -> count 4, mean 3.0
    assert row.accident_count == 4
    assert row.avg_severity == pytest.approx(3.0)


def test_aggregate_by_weather_counts_and_avg_severity():
    con = load_raw(str(FIXTURE))
    df = aggregate_by_weather(con)
    row = df[df.Weather_Condition == "Clear"].iloc[0]
    # Clear rows: A-3(2), A-4(4), A-9(1), A-10(2) -> count 4, mean 2.25
    assert row.accident_count == 4
    assert row.avg_severity == pytest.approx(2.25)


def test_aggregate_by_hour_counts_and_avg_severity():
    con = load_raw(str(FIXTURE))
    df = aggregate_by_hour(con)
    row = df[df.hour == 6].iloc[0]
    # Hour 6 rows: A-2 (06:07, sev 2), A-11 (06:45, sev 3) -> count 2, mean 2.5
    assert row.accident_count == 2
    assert row.avg_severity == pytest.approx(2.5)


def test_sample_rows_returns_requested_count_and_columns():
    con = load_raw(str(FIXTURE))
    df = sample_rows(con, n=5, seed=42)
    assert len(df) == 5
    assert set(df.columns) == {
        "Severity",
        "State",
        "Weather_Condition",
        "temperature_f",
        "visibility_mi",
        "Junction",
        "Crossing",
        "Traffic_Signal",
        "Stop",
        "Sunrise_Sunset",
        "hour",
    }


def test_write_parquet_round_trips(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    out_path = tmp_path / "out.parquet"
    write_parquet(df, str(out_path))
    result = pd.read_parquet(out_path)
    pd.testing.assert_frame_equal(result, df)
