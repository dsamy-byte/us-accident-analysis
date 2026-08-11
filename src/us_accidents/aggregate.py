"""Aggregation and sampling queries over the `accidents` DuckDB view.

Two kinds of output feed the rest of the project:
  - Small group-by rollups (by state, weather, hour) — used directly by
    the dashboard and for descriptive-stats sections of the report.
  - A row-level stratified sample — used by the notebook's hypothesis
    tests and logistic regression, which need individual observations,
    not just counts.
Neither ever pulls the full 7.7M-row dataset into pandas.
"""
import duckdb
import pandas as pd


def aggregate_by_state(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Accident count and mean severity per US state.

    Args:
        con: connection with the `accidents` view registered.

    Returns:
        DataFrame with columns [State, accident_count, avg_severity],
        sorted by accident_count descending.
    """
    return con.execute(
        """
        SELECT State, COUNT(*) AS accident_count, AVG(Severity) AS avg_severity
        FROM accidents
        GROUP BY State
        ORDER BY accident_count DESC
        """
    ).fetchdf()


def aggregate_by_weather(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Accident count and mean severity per weather condition.

    Args:
        con: connection with the `accidents` view registered.

    Returns:
        DataFrame with columns [Weather_Condition, accident_count, avg_severity].
    """
    return con.execute(
        """
        SELECT Weather_Condition, COUNT(*) AS accident_count, AVG(Severity) AS avg_severity
        FROM accidents
        GROUP BY Weather_Condition
        ORDER BY accident_count DESC
        """
    ).fetchdf()


def aggregate_by_hour(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Accident count and mean severity per hour of day (0-23).

    Args:
        con: connection with the `accidents` view registered.

    Returns:
        DataFrame with columns [hour, accident_count, avg_severity],
        sorted by hour ascending.
    """
    return con.execute(
        """
        SELECT EXTRACT(HOUR FROM Start_Time) AS hour,
               COUNT(*) AS accident_count,
               AVG(Severity) AS avg_severity
        FROM accidents
        GROUP BY hour
        ORDER BY hour
        """
    ).fetchdf()


def aggregate_by_year(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Accident count and mean severity per calendar year.

    Args:
        con: connection with the `accidents` view registered.

    Returns:
        DataFrame with columns [year, accident_count, avg_severity],
        sorted by year ascending.
    """
    return con.execute(
        """
        SELECT EXTRACT(YEAR FROM Start_Time) AS year,
               COUNT(*) AS accident_count,
               AVG(Severity) AS avg_severity
        FROM accidents
        GROUP BY year
        ORDER BY year
        """
    ).fetchdf()


def sample_rows(con: duckdb.DuckDBPyConnection, n: int = 200_000, seed: int = 42) -> pd.DataFrame:
    """Draw a reproducible row-level sample for hypothesis testing and
    regression, which need individual observations rather than pre-
    aggregated counts.

    Args:
        con: connection with the `accidents` view registered.
        n: number of rows to sample.
        seed: DuckDB reservoir-sampling seed, fixed for reproducibility
            so re-running the pipeline yields the same sample.

    Returns:
        DataFrame with columns [Severity, State, Weather_Condition,
        temperature_f, visibility_mi, Junction, Crossing, Traffic_Signal,
        Stop, Sunrise_Sunset, hour, year]. `year` and `State` let the
        dashboard cross-filter the weather/hour views without needing a
        separate pre-aggregated table for every combination.
    """
    return con.execute(
        f"""
        SELECT
            Severity,
            State,
            Weather_Condition,
            "Temperature(F)" AS temperature_f,
            "Visibility(mi)" AS visibility_mi,
            Junction,
            Crossing,
            Traffic_Signal,
            Stop,
            Sunrise_Sunset,
            EXTRACT(HOUR FROM Start_Time) AS hour,
            EXTRACT(YEAR FROM Start_Time) AS year
        FROM accidents
        USING SAMPLE {n} ROWS (reservoir, {seed})
        """
    ).fetchdf()


def write_parquet(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame to Parquet without the pandas row index, which
    downstream readers (dashboard, notebook) don't need.

    Args:
        df: the DataFrame to persist.
        path: destination file path.
    """
    df.to_parquet(path, index=False)
