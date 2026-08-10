"""One-time pipeline run: load the real US Accidents CSV, validate its
schema, and write the aggregate/sample Parquet files that the notebook
and dashboard depend on.

Run with: uv run python scripts/build_aggregates.py

Not part of the automated test suite — this operates on the ~3GB raw
CSV, which is never committed to git and isn't available in CI.
"""
from pathlib import Path

from us_accidents.aggregate import (
    aggregate_by_hour,
    aggregate_by_state,
    aggregate_by_weather,
    sample_rows,
    write_parquet,
)
from us_accidents.ingest import load_raw, validate_schema

RAW_CSV = Path("data/raw/US_Accidents_March23.csv")
OUT_DIR = Path("data/processed")


def main() -> None:
    """Run ingestion, validation, and every aggregation/sampling query,
    writing each result to data/processed/ as Parquet.
    """
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"{RAW_CSV} not found. Download it from "
            "https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents "
            f"and place it at {RAW_CSV} before running this script."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    con = load_raw(str(RAW_CSV))
    validate_schema(con)

    write_parquet(aggregate_by_state(con), str(OUT_DIR / "by_state.parquet"))
    write_parquet(aggregate_by_weather(con), str(OUT_DIR / "by_weather.parquet"))
    write_parquet(aggregate_by_hour(con), str(OUT_DIR / "by_hour.parquet"))
    write_parquet(
        sample_rows(con, n=200_000, seed=42),
        str(OUT_DIR / "sample_for_stats.parquet"),
    )

    print(f"Wrote aggregates to {OUT_DIR}/")


if __name__ == "__main__":
    main()
