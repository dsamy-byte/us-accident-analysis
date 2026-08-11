"""One-time pipeline run: load the real US Accidents CSV, validate its
schema, and write the aggregate/sample Parquet files (plus a JSON summary
of the regression findings) that the notebook and dashboard depend on.

Run with: uv run python scripts/build_aggregates.py

Not part of the automated test suite — this operates on the ~3GB raw
CSV, which is never committed to git and isn't available in CI.
"""
import json
from pathlib import Path

from us_accidents.aggregate import (
    aggregate_by_hour,
    aggregate_by_state,
    aggregate_by_weather,
    aggregate_by_year,
    sample_rows,
    write_parquet,
)
from us_accidents.ingest import load_raw, validate_schema
from us_accidents.stats_tests import logistic_regression_high_severity

RAW_CSV = Path("data/raw/US_Accidents_March23.csv")
OUT_DIR = Path("data/processed")
REGRESSION_FEATURES = ["Junction", "Crossing", "Traffic_Signal", "Stop"]


def build_regression_summary(sample) -> dict:
    """Run the high-severity logistic regression on the row-level sample
    and return a JSON-serializable summary, so the dashboard's "key
    drivers" panel can display the same findings as the notebook without
    re-running statsmodels itself.

    Args:
        sample: DataFrame from aggregate.sample_rows(), with Severity and
            REGRESSION_FEATURES columns present.

    Returns:
        dict with keys: features (list of per-feature dicts with
        odds_ratio, ci_lower, ci_upper, p_value) and pseudo_r2.
    """
    sample = sample.copy()
    sample["High_Severity"] = (sample["Severity"] >= 3).astype(int)
    sample[REGRESSION_FEATURES] = sample[REGRESSION_FEATURES].astype(float)

    result = logistic_regression_high_severity(sample, REGRESSION_FEATURES)

    features = []
    for feature in REGRESSION_FEATURES:
        lower, upper = result["conf_int"][feature]
        features.append(
            {
                "feature": feature,
                "odds_ratio": float(result["odds_ratios"][feature]),
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "p_value": float(result["p_values"][feature]),
            }
        )

    return {"features": features, "pseudo_r2": float(result["pseudo_r2"])}


def main() -> None:
    """Run ingestion, validation, and every aggregation/sampling query,
    writing each result to data/processed/ as Parquet (and the regression
    summary as JSON).
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
    write_parquet(aggregate_by_year(con), str(OUT_DIR / "by_year.parquet"))

    sample = sample_rows(con, n=200_000, seed=42)
    write_parquet(sample, str(OUT_DIR / "sample_for_stats.parquet"))

    regression_summary = build_regression_summary(sample)
    with open(OUT_DIR / "regression_results.json", "w", encoding="utf-8") as f:
        json.dump(regression_summary, f, indent=2)

    print(f"Wrote aggregates to {OUT_DIR}/")


if __name__ == "__main__":
    main()
