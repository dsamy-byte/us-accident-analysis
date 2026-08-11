# US Accidents Statistical Analysis

A statistical analysis of the [US Accidents (2016-2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
dataset (~7.7M records), built as a Data Analyst/BI portfolio project.

## Contents

- `notebooks/01_eda_and_hypothesis_testing.ipynb` — full analysis: EDA,
  hypothesis tests, logistic regression, interpreted results.
- `report/executive_summary.md` — non-technical write-up of findings.
- `dashboard/` — Streamlit dashboard ([live demo](https://us-accident-analysis-dashboard.onrender.com)).
- `vault/` — project documentation vault (data dictionary, methodology log,
  decisions, findings) in Obsidian-compatible Markdown.

## Setup

1. Install [uv](https://docs.astral.sh/uv/).
2. `uv sync`
3. Download the raw dataset from Kaggle (see `vault/Methodology-Log.md`)
   into `data/raw/US_Accidents_March23.csv`.
4. `uv run python scripts/build_aggregates.py` to regenerate the
   processed Parquet files (already committed under `data/processed/`,
   this step is only needed if you want to rebuild them from scratch).
5. `uv run jupyter lab` to open the analysis notebook, or
   `uv run streamlit run dashboard/app.py` to run the dashboard locally.

## Testing

`uv run pytest`
