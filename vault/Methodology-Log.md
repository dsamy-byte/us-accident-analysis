---
title: Methodology Log
tags: [log]
---

# Methodology Log

## Data acquisition

Downloaded manually from the [Kaggle dataset page](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
(no Kaggle API/credentials required) into `data/raw/US_Accidents_March23.csv`
(~3.06GB, ~7.7M rows). See [[Decisions]] for why manual download was chosen
over the Kaggle API.

## Processing pipeline

1. `us_accidents.ingest.load_raw` registers the CSV as a DuckDB view
   without loading it into memory.
2. `us_accidents.ingest.validate_schema` checks the columns in
   [[Data-Dictionary]] are present.
3. `us_accidents.aggregate` produces:
   - Three group-by rollups (by state, weather, hour) for the dashboard
     and descriptive statistics.
   - A 200,000-row reproducible random sample (seed 42) for hypothesis
     testing and regression — full-dataset granularity isn't needed for
     valid inference at this scale.
4. All outputs are written to `data/processed/` as Parquet and committed
   to git (small, derived files — unlike the raw CSV).

Run via: `uv run python scripts/build_aggregates.py`

## Development environment note

The project lives on an external exFAT-formatted drive, which doesn't
support filesystem hardlinks. `uv sync` therefore does full file copies
instead of the usual instant hardlinks — installs/reinstalls are slower
than on an NTFS drive, but functionally unaffected. Accepted as a known
tradeoff rather than relocating the project.

## Project status

All phases complete: data pipeline, statistical analysis, dashboard,
CI/CD, and live deployment. See [[Findings]] for results and
`report/executive_summary.md` (also exported as `executive_summary.pdf`)
for the employer-facing summary.
