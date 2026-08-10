# US Accidents Statistical Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Data Analyst/BI portfolio project — statistical analysis of the
US Accidents (2016-2023) dataset — delivered as a documented notebook, an
incrementally-written executive report, and a live Streamlit dashboard on
Render, in a public GitHub repo under a dedicated `dsamy-byte` account.

**Architecture:** DuckDB queries the raw 3GB CSV directly for
filtering/aggregation without loading it into memory; small pre-aggregated
Parquet outputs (state/weather/hour rollups + a 200k-row stratified sample)
are committed to git and are what both the notebook's statistical tests and
the deployed dashboard actually read. The raw CSV itself never leaves the
local machine.

**Tech Stack:** Python 3.11, `uv`, DuckDB, pandas, scipy, statsmodels,
pyarrow, Streamlit, GitHub Actions, Render.

## Global Constraints

- Timeline is a few days — prefer the simplest approach that meets the spec; no gold-plating.
- All code must be well documented: every module has a module-level docstring, every function has a docstring (purpose, Args, Returns), and non-obvious logic (statistical method choices, SQL sampling behavior, etc.) gets an inline comment explaining *why*, not what.
- Dashboard must be Python-native (Streamlit) — not Tableau/Power BI.
- Raw CSV (`data/raw/US_Accidents_March23.csv`) is never committed to git.
- Dashboard and CI/CD both read only pre-aggregated Parquet files (`data/processed/`) — never the raw CSV. These small Parquet files ARE committed to git (they're KBs-MBs, and Render/CI need them since neither has access to the local raw file).
- `report/executive_summary.md` is written incrementally — every task that produces findings appends its section to this report as part of that task, not at the end.
- Git identity for this repo is local-only (`dsamy-byte` / `dsamyuktha@gmail.com`, already configured, no `--global` changes). Remote already set: `origin` → `git@github.com-dsamy-byte:dsamy-byte/us-accident-analysis.git` on branch `main`.
- CI (GitHub Actions) runs lint + unit tests on every push. CD (Render) auto-redeploys the dashboard on every push to `main`. Notebook execution is NOT part of automated CI (too slow/heavy for free-tier Actions minutes) — it's run and verified manually.
- Runs on Windows via `uv`-managed Python; all commands below assume the Bash/Git-Bash shell already used in this project.

---

## File Structure

```
Kaggle-Data-Project/
  pyproject.toml
  .python-version
  README.md
  src/
    us_accidents/
      __init__.py
      ingest.py          # DuckDB loading + schema validation
      aggregate.py        # aggregation + sampling queries, Parquet writer
      stats_tests.py      # chi-square, ANOVA, logistic regression wrappers
  tests/
    fixtures/
      sample_accidents.csv
    test_ingest.py
    test_aggregate.py
    test_stats_tests.py
    test_dashboard.py
  scripts/
    build_aggregates.py   # one-time pipeline run against the real 3GB CSV
  notebooks/
    01_eda_and_hypothesis_testing.ipynb
  report/
    executive_summary.md
  dashboard/
    app.py
    Dockerfile
    render.yaml
  vault/
    00-Index.md
    Data-Dictionary.md
    Methodology-Log.md
    Decisions.md
    Findings.md
  data/
    raw/                  # gitignored — manually downloaded, never committed
    processed/             # committed — small aggregate Parquet files
  .github/
    workflows/
      ci.yml
```

---

### Task 1: Project Scaffolding with uv

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Modify: `.gitignore` (fix `data/processed/` — must NOT be ignored, only `data/raw/` should be)

**Interfaces:**
- Produces: a working `uv`-managed environment (`.venv/`) with all project dependencies installed and importable; corrected `.gitignore` that every later task relies on for what gets committed.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "us-accident-analysis"
version = "0.1.0"
description = "Statistical analysis of the US Accidents (2016-2023) dataset — a Data Analyst/BI portfolio project"
requires-python = ">=3.11"
dependencies = [
    "duckdb>=1.0.0",
    "pandas>=2.2.0",
    "numpy>=1.26.0",
    "scipy>=1.13.0",
    "statsmodels>=0.14.0",
    "pyarrow>=16.0.0",
    "matplotlib>=3.9.0",
    "seaborn>=0.13.0",
    "plotly>=5.22.0",
    "streamlit>=1.38.0",
    "jupyter>=1.0.0",
    "ipykernel>=6.29.0",
]

[tool.uv]
dev-dependencies = [
    "pytest>=8.3.0",
    "ruff>=0.6.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/us_accidents"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
src = ["src", "tests", "scripts", "dashboard"]
```

- [ ] **Step 2: Pin Python and sync the environment**

Run: `uv python pin 3.11`
Expected: creates `.python-version` containing `3.11`.

Run: `uv sync`
Expected: creates `.venv/`, installs all dependencies, installs `us_accidents` itself in editable mode. Exit code 0.

- [ ] **Step 3: Verify all key libraries import**

Run: `uv run python -c "import duckdb, pandas, numpy, scipy, statsmodels, pyarrow, matplotlib, seaborn, plotly, streamlit; print('ok')"`
Expected output: `ok`

- [ ] **Step 4: Fix `.gitignore` — `data/processed/` must be committed, not ignored**

The current `.gitignore` ignores all of `data/processed/`. That's wrong: the
dashboard, CI, and Render deployment all depend on the small aggregate
Parquet files living there — only the multi-GB raw CSV should be excluded.

Edit `.gitignore`, replacing:
```
# Data (downloaded via Kaggle API, never committed)
data/raw/
data/processed/
```
with:
```
# Raw data is 1-3GB and never committed. Small pre-aggregated Parquet
# files under data/processed/ ARE committed — the dashboard, CI, and
# Render deployment all depend on them and have no way to regenerate
# them without the local raw CSV.
data/raw/
```

- [ ] **Step 5: Write `README.md`**

```markdown
# US Accidents Statistical Analysis

A statistical analysis of the [US Accidents (2016-2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
dataset (~7.7M records), built as a Data Analyst/BI portfolio project.

## Contents

- `notebooks/01_eda_and_hypothesis_testing.ipynb` — full analysis: EDA,
  hypothesis tests, logistic regression, interpreted results.
- `report/executive_summary.md` — non-technical write-up of findings.
- `dashboard/` — Streamlit dashboard ([live demo](#) — link added once deployed).
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
```

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .python-version README.md .gitignore
git commit -m "chore: scaffold uv project, fix data/processed gitignore rule"
git push
```

---

### Task 2: Data Ingestion Module (DuckDB)

**Files:**
- Create: `src/us_accidents/__init__.py`
- Create: `src/us_accidents/ingest.py`
- Create: `tests/fixtures/sample_accidents.csv`
- Create: `tests/test_ingest.py`

**Interfaces:**
- Consumes: nothing (first module in the pipeline).
- Produces: `load_raw(csv_path: str) -> duckdb.DuckDBPyConnection` (registers
  a view named `accidents`), `validate_schema(con: duckdb.DuckDBPyConnection) -> None`
  (raises `SchemaValidationError` if required columns are missing),
  `REQUIRED_COLUMNS: list[str]`, `SchemaValidationError` — all consumed by
  Task 3's aggregation module and Task 7's real-data pipeline script.

- [ ] **Step 1: Create the package init**

`src/us_accidents/__init__.py`:
```python
"""us_accidents: statistical analysis toolkit for the US Accidents
(2016-2023) Kaggle dataset.

Modules:
    ingest: load the raw CSV into DuckDB and validate its schema.
    aggregate: aggregation/sampling queries and Parquet output.
    stats_tests: hypothesis testing and regression wrappers used by the
        analysis notebook.
"""
```

- [ ] **Step 2: Create the test fixture CSV**

`tests/fixtures/sample_accidents.csv` — a small, hand-built 12-row fixture
matching the columns our analysis actually uses (a subset of the real
dataset's 46 columns; the real CSV has more columns than this, which is
fine since our SQL only selects the ones it needs):

```csv
ID,Severity,Start_Time,State,Weather_Condition,Temperature(F),Visibility(mi),Junction,Crossing,Traffic_Signal,Stop,Sunrise_Sunset
A-1,3,2016-02-08 05:46:00,OH,Light Rain,36.9,10.0,False,False,False,False,Night
A-2,2,2016-02-08 06:07:59,OH,Light Rain,37.9,10.0,False,False,False,False,Night
A-3,2,2016-06-15 14:20:00,CA,Clear,75.0,10.0,True,False,True,False,Day
A-4,4,2016-06-15 15:10:00,CA,Clear,78.0,10.0,True,False,True,False,Day
A-5,1,2017-01-10 08:00:00,TX,Fog,42.0,2.0,False,True,False,False,Day
A-6,3,2017-01-10 09:15:00,TX,Fog,44.0,1.5,False,True,False,False,Day
A-7,2,2018-03-22 22:00:00,NY,Snow,20.0,0.5,True,False,False,True,Night
A-8,4,2018-03-22 23:30:00,NY,Snow,18.0,0.5,True,False,False,True,Night
A-9,1,2019-07-04 12:00:00,FL,Clear,90.0,10.0,False,False,True,False,Day
A-10,2,2019-07-04 13:00:00,FL,Clear,91.0,10.0,False,False,True,False,Day
A-11,3,2020-11-11 06:45:00,OH,Rain,45.0,5.0,False,False,False,False,Night
A-12,4,2020-11-11 07:30:00,OH,Rain,44.0,4.5,False,False,False,False,Night
```

- [ ] **Step 3: Write the failing tests**

`tests/test_ingest.py`:
```python
"""Tests for us_accidents.ingest: loading the CSV into DuckDB and
validating that the columns our analysis depends on are present.
"""
from pathlib import Path

import pytest

from us_accidents.ingest import SchemaValidationError, load_raw, validate_schema

FIXTURE = Path(__file__).parent / "fixtures" / "sample_accidents.csv"


def test_load_raw_returns_expected_row_count():
    """load_raw should expose all rows of the CSV via the `accidents` view."""
    con = load_raw(str(FIXTURE))
    count = con.execute("SELECT COUNT(*) FROM accidents").fetchone()[0]
    assert count == 12


def test_validate_schema_passes_for_valid_fixture():
    """The fixture contains every column our analysis needs, so validation
    should pass silently (no exception raised).
    """
    con = load_raw(str(FIXTURE))
    validate_schema(con)  # should not raise


def test_validate_schema_raises_for_missing_column(tmp_path):
    """A CSV missing required columns (e.g. Weather_Condition) must fail
    fast with a clear error, rather than letting downstream SQL fail with
    a confusing "column not found" error deep in the pipeline.
    """
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("ID,Severity\nA-1,1\n")
    con = load_raw(str(bad_csv))
    with pytest.raises(SchemaValidationError):
        validate_schema(con)
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'us_accidents.ingest'`

- [ ] **Step 5: Implement `src/us_accidents/ingest.py`**

```python
"""Load the US Accidents CSV into an in-memory DuckDB view and validate
that it has the columns the rest of the pipeline depends on.

DuckDB is used instead of pandas here because the real dataset is a
~3GB / 7.7M-row CSV: DuckDB queries it directly off disk without loading
the whole file into memory, which pandas.read_csv would require.
"""
import duckdb

# Only the columns actually used downstream (aggregation, hypothesis
# tests, regression) need to be present. The real CSV has 46 columns;
# we don't care about most of them.
REQUIRED_COLUMNS = [
    "ID",
    "Severity",
    "Start_Time",
    "State",
    "Weather_Condition",
    "Temperature(F)",
    "Visibility(mi)",
    "Junction",
    "Crossing",
    "Traffic_Signal",
    "Stop",
    "Sunrise_Sunset",
]


class SchemaValidationError(ValueError):
    """Raised when a CSV is missing one or more REQUIRED_COLUMNS."""


def load_raw(csv_path: str) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection and register the CSV at
    `csv_path` as a view named `accidents`.

    Args:
        csv_path: path to the accidents CSV (fixture or the real
            3GB dataset).

    Returns:
        A DuckDB connection with the `accidents` view registered.
        Callers query it with `con.execute(sql).fetchdf()`.
    """
    con = duckdb.connect(database=":memory:")
    # read_csv_auto infers types from a sample of rows; this is fine here
    # because the dataset's columns are consistently typed throughout.
    con.execute(
        "CREATE OR REPLACE VIEW accidents AS SELECT * FROM read_csv_auto(?)",
        [csv_path],
    )
    return con


def validate_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Raise SchemaValidationError if the `accidents` view is missing any
    column in REQUIRED_COLUMNS.

    Failing fast here means a malformed or unexpected CSV produces one
    clear error message instead of a confusing failure deep inside an
    aggregation query.

    Args:
        con: a connection returned by load_raw().

    Raises:
        SchemaValidationError: if any required column is absent.
    """
    columns = {row[0] for row in con.execute("DESCRIBE accidents").fetchall()}
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise SchemaValidationError(f"Missing required columns: {missing}")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: 3 passed

- [ ] **Step 7: Commit**

```bash
git add src/us_accidents/__init__.py src/us_accidents/ingest.py tests/fixtures/sample_accidents.csv tests/test_ingest.py
git commit -m "feat: add DuckDB ingestion module with schema validation"
git push
```

---

### Task 3: Aggregation & Sampling Module

**Files:**
- Create: `src/us_accidents/aggregate.py`
- Create: `tests/test_aggregate.py`

**Interfaces:**
- Consumes: `load_raw` from Task 2 (`us_accidents.ingest`).
- Produces: `aggregate_by_state`, `aggregate_by_weather`, `aggregate_by_hour`
  (each `(con) -> pd.DataFrame`), `sample_rows(con, n, seed) -> pd.DataFrame`,
  `write_parquet(df, path) -> None` — all consumed by Task 7's real-data
  pipeline script and, transitively, by the notebook (Task 8) and dashboard
  (Task 9) which read the Parquet files these functions produce.

- [ ] **Step 1: Write the failing tests**

`tests/test_aggregate.py`. Expected values below are hand-computed from the
12-row fixture (e.g. OH rows are A-1, A-2, A-11, A-12 with severities
3, 2, 3, 4 → count 4, mean 3.0):

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_aggregate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'us_accidents.aggregate'`

- [ ] **Step 3: Implement `src/us_accidents/aggregate.py`**

```python
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
        Stop, Sunrise_Sunset, hour].
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
            EXTRACT(HOUR FROM Start_Time) AS hour
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_aggregate.py -v`
Expected: 5 passed

If `USING SAMPLE ... ROWS (reservoir, seed)` errors on the installed DuckDB
version, check `uv run python -c "import duckdb; print(duckdb.__version__)"`
and consult that version's sampling syntax docs — the reservoir-sampling
method name and seed placement are the parts most likely to have shifted
between versions.

- [ ] **Step 5: Commit**

```bash
git add src/us_accidents/aggregate.py tests/test_aggregate.py
git commit -m "feat: add aggregation and sampling queries"
git push
```

---

### Task 4: Statistical Tests Module

**Files:**
- Create: `src/us_accidents/stats_tests.py`
- Create: `tests/test_stats_tests.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (operates on any DataFrame with the
  right columns — tested here with synthetic data, used in Task 8 with the
  real sample from Task 3's `sample_rows`).
- Produces: `chi_square_severity_by_weather(df) -> dict`,
  `anova_severity_by_hour(df) -> dict`,
  `logistic_regression_high_severity(df, feature_cols) -> dict` — all
  consumed by the notebook in Task 8.

- [ ] **Step 1: Write the failing tests**

Synthetic fixtures are used here (not the CSV fixture) because these tests
need to assert *statistical* properties — a clear effect vs. no effect —
which is easiest to construct directly rather than reverse-engineer from
12 rows of accident data.

`tests/test_stats_tests.py`:
```python
"""Tests for us_accidents.stats_tests. Each test builds a small synthetic
DataFrame with a known statistical property (a strong effect, or no
effect at all) so the expected significance of the result is known in
advance, independent of the real dataset.
"""
import pandas as pd
import pytest

from us_accidents.stats_tests import (
    anova_severity_by_hour,
    chi_square_severity_by_weather,
    logistic_regression_high_severity,
)


def test_chi_square_detects_strong_association():
    """Weather A always maps to severity 1, weather B always to severity
    2 — a perfect association, so the test must find significance.
    """
    df = pd.DataFrame(
        {
            "Weather_Condition": ["A"] * 25 + ["B"] * 25,
            "Severity": [1] * 25 + [2] * 25,
        }
    )
    result = chi_square_severity_by_weather(df)
    assert result["p_value"] < 0.05
    assert result["significant"] is True


def test_chi_square_finds_no_association_in_balanced_table():
    """A perfectly balanced contingency table (equal counts in every
    cell) has zero association, so the test must NOT find significance.
    """
    df = pd.DataFrame(
        {
            "Weather_Condition": ["A"] * 25 + ["A"] * 25 + ["B"] * 25 + ["B"] * 25,
            "Severity": [1] * 25 + [2] * 25 + [1] * 25 + [2] * 25,
        }
    )
    result = chi_square_severity_by_weather(df)
    assert result["p_value"] > 0.05
    assert result["significant"] is False


def test_anova_detects_separated_group_means():
    """Three hour-groups with clearly separated severity distributions
    must produce a significant ANOVA result.
    """
    df = pd.DataFrame(
        {
            "hour": [0] * 8 + [12] * 8 + [18] * 8,
            "Severity": (
                [1, 1, 2, 1, 1, 2, 1, 1]
                + [3, 2, 3, 3, 2, 3, 3, 2]
                + [5, 4, 5, 5, 4, 5, 5, 4]
            ),
        }
    )
    result = anova_severity_by_hour(df)
    assert result["p_value"] < 0.05
    assert result["significant"] is True


def test_anova_finds_no_difference_for_identical_groups():
    """Three groups with the exact same distribution must NOT produce a
    significant ANOVA result.
    """
    df = pd.DataFrame(
        {
            "hour": [0] * 8 + [12] * 8 + [18] * 8,
            "Severity": [2, 3, 2, 3, 2, 3, 2, 3] * 3,
        }
    )
    result = anova_severity_by_hour(df)
    assert result["p_value"] > 0.05
    assert result["significant"] is False


def test_logistic_regression_detects_positive_effect():
    """Junction=1 rows are high-severity 80% of the time, Junction=0 rows
    only 20% of the time — a real, sizeable effect (not perfect
    separation, which would make statsmodels' Logit fail to converge).
    """
    df = pd.DataFrame(
        {
            "High_Severity": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0] + [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            "Junction": [1] * 10 + [0] * 10,
        }
    )
    result = logistic_regression_high_severity(df, feature_cols=["Junction"])
    assert result["p_values"]["Junction"] < 0.05
    assert result["odds_ratios"]["Junction"] > 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_stats_tests.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'us_accidents.stats_tests'`

- [ ] **Step 3: Implement `src/us_accidents/stats_tests.py`**

```python
"""Statistical test wrappers used by the analysis notebook.

Each function returns a plain dict of results (statistic, p-value,
significance flag, and any effect-size/odds-ratio numbers) rather than a
raw scipy/statsmodels object, so the notebook can print or format results
without needing to know the internals of each library's result type.
"""
import pandas as pd
import statsmodels.api as sm
from scipy import stats

SIGNIFICANCE_ALPHA = 0.05


def chi_square_severity_by_weather(df: pd.DataFrame) -> dict:
    """Chi-square test of independence between Severity and
    Weather_Condition.

    Tests whether accident severity is associated with weather condition,
    or whether the two are independent.

    Args:
        df: DataFrame with 'Severity' and 'Weather_Condition' columns.

    Returns:
        dict with keys: statistic, p_value, dof, significant.
    """
    contingency = pd.crosstab(df["Weather_Condition"], df["Severity"])
    statistic, p_value, dof, _expected = stats.chi2_contingency(contingency)
    return {
        "statistic": statistic,
        "p_value": p_value,
        "dof": dof,
        "significant": p_value < SIGNIFICANCE_ALPHA,
    }


def anova_severity_by_hour(df: pd.DataFrame) -> dict:
    """One-way ANOVA testing whether mean Severity differs across hour of
    day.

    Args:
        df: DataFrame with numeric 'Severity' and integer 'hour' (0-23)
            columns.

    Returns:
        dict with keys: statistic, p_value, significant.
    """
    groups = [group["Severity"].values for _, group in df.groupby("hour")]
    statistic, p_value = stats.f_oneway(*groups)
    return {
        "statistic": statistic,
        "p_value": p_value,
        "significant": p_value < SIGNIFICANCE_ALPHA,
    }


def logistic_regression_high_severity(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Logistic regression predicting High_Severity (0/1) from the given
    feature columns, with results reported as odds ratios (more
    interpretable to a BI audience than raw log-odds coefficients).

    Args:
        df: DataFrame with a binary 'High_Severity' target column and
            numeric/boolean feature_cols predictors.
        feature_cols: names of predictor columns to include.

    Returns:
        dict with keys:
            odds_ratios: {feature: exp(coefficient)}
            conf_int: {feature: (lower, upper)} — 95% CI on the odds ratio
            p_values: {feature: p_value}
            pseudo_r2: McFadden's pseudo R-squared for the fitted model
    """
    X = sm.add_constant(df[feature_cols].astype(float))
    y = df["High_Severity"].astype(float)
    model = sm.Logit(y, X).fit(disp=0)

    odds_ratios = model.params.apply(lambda coef: pow(2.718281828, coef))
    conf = model.conf_int()
    conf_odds = conf.apply(lambda col: col.apply(lambda coef: pow(2.718281828, coef)))

    return {
        "odds_ratios": {f: odds_ratios[f] for f in feature_cols},
        "conf_int": {f: (conf_odds.loc[f, 0], conf_odds.loc[f, 1]) for f in feature_cols},
        "p_values": {f: model.pvalues[f] for f in feature_cols},
        "pseudo_r2": model.prsquared,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_stats_tests.py -v`
Expected: 5 passed. (A `ConvergenceWarning` or similar from statsmodels is
fine as long as the test assertions pass — it does not indicate failure.)

- [ ] **Step 5: Commit**

```bash
git add src/us_accidents/stats_tests.py tests/test_stats_tests.py
git commit -m "feat: add chi-square, ANOVA, and logistic regression wrappers"
git push
```

---

### Task 5: Obsidian Vault Scaffold

**Files:**
- Create: `vault/00-Index.md`
- Create: `vault/Data-Dictionary.md`
- Create: `vault/Methodology-Log.md`
- Create: `vault/Decisions.md`
- Create: `vault/Findings.md`

**Interfaces:**
- Produces: the vault structure that Task 8 (notebook) appends real
  findings to, and that documents decisions already made in this project
  for anyone (including the employer) browsing the repo.

- [ ] **Step 1: Create `vault/00-Index.md`**

```markdown
---
title: Project Index
tags: [index]
---

# US Accidents Analysis — Project Vault

- [[Data-Dictionary]] — columns used in this analysis and what they mean
- [[Methodology-Log]] — how the data was acquired and processed
- [[Decisions]] — key project decisions and the reasoning behind them
- [[Findings]] — statistical findings as they're produced by the analysis
```

- [ ] **Step 2: Create `vault/Data-Dictionary.md`**

```markdown
---
title: Data Dictionary
tags: [reference]
---

# Data Dictionary

Source: [US Accidents (2016-2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
by Sobhan Moosavi. Columns below are the ones this analysis actually uses
(the raw file has 46 columns total; see [[Methodology-Log]] for why we
restrict to this subset).

| Column | Type | Meaning |
|---|---|---|
| Severity | int (1-4) | Impact on traffic, 1 = least, 4 = most severe |
| Start_Time | timestamp | When the accident began |
| State | string | US state abbreviation |
| Weather_Condition | string | Weather at the time of the accident |
| Temperature(F) | float | Temperature in Fahrenheit |
| Visibility(mi) | float | Visibility in miles |
| Junction | bool | Whether the accident occurred near a road junction |
| Crossing | bool | Whether the accident occurred near a crossing |
| Traffic_Signal | bool | Whether a traffic signal was present nearby |
| Stop | bool | Whether a stop sign was present nearby |
| Sunrise_Sunset | string | "Day" or "Night" at the time of the accident |

See [[Findings]] for how these are used in hypothesis tests and regression.
```

- [ ] **Step 3: Create `vault/Methodology-Log.md`**

```markdown
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
```

- [ ] **Step 4: Create `vault/Decisions.md`**

```markdown
---
title: Decisions
tags: [log]
---

# Decisions

## Dataset: US Accidents (2016-2023)
Chosen for scale (~7.7M rows) and popularity, while remaining tractable
for a few-day timeline via DuckDB + aggregation rather than loading
everything into pandas.

## DuckDB for querying, not a hosted database
DuckDB is an embedded analytical engine — it queries the CSV directly off
local disk, no server or upload required. A hosted database (e.g.
Render's free Postgres) was considered and rejected: free-tier storage
caps are too small for the raw file, free databases expire without
upgrading to paid, and row-store Postgres is the wrong engine for bulk
analytical aggregation compared to DuckDB's columnar execution. The
dashboard never needs the raw data anyway — it only reads small
pre-aggregated Parquet files.

## Manual dataset download instead of the Kaggle API
Simpler for this project's scope — avoids needing Kaggle API credentials
or a `.env` file for a one-time download step.

## Separate GitHub account (`dsamy-byte`)
Kept fully isolated from the user's personal GitHub account: git identity
is set at the repo level only (no `--global` changes), and a dedicated
SSH key with a host alias (`github.com-dsamy-byte`) means this repo
authenticates as the new account automatically via its remote URL.

## Render over Streamlit Community Cloud
User preference — Render hosts the dashboard as a web service, deploying
automatically from GitHub on every push to `main`.

## Report/narrative built incrementally
`report/executive_summary.md` is updated at the end of each project phase
(not written only at the end), so there's always an employer-presentable
deliverable available, even if the project were paused partway through.
```

- [ ] **Step 5: Create `vault/Findings.md`**

```markdown
---
title: Findings
tags: [log]
---

# Findings

_Populated incrementally as analysis tasks complete. See [[Methodology-Log]]
for how each finding was produced._
```

- [ ] **Step 6: Commit**

```bash
git add vault/
git commit -m "docs: scaffold Obsidian vault with data dictionary and decisions log"
git push
```

---

### Task 6: Report Scaffold

**Files:**
- Create: `report/executive_summary.md`

**Interfaces:**
- Produces: the report template that Task 8 (notebook), Task 9 (dashboard),
  and Task 12 (deployment) each append a section to.

- [ ] **Step 1: Create `report/executive_summary.md`**

```markdown
# US Accidents (2016-2023): Statistical Analysis — Executive Summary

**Prepared as a Data Analyst/BI portfolio project.** Full technical detail
is in `notebooks/01_eda_and_hypothesis_testing.ipynb`; this document is
the non-technical summary of what was found and why it matters.

## Project overview

Statistical analysis of ~7.7 million US traffic accident records
(2016-2023), examining what factors are associated with accident severity
— weather, time of day, and road features — using hypothesis testing and
logistic regression, presented alongside an interactive dashboard.

## Findings

_(Added once the analysis notebook — Task 8 — is complete.)_

## Dashboard

_(Added once the dashboard — Task 9 — is complete.)_

## Live demo

_(Added once deployment — Task 12 — is complete.)_
```

- [ ] **Step 2: Commit**

```bash
git add report/executive_summary.md
git commit -m "docs: scaffold executive summary report"
git push
```

---

### Task 7: Build Real Aggregates from the Full Dataset

**Files:**
- Create: `scripts/build_aggregates.py`
- Create (generated, not hand-written): `data/processed/by_state.parquet`,
  `data/processed/by_weather.parquet`, `data/processed/by_hour.parquet`,
  `data/processed/sample_for_stats.parquet`

**Interfaces:**
- Consumes: `load_raw`, `validate_schema` (Task 2); `aggregate_by_state`,
  `aggregate_by_weather`, `aggregate_by_hour`, `sample_rows`, `write_parquet`
  (Task 3).
- Produces: the committed Parquet files that Task 8 (notebook) and Task 9
  (dashboard) both read.

This task runs against the real 3GB CSV, so it is a manual
run-and-verify step rather than an automated pytest test (too slow/heavy
for a test suite, and the raw file isn't available in CI anyway).

- [ ] **Step 1: Write `scripts/build_aggregates.py`**

```python
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
```

- [ ] **Step 2: Run it against the real dataset**

Run: `uv run python scripts/build_aggregates.py`
Expected: prints `Wrote aggregates to data/processed/`. Takes a few
minutes given the file size — this is expected, DuckDB is scanning the
full 3GB file for the aggregation queries.

- [ ] **Step 3: Verify the outputs**

Run:
```bash
uv run python -c "
import pandas as pd
for name in ['by_state', 'by_weather', 'by_hour', 'sample_for_stats']:
    df = pd.read_parquet(f'data/processed/{name}.parquet')
    print(name, df.shape)
"
```
Expected: four lines printed, each with a nonzero row count — e.g.
`by_state (50-ish, 3)`, `sample_for_stats (200000, 11)`. Exact state/hour
counts depend on the real data and aren't predictable in advance; the
check here is "non-empty, right shape," not exact values (those exact
values were already verified against the fixture in Task 3's tests).

- [ ] **Step 4: Commit the generated Parquet files**

These are small (KBs to low MBs) and, per the Global Constraints, must be
committed — the dashboard and Render deployment have no other way to get
this data.

```bash
git add scripts/build_aggregates.py data/processed/
git commit -m "feat: build and commit aggregate Parquet files from the full dataset"
git push
```

---

### Task 8: EDA & Hypothesis Testing Notebook

**Files:**
- Create: `notebooks/01_eda_and_hypothesis_testing.ipynb`
- Modify: `vault/Findings.md`
- Modify: `report/executive_summary.md`

**Interfaces:**
- Consumes: `data/processed/*.parquet` (Task 7); `chi_square_severity_by_weather`,
  `anova_severity_by_hour`, `logistic_regression_high_severity` (Task 4).
- Produces: real statistical findings that Findings.md, the report, and
  (for the headline numbers) the dashboard's narrative text all draw from.

Notebooks are authored interactively rather than hand-written as JSON.
Create `notebooks/01_eda_and_hypothesis_testing.ipynb` in Jupyter with
the following cells, in order. Every markdown cell that needs to state a
real result uses an f-string in the preceding code cell to print the
sentence from the actual computed statistic — never a hand-typed number —
so the notebook's narrative is always accurate to what the data actually
shows.

- [ ] **Step 1: Title and setup cells**

Markdown cell:
```markdown
# US Accidents (2016-2023): Exploratory Analysis & Hypothesis Testing

Data Analyst/BI portfolio project. See `vault/Data-Dictionary.md` for
column definitions and `vault/Methodology-Log.md` for how this data was
processed.
```

Code cell:
```python
"""Setup: load the pre-aggregated Parquet files produced by
scripts/build_aggregates.py. This notebook never touches the raw
3GB CSV directly.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from us_accidents.stats_tests import (
    anova_severity_by_hour,
    chi_square_severity_by_weather,
    logistic_regression_high_severity,
)

by_state = pd.read_parquet("../data/processed/by_state.parquet")
by_weather = pd.read_parquet("../data/processed/by_weather.parquet")
by_hour = pd.read_parquet("../data/processed/by_hour.parquet")
sample = pd.read_parquet("../data/processed/sample_for_stats.parquet")

sns.set_theme(style="whitegrid")
```

- [ ] **Step 2: Descriptive EDA cells**

Code cell (top states by accident count):
```python
fig, ax = plt.subplots(figsize=(10, 5))
top_states = by_state.head(15)
sns.barplot(data=top_states, x="State", y="accident_count", ax=ax)
ax.set_title("Top 15 states by accident count")
plt.tight_layout()
plt.savefig("../report/fig_top_states.png", dpi=150)
plt.show()
```

Code cell (severity by hour of day):
```python
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=by_hour, x="hour", y="avg_severity", marker="o", ax=ax)
ax.set_title("Average accident severity by hour of day")
ax.set_xlabel("Hour (0-23)")
plt.tight_layout()
plt.savefig("../report/fig_severity_by_hour.png", dpi=150)
plt.show()
```

- [ ] **Step 3: Hypothesis test — severity vs. weather**

Code cell:
```python
"""H1: Accident severity is associated with weather condition."""
chi2_result = chi_square_severity_by_weather(sample)
print(
    f"Chi-square = {chi2_result['statistic']:.2f}, "
    f"p = {chi2_result['p_value']:.4g}, "
    f"df = {chi2_result['dof']}"
)
chi2_finding = (
    f"Severity IS significantly associated with weather condition "
    f"(chi-square = {chi2_result['statistic']:.2f}, p = {chi2_result['p_value']:.4g})."
    if chi2_result["significant"]
    else
    f"No significant association found between severity and weather "
    f"condition (chi-square = {chi2_result['statistic']:.2f}, p = {chi2_result['p_value']:.4g})."
)
print(chi2_finding)
```

Markdown cell:
```markdown
**Interpretation:** see the printed finding above — generated directly
from the test result, not hand-written, so it always matches what the
data actually shows.
```

- [ ] **Step 4: Hypothesis test — severity vs. hour of day**

Code cell:
```python
"""H2: Mean accident severity differs by hour of day."""
anova_result = anova_severity_by_hour(sample)
anova_finding = (
    f"Mean severity DOES differ significantly by hour of day "
    f"(F = {anova_result['statistic']:.2f}, p = {anova_result['p_value']:.4g})."
    if anova_result["significant"]
    else
    f"No significant difference in mean severity across hours of day "
    f"(F = {anova_result['statistic']:.2f}, p = {anova_result['p_value']:.4g})."
)
print(anova_finding)
```

- [ ] **Step 5: Logistic regression — predictors of high severity**

Code cell:
```python
"""Logistic regression: which road/weather/time features predict a
high-severity accident (Severity >= 3)?
"""
sample["High_Severity"] = (sample["Severity"] >= 3).astype(int)
feature_cols = ["Junction", "Crossing", "Traffic_Signal", "Stop"]
sample[feature_cols] = sample[feature_cols].astype(float)

logit_result = logistic_regression_high_severity(sample, feature_cols)

for feature in feature_cols:
    odds = logit_result["odds_ratios"][feature]
    p = logit_result["p_values"][feature]
    lower, upper = logit_result["conf_int"][feature]
    direction = "increases" if odds > 1 else "decreases"
    sig = "significant" if p < 0.05 else "not significant"
    print(
        f"{feature}: odds ratio = {odds:.2f} (95% CI [{lower:.2f}, {upper:.2f}]), "
        f"p = {p:.4g} ({sig}) — presence of {feature} {direction} the odds "
        f"of a high-severity accident."
    )
print(f"Pseudo R-squared: {logit_result['pseudo_r2']:.4f}")
```

- [ ] **Step 6: Run the notebook end-to-end**

Run: `uv run jupyter nbconvert --to notebook --execute notebooks/01_eda_and_hypothesis_testing.ipynb --output 01_eda_and_hypothesis_testing.ipynb`
Expected: exits 0, no cell raises an exception. Open the executed notebook
and read the printed finding strings from Steps 3-5 — copy those exact
printed sentences (they contain the real numbers) into `vault/Findings.md`
and `report/executive_summary.md` in the next step.

- [ ] **Step 7: Update `vault/Findings.md` with the real results**

Replace the placeholder body with the actual printed findings from Step 6,
e.g. (illustrative structure — use the real printed sentences from your
run, not these exact numbers):

```markdown
---
title: Findings
tags: [log]
---

# Findings

## H1: Severity vs. weather condition
<paste the chi-square finding sentence printed in the notebook>

## H2: Severity vs. hour of day
<paste the ANOVA finding sentence printed in the notebook>

## Predictors of high-severity accidents (logistic regression)
<paste the per-feature odds-ratio sentences printed in the notebook>

Pseudo R-squared: <value printed in the notebook>

See `notebooks/01_eda_and_hypothesis_testing.ipynb` for the full analysis.
```

- [ ] **Step 8: Update `report/executive_summary.md`'s Findings section**

Replace `_(Added once the analysis notebook — Task 8 — is complete.)_`
under `## Findings` with a plain-English translation of the same results
(no p-values/statistics jargon — this section is for a non-technical
reader), e.g.:

```markdown
## Findings

- Weather conditions are meaningfully associated with how severe an
  accident is — [summarize the direction, e.g. "accidents in fog/low
  visibility skew more severe than accidents in clear weather"].
- Accident severity varies by time of day — [summarize the pattern from
  the hour-of-day chart].
- Road features near the accident location predict severity: [summarize
  which of Junction/Crossing/Traffic_Signal/Stop increased or decreased
  the odds of a high-severity accident, in plain terms].

Full statistical detail (p-values, confidence intervals, odds ratios) is
in `notebooks/01_eda_and_hypothesis_testing.ipynb` and `vault/Findings.md`.
```

- [ ] **Step 9: Commit**

```bash
git add notebooks/ vault/Findings.md report/executive_summary.md
git commit -m "feat: complete EDA and hypothesis testing, document findings"
git push
```

---

### Task 9: Streamlit Dashboard

**Files:**
- Create: `dashboard/app.py`
- Create: `tests/test_dashboard.py`
- Modify: `report/executive_summary.md`

**Interfaces:**
- Consumes: `data/processed/*.parquet` (Task 7, already committed).
- Produces: a runnable Streamlit app, verified both by an automated
  `AppTest`-based test and by manual local run.

- [ ] **Step 1: Write the failing test**

Streamlit's `AppTest` runs the app script headlessly (no browser) and
lets us assert it renders without raising — this is Streamlit's own
supported testing API, so no browser automation tooling is needed.

`tests/test_dashboard.py`:
```python
"""Smoke test for the Streamlit dashboard using Streamlit's AppTest API,
which runs the app script headlessly and exposes any exception it raised.
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "dashboard" / "app.py")


def test_dashboard_runs_without_exception():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    assert not at.exception


def test_dashboard_renders_state_selector():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    assert len(at.selectbox) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_dashboard.py -v`
Expected: FAIL — `FileNotFoundError` or `ModuleNotFoundError` (`dashboard/app.py` doesn't exist yet).

- [ ] **Step 3: Implement `dashboard/app.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dashboard.py -v`
Expected: 2 passed

- [ ] **Step 5: Manually verify the app renders**

Run: `uv run streamlit run dashboard/app.py`
Expected: opens in the browser at `localhost:8501`, shows the title, a
state selector, and three charts with no errors in the terminal. Stop
with Ctrl+C when confirmed.

- [ ] **Step 6: Update `report/executive_summary.md`'s Dashboard section**

Replace `_(Added once the dashboard — Task 9 — is complete.)_` under
`## Dashboard` with:

```markdown
## Dashboard

An interactive Streamlit dashboard lets you explore accident counts and
severity by state, weather condition, and hour of day. Run locally with
`uv run streamlit run dashboard/app.py`, or see the live link below.
```

- [ ] **Step 7: Commit**

```bash
git add dashboard/app.py tests/test_dashboard.py report/executive_summary.md
git commit -m "feat: add Streamlit dashboard with state/weather/hour views"
git push
```

---

### Task 10: GitHub Actions CI Workflow

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `pyproject.toml` (Task 1), the full `tests/` suite (Tasks 2-4, 9).
- Produces: automated lint + test run on every push/PR, which is the "CI"
  half of the CI/CD pipeline the user asked for.

- [ ] **Step 1: Write `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          version: "latest"

      - name: Set up Python
        run: uv python install 3.11

      - name: Install dependencies
        run: uv sync

      - name: Lint
        run: uv run ruff check .

      - name: Run tests
        run: uv run pytest -v
```

- [ ] **Step 2: Verify locally before pushing**

Run: `uv run ruff check .`
Expected: exits 0, or lists fixable issues — if any, run `uv run ruff check . --fix` and re-check.

Run: `uv run pytest -v`
Expected: all tests from Tasks 2, 3, 4, and 9 pass.

- [ ] **Step 3: Commit and push, then verify the workflow runs**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions workflow for lint and tests"
git push
```

Then check `https://github.com/dsamy-byte/us-accident-analysis/actions` —
expect a green run for this push.

---

### Task 11: Dockerfile + Render Config

**Files:**
- Create: `dashboard/Dockerfile`
- Create: `render.yaml`

**Interfaces:**
- Consumes: `dashboard/app.py` (Task 9), `pyproject.toml` (Task 1).
- Produces: the artifacts Render needs to build and run the dashboard as a
  web service.

- [ ] **Step 1: Write `dashboard/Dockerfile`**

```dockerfile
FROM python:3.11-slim

# Install uv for fast, reproducible dependency installation matching
# the lockfile used in development.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml ./
COPY src/ ./src/
COPY dashboard/ ./dashboard/
COPY data/processed/ ./data/processed/

RUN uv sync --no-dev

# Render provides $PORT at runtime; Streamlit must bind to it.
EXPOSE 8501
CMD uv run streamlit run dashboard/app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true
```

- [ ] **Step 2: Write `render.yaml`**

```yaml
services:
  - type: web
    name: us-accident-analysis-dashboard
    runtime: docker
    dockerfilePath: ./dashboard/Dockerfile
    dockerContext: .
    plan: free
    autoDeploy: true
    branch: main
```

- [ ] **Step 3: Verify the Dockerfile builds, if Docker is available locally**

Run: `docker build -f dashboard/Dockerfile -t us-accident-dashboard .`
Expected: exits 0. If Docker isn't installed locally, skip this step —
Render will build it directly; verification happens in Task 12 once
deployed.

- [ ] **Step 4: Commit**

```bash
git add dashboard/Dockerfile render.yaml
git commit -m "build: add Dockerfile and render.yaml for Render deployment"
git push
```

---

### Task 12: Deploy to Render

**Files:**
- Modify: `report/executive_summary.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: `render.yaml`, `dashboard/Dockerfile` (Task 11), and the
  `dsamy-byte/us-accident-analysis` GitHub repo (already connected).
- Produces: a live public URL for the dashboard.

This task requires manual steps on Render's website (account creation and
GitHub connection aren't automatable from the terminal) — same pattern as
the earlier GitHub account setup.

- [ ] **Step 1: Create a Render account and connect GitHub**

At render.com, sign up (or log in), then connect your GitHub account and
grant access to the `dsamy-byte/us-accident-analysis` repository.

- [ ] **Step 2: Create the web service from `render.yaml`**

In the Render dashboard: New → Blueprint → select the repo → Render
detects `render.yaml` and proposes the `us-accident-analysis-dashboard`
service → confirm and deploy.

- [ ] **Step 3: Wait for the first deploy and verify**

Watch the build logs in Render's dashboard until status is "Live". Visit
the assigned `https://<service-name>.onrender.com` URL and confirm the
dashboard renders with its charts, matching the local run from Task 9
Step 5.

- [ ] **Step 4: Update `report/executive_summary.md`'s Live demo section**

Replace `_(Added once deployment — Task 12 — is complete.)_` under
`## Live demo` with:

```markdown
## Live demo

[Live dashboard](https://<actual-service-url>.onrender.com)

Note: on Render's free tier the service sleeps after inactivity and may
take ~30 seconds to wake on first load.
```

- [ ] **Step 5: Update `README.md` with the live link**

Replace `([live demo](#) — link added once deployed)` with the real URL.

- [ ] **Step 6: Commit**

```bash
git add report/executive_summary.md README.md
git commit -m "docs: add live dashboard link"
git push
```

---

### Task 13: Final Polish

**Files:**
- Modify: `report/executive_summary.md`
- Modify: `vault/Methodology-Log.md`

**Interfaces:**
- Consumes: everything produced by Tasks 1-12.
- Produces: the final employer-ready state of the repo.

- [ ] **Step 1: Add a closing "Conclusion" section to the report**

Append to `report/executive_summary.md`:
```markdown
## Conclusion

This project demonstrates an end-to-end Data Analyst/BI workflow: scalable
data processing (DuckDB) on a real 7.7M-record dataset, rigorous
statistical inference (hypothesis testing, logistic regression with
interpreted effect sizes) rather than purely descriptive analysis, and a
deployed, interactive way to explore the results — with full reproducibility
via automated tests and CI/CD.
```

- [ ] **Step 2: Add a final entry to the methodology log noting completion**

Append to `vault/Methodology-Log.md`:
```markdown
## Project status

All phases complete: data pipeline, statistical analysis, dashboard,
CI/CD, and live deployment. See [[Findings]] for results and
`report/executive_summary.md` for the employer-facing summary.
```

- [ ] **Step 3: Run the full test suite one last time**

Run: `uv run pytest -v`
Expected: all tests pass.

Run: `uv run ruff check .`
Expected: no issues.

- [ ] **Step 4: Final commit and push**

```bash
git add report/executive_summary.md vault/Methodology-Log.md
git commit -m "docs: finalize report and methodology log"
git push
```
