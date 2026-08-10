# Design: US Accidents Statistical Analysis (Portfolio Project)

## Purpose

A Python statistical analysis project built to demonstrate Data Analyst / BI
skills to a prospective employer. The deliverable set (notebook, report,
live dashboard, public GitHub repo) is designed to be reviewable by a
non-technical hiring stakeholder as well as a technical one.

## Dataset

**US Accidents (2016-2023)** — Kaggle, ~7.7M rows, ~1-3GB raw CSV. Chosen
for scale (impressive, well-known dataset) while remaining tractable for a
few-day timeline via the data strategy below.

## Constraints

- Timeline: a few days.
- Audience: Data Analyst / BI hiring manager — favor clear business/statistical
  narrative over pure modeling sophistication.
- Must run on a Windows machine with `uv`-managed Python.

## Data strategy (handles the scale risk)

1. **DuckDB** queries the raw CSV directly (no full in-memory load) for
   filtering, joins, and aggregation.
2. Aggregated/filtered outputs (state-level, monthly, weather-bucketed,
   representative row-level samples) are pushed into **pandas** for
   statistical testing — full 7.7M-row granularity is not required for
   valid inference at this level.
3. The dashboard reads only **pre-aggregated Parquet files**, not the raw
   CSV, so it stays fast on Render's free tier.
4. Raw CSV is never committed to git — `.gitignore`'d. Downloaded manually
   from the Kaggle dataset page and placed in `data/raw/`; no Kaggle API
   credentials or `.env` are needed for this step.

## Statistical analysis plan

- Hypothesis tests: severity distribution vs. weather condition, time-of-day,
  road feature (chi-square, ANOVA).
- Logistic regression predicting high-severity accidents from weather/time/
  road features, with odds ratios interpreted (not just accuracy reported).
- Confidence intervals and effect sizes reported alongside p-values.
- Temporal and geospatial trend analysis (accident rates by year, state,
  season).

## Tooling

- **uv** for Python environment/dependency management.
- **DuckDB** for large-scale querying/aggregation.
- **pandas / scipy / statsmodels** for statistical testing.
- **Streamlit** for the interactive dashboard.
- **GitHub Actions** for CI (lint + test on every push).
- **Render** for CD (auto-deploys the Streamlit dashboard on push to main).

## Deliverables

1. `notebooks/` — narrative analysis notebook: EDA → hypotheses → tests →
   findings.
2. `report/` — executive-summary report (PDF, generated from
   Markdown/Quarto) for a non-technical stakeholder. Built incrementally:
   every implementation phase adds its findings/narrative to this report
   as that phase completes, rather than writing it up only at the end.
   This keeps a presentable, employer-ready deliverable available at every
   checkpoint, not just after the full project is done.
3. `dashboard/` — Streamlit app (severity trends, geographic hotspots,
   weather/time patterns), deployed live on Render.
4. `vault/` — in-repo Obsidian vault: data dictionary, methodology log,
   decisions-and-why, findings. Linked notes, versioned in git.

## GitHub / deployment setup

- New GitHub account `dsamy-byte`, kept fully separate from the user's
  existing personal account:
  - Local git identity (`user.name`/`user.email`) set at the **repo level**
    only (no `--global` change).
  - Dedicated SSH key (`~/.ssh/id_ed25519_dsamy_byte`) with a host alias
    (`github.com-dsamy-byte` in `~/.ssh/config`) so this repo authenticates
    as the new account automatically via its remote URL — no manual
    account switching.
- Repo: `dsamy-byte/us-accident-analysis`, public, empty (no
  README/.gitignore/license at creation — those are added by this project's
  scaffolding).
- CI/CD flow: `git push` → GitHub → GitHub Actions runs tests/lint →
  Render detects the new commit on main → auto-redeploys the live
  dashboard.

## Project structure (top level)

```
Kaggle-Data-Project/
  .gitignore
  pyproject.toml        # uv-managed deps
  data/
    raw/                # gitignored, manually downloaded from Kaggle
    processed/          # committed — small Parquet aggregates (dashboard/CI/Render depend on these)
  notebooks/
  report/
  dashboard/
    app.py
  render.yaml            # Render native Python deploy config (no Docker)
  vault/                # Obsidian vault (data dictionary, methodology,
                         # decisions, findings)
  docs/
    superpowers/specs/  # design docs (this file)
  .github/
    workflows/          # CI
```

## Out of scope

- Tableau/Power BI — dashboard must be Python-native per the employer's
  "Python project" requirement.
- Full-fidelity dashboard queries over all 7.7M raw rows — dashboard works
  off pre-aggregated data by design.
