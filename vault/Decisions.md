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

## Render native Python runtime, not Docker
No Dockerfile — `render.yaml` uses `runtime: python` with a plain build/
start command. Simpler, nothing to maintain, and the dashboard has no
dependency (like DuckDB) that actually needs to run at deploy time — it
only reads pre-built Parquet files with pandas.

## Report/narrative built incrementally
`report/executive_summary.md` is updated at the end of each project phase
(not written only at the end), so there's always an employer-presentable
deliverable available, even if the project were paused partway through.
