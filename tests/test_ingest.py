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
