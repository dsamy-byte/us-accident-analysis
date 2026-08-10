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
    # DuckDB's CREATE VIEW is a DDL statement and can't take a prepared
    # parameter (`?`) for the file path — only DML statements like SELECT
    # support parameter binding. The path is embedded directly instead,
    # with single quotes escaped since it's a SQL string literal.
    escaped_path = csv_path.replace("'", "''")
    # read_csv_auto infers types from a sample of rows; this is fine here
    # because the dataset's columns are consistently typed throughout.
    con.execute(f"CREATE OR REPLACE VIEW accidents AS SELECT * FROM read_csv_auto('{escaped_path}')")
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
