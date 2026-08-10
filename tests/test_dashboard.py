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
