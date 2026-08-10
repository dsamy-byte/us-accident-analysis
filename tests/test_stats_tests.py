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
