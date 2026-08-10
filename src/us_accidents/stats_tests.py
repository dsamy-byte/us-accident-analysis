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
        "significant": bool(p_value < SIGNIFICANCE_ALPHA),
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
        "significant": bool(p_value < SIGNIFICANCE_ALPHA),
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
