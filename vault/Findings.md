---
title: Findings
tags: [log]
---

# Findings

Based on a reproducible 200,000-row stratified sample (seed 42) of the
full 7.7M-row dataset. See [[Methodology-Log]] and
`notebooks/01_eda_and_hypothesis_testing.ipynb` for the full analysis.

## H1: Severity vs. weather condition

Severity IS significantly associated with weather condition
(chi-square = 9674.69, df = 276, p < 0.0001 — the true p-value underflows
to 0 given the sample size and effect size, so reported as p < 0.0001
rather than literal 0).

## H2: Severity vs. hour of day

Mean severity DOES differ significantly by hour of day
(F = 9.83, p = 2.797e-35).

## Predictors of high-severity accidents (logistic regression)

Target: `High_Severity` (Severity >= 3). All four road-feature predictors
are statistically significant:

- **Junction**: odds ratio = 1.34 (95% CI [1.29, 1.39]), p = 9.946e-49 —
  presence of a junction *increases* the odds of a high-severity accident.
- **Crossing**: odds ratio = 0.41 (95% CI [0.39, 0.44]), p = 5.143e-208 —
  presence of a crossing *decreases* the odds of a high-severity accident.
- **Traffic_Signal**: odds ratio = 0.54 (95% CI [0.52, 0.57]), p = 2.85e-165 —
  presence of a traffic signal *decreases* the odds of a high-severity accident.
- **Stop**: odds ratio = 0.30 (95% CI [0.27, 0.34]), p = 8.005e-99 —
  presence of a stop sign *decreases* the odds of a high-severity accident.

Pseudo R-squared: 0.0243 — these four road features alone explain a small
but statistically robust share of variance in severity; weather and time
of day (H1, H2) are additional contributing factors not included in this
particular model.
