---
title: Findings
tags: [log]
---

# Findings

These results come from a reproducible 200,000-row sample (seed 42) of
the full 7.7M-row dataset. See [[Methodology-Log]] and
`notebooks/01_eda_and_hypothesis_testing.ipynb` for the full analysis.

## H1: Severity vs. weather condition

Severity is significantly associated with weather condition
(chi-square = 9674.69, df = 276, p < 0.0001). The true p-value actually
underflows to 0 given the sample size and effect size, so it's reported
here as p < 0.0001 instead of a literal zero.

## H2: Severity vs. hour of day

Mean severity does differ by hour of day
(F = 9.83, p = 2.797e-35).

## Predictors of high-severity accidents (logistic regression)

Target: `High_Severity` (Severity >= 3). All four road-feature predictors
turned out to be statistically significant:

- **Junction**: odds ratio 1.34 (95% CI [1.29, 1.39]), p = 9.946e-49.
  Being near a junction increases the odds of a high-severity accident.
- **Crossing**: odds ratio 0.41 (95% CI [0.39, 0.44]), p = 5.143e-208.
  Being near a crossing decreases the odds.
- **Traffic_Signal**: odds ratio 0.54 (95% CI [0.52, 0.57]), p = 2.85e-165.
  Being near a traffic signal decreases the odds.
- **Stop**: odds ratio 0.30 (95% CI [0.27, 0.34]), p = 8.005e-99.
  Being near a stop sign decreases the odds, and by the largest margin
  of the four.

Pseudo R-squared: 0.0243. That's low, and it should be. These four road
features alone only explain a small slice of what drives severity.
Weather and time of day (H1, H2) clearly matter too, and neither is
included in this particular model.
