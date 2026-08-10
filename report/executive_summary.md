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

Based on a reproducible 200,000-accident sample drawn from the full
7.7-million-record dataset:

- **Weather matters.** Accident severity is strongly associated with
  weather conditions at the time of the crash — this isn't a small or
  coincidental pattern; it's one of the strongest relationships found in
  the data.
- **Time of day matters.** Average accident severity meaningfully shifts
  depending on the hour of day the accident occurred.
- **Road features are strong, reliable predictors of severity.** Accidents
  near a **junction** are meaningfully more likely to be high-severity
  than accidents without one nearby. Conversely, accidents near a
  **crossing**, a **traffic signal**, or a **stop sign** are all
  meaningfully *less* likely to be high-severity — consistent with these
  features generally slowing traffic down before a potential collision.
  All four effects are statistically robust, not noise.

Full statistical detail (test statistics, p-values, confidence intervals,
odds ratios) is in `notebooks/01_eda_and_hypothesis_testing.ipynb` and
`vault/Findings.md`.

## Dashboard

_(Added once the dashboard — Task 9 — is complete.)_

## Live demo

_(Added once deployment — Task 12 — is complete.)_
