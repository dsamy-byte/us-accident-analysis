---
title: Data Dictionary
tags: [reference]
---

# Data Dictionary

Source: [US Accidents (2016-2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
by Sobhan Moosavi. Columns below are the ones this analysis actually uses
(the raw file has 46 columns total; see [[Methodology-Log]] for why we
restrict to this subset).

| Column | Type | Meaning |
|---|---|---|
| Severity | int (1-4) | Impact on traffic, 1 = least, 4 = most severe |
| Start_Time | timestamp | When the accident began |
| State | string | US state abbreviation |
| Weather_Condition | string | Weather at the time of the accident |
| Temperature(F) | float | Temperature in Fahrenheit |
| Visibility(mi) | float | Visibility in miles |
| Junction | bool | Whether the accident occurred near a road junction |
| Crossing | bool | Whether the accident occurred near a crossing |
| Traffic_Signal | bool | Whether a traffic signal was present nearby |
| Stop | bool | Whether a stop sign was present nearby |
| Sunrise_Sunset | string | "Day" or "Night" at the time of the accident |

See [[Findings]] for how these are used in hypothesis tests and regression.
