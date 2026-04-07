---
doc_type: limitations
title: NIBRS + Reporting Caveats (Sample)
citation_id: LIMIT-NIBRS-V1
years_covered: 2021-2024
geo_level: state, county, tribal
metric_tags: reporting_proxy, dv_rate, sexual_assault_rate
---
## Why “reporting increased” is tricky
Even when incident counts or rates change over time, several non-causal factors can contribute:
- Agency participation changes (onboarding/offboarding)
- Data submission completeness changes
- Classification and coding practice changes
- Population denominator changes (if rates are per population)

## Interpret reporting_proxy carefully
In this prototype, `reporting_proxy` is only a placeholder. In a real pipeline you should document:
- how it is calculated
- which agencies are included
- how missingness is handled

## Recommended caveat language
Observed increases/decreases can be descriptive of the reporting system as much as underlying victimization.

