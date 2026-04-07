---
doc_type: methodology
title: Methodology Notes + Metric Definitions (Sample)
citation_id: METH-METRICS-V1
years_covered: 2021-2024
geo_level: state, county, tribal
metric_tags: dv_rate, sexual_assault_rate, firearm_share, dating_partner_share, reporting_proxy, risk_index
---
## Metric definitions (V1 prototype)
This document defines the V1 “sample metrics” used by the structured tools.

### dv_rate
Descriptive rate for domestic-violence-related incidents (units depend on the upstream denominator; in V1 sample data, treat as a comparable index).

### sexual_assault_rate
Descriptive rate for sexual assault incidents (same caveat about denominators in V1 sample data).

### firearm_share
Share of incidents involving a firearm (0–1).

### dating_partner_share
Share of incidents where the relationship category is “dating partner” (0–1).

### reporting_proxy
A proxy for reporting completeness/participation. **Not a direct measure** of victim reporting behavior and can reflect system changes (agency participation, data submission, classification changes).

### risk_index
A composite index intended for screening/triage in a policy analytics workflow. In V1, it is a **demonstration** value with an example component breakdown table.

## Method notes (prototype)
- Numbers returned by tools come from `metrics_sample.csv` and `risk_components_sample.csv`.
- If the tool cannot find rows for a geo/year/metric, the bot should say it does not have enough data.

