---
doc_type: faq
title: VAWA Insights Bot – Frequently Asked Questions
citation_id: vawa_insights_faq
years_covered: varies_by_dataset
geo_level: state, county (where available)
metric_tags: [dv_rate, firearm_share, dating_partner_share, reporting_proxy, risk_index]
source: internal_kb
---

## About This Tool

### What is the VAWA Insights Bot meant to do (and not do)?
The VAWA Insights Bot is designed for exploratory analysis of violence-related metrics using structured datasets and a curated knowledge base. It supports comparisons, rankings, time series exploration, and risk profiling. It is not a legal advisory tool, does not provide case-specific guidance, and does not represent a complete interpretation of the Violence Against Women Act (VAWA). Outputs should be treated as analytical summaries, not definitive conclusions.

### Where do the numbers come from?
All numerical outputs are derived from structured datasets processed through a defined data pipeline. The knowledge base provides definitions, methodology, and context but does not generate or store raw numerical values. This separation ensures that metrics are reproducible and traceable to underlying data sources.

### Why do answers include “citations” and “debug” information?
Citations and debug details are included to support transparency and reproducibility. They allow users to verify how a result was generated, including which dataset, metric, and filters were applied. This is especially important for academic or policy-oriented use.

### Does this system use a large language model (LLM)?
The system primarily relies on deterministic functions (e.g., compare, rank, time series) and retrieval from a structured knowledge base. Any language generation is constrained and does not independently create numerical claims. This design minimizes hallucination risk.

---

## Data & Geography

### What years and geographies does the data cover?
Coverage depends on the dataset used in the analysis. Most examples operate at the state level, with some datasets supporting county-level analysis. Users should verify the requested year and geography are available before interpreting results.

### What is a “state” vs “county” vs “tribal” row?
- **State:** Aggregated data at the state level  
- **County:** Sub-state administrative regions  
- **Tribal:** Jurisdictions associated with tribal lands (availability may vary)  

These categories reflect how data is reported and may differ in completeness.

### What does a data quality or coverage flag mean?
Coverage flags indicate whether a dataset for a given geography and year is complete, partial, or missing. Partial coverage may reflect underreporting or incomplete agency participation.

### Why might a state or year be missing?
Missing data may result from:
- Non-participation in reporting systems
- Incomplete submissions
- Data validation exclusions

Absence of data does not imply absence of incidents.

---

## Metrics & Definitions

### What is `dv_rate` (and what is it not)?
`dv_rate` represents a normalized rate of reported domestic violence incidents, typically adjusted by population. It reflects reported incidents, not total prevalence, and should not be interpreted as the true rate of occurrence.

### What are `firearm_share`, `dating_partner_share`, `reporting_proxy`, and `risk_index`?
- **firearm_share:** Proportion of incidents involving a firearm  
- **dating_partner_share:** Proportion involving dating partners  
- **reporting_proxy:** Indicator approximating reporting behavior or system engagement  
- **risk_index:** Composite metric combining multiple indicators  

See `glossary.md` for full details.

### Why are some metrics called “share” or “proxy”?
- **Share:** Represents a proportion within reported incidents  
- **Proxy:** Represents an indirect measure of a concept that cannot be directly observed (e.g., reporting behavior)

### Can I compare metrics across states fairly?
Comparisons require caution:
- Population size varies
- Reporting practices differ
- Data coverage is inconsistent  

Cross-state comparisons are best used for directional insights, not definitive rankings.

---

## VAWA / Policy Context

### What is VAWA in one paragraph?
The Violence Against Women Act (VAWA) is a U.S. federal law that supports prevention, protection, and prosecution efforts related to domestic violence, sexual assault, dating violence, and stalking. It provides funding, establishes legal protections, and promotes coordinated responses across agencies.

### What did the 2022 reauthorization emphasize?
The 2022 reauthorization expanded protections for underserved populations, strengthened firearm-related provisions, and increased support for prevention and survivor services. It also emphasized equity and access.

### How does federal policy relate to incident-level data?
Federal policy shapes funding, enforcement priorities, and reporting structures. However, policy changes are not directly observable in incident-level datasets without careful analytical design.

---

## Methods & Limitations

### What is NIBRS and how is it different from legacy UCR?
The National Incident-Based Reporting System (NIBRS) provides detailed, incident-level data, including victim-offender relationships and context. It replaces the older Summary Reporting System (UCR), which reported aggregated counts with less detail.

### What are limitations of administrative data for this topic?
- Underreporting is common  
- Participation varies by jurisdiction  
- Definitions and classifications may differ  

These factors limit the ability to measure true prevalence.

### Why doesn’t a trend prove causation?
Observed changes in metrics may reflect reporting shifts, policy changes, or data coverage differences. Without controlled analysis, trends should not be interpreted as causal effects.

### What does EDA or modeling add beyond raw counts?
Exploratory Data Analysis (EDA) and modeling help identify patterns, normalize comparisons, and construct composite indicators. However, they do not eliminate underlying data limitations.

---

## How to Ask Good Questions

### What are examples of effective questions?
- Compare dv_rate between two states in a given year  
- Rank states by risk_index  
- Show time trends for a metric  
- Generate a risk profile for a state  

### What should I include in comparison or ranking questions?
- Metric name  
- Year  
- Geography level (state or county)  

### What should I ask for definitions or caveats?
Use keywords such as:
- “definition”  
- “methodology”  
- “limitations”  
- “NIBRS”  

---

## Trust, Safety, and Scope

### Is this official government data?
The system uses structured datasets that may originate from government sources but includes derived metrics created by the project team. Outputs should be considered analytical, not official statistics.

### Who should I contact in an emergency or personal situation?
This tool is not designed for crisis response. For immediate help:
- Call 911 in emergencies  
- Contact the National Domestic Violence Hotline (1-800-799-7233)  

Always seek qualified support services for personal situations.