---
doc_type: analysis_results
title: Post-2022 Analysis Results for VAWA-Aligned Indicators
citation_id: vawa_post2022_analysis_results
years_covered: pre_2022_vs_post_2022
geo_level: state
metric_tags: [sexual_assault_share, sexual_assault_rate, native_victim_share, tribal_land_incident_rate, unmarried_partner_share, dv_firearm_rate, reporting_rate, composite_improvement_score]
source: internal_analysis
---

## Overview

This document summarizes selected post-2022 findings for policy-relevant indicators aligned with the 2022 VAWA reauthorization. The analysis compares **pre-2022** and **post-2022** periods using state-level data and a mix of descriptive summaries, fixed-effects regressions, paired tests, Wilcoxon signed-rank tests, and binomial tests.

These findings should be interpreted as **associations in observed data**, not proof that policy changes caused the measured outcomes.

---

## Component 1: Rates of Sexual Assault

### Policy Context
VAWA reauthorized federal funding for sexual assault prevention and response programs through 2027.

### Research Question
Is there evidence of a shift in sexual assault cases after 2022?

### Results
- **Overall share declined after 2022**
  - Panel regression: **Post_2022 = -0.0141, p < 0.001**
  - Interpreted as a decline of about **1.4 percentage points** in the share of sexual assault incidents after 2022, controlling for state and year fixed effects.

- **Trend is widespread across states**
  - **50 states decreased**
  - **1 state increased**: Florida (**+0.008**)
  - Mean change across states: **-0.0204** (about **-2.0 percentage points**)

- **Within-state decline is statistically significant**
  - Paired t-test: **t = -12.03, p < 0.001**
  - Wilcoxon signed-rank test: **p = 7.8e-10**

- **Population-adjusted rate also declined**
  - Absolute rate model: **Post_2022 = -0.000150, p < 0.001**
  - Interpreted in the project notes as approximately **15 fewer cases per 10,000 women after 2022**

### Interpretation Notes
- Both the **share** and the **population-adjusted rate** declined significantly after 2022.
- This pattern is more consistent with a reduction in incidents involving sexual assault than with a simple compositional shift in reported crime types.
- However, this analysis does **not** establish policy efficacy or causation.

---

## Component 2: Proportion of Victims Who Are Native American

### Policy Context
The 2022 reauthorization expanded tribal jurisdiction, including authority to prosecute additional crimes involving non-Native offenders.

### Research Question
Is there evidence of a shift in Native American victim representation after 2022?

### Results
**Note:** Florida was excluded from this component because it was treated as an extreme outlier.

- **No significant within-state change in Native victim share**
  - Pre-period mean: **4.78%**
  - Post-period mean: **4.61%**
  - Paired t-test: **p = 0.255**
  - Direction is slightly downward, but not statistically significant

- **No consistent state-level trend**
  - **56% of states increased**
  - Binomial test: **p = 0.48**
  - This does not support a systematic national shift

- **Overrepresentation remained, but did not change significantly**
  - Adjusted Native victim share:
    - Pre: **2.38%**
    - Post: **2.49%**
  - Paired t-test: **p = 0.253**
  - Native American victims remained overrepresented at about **2.4 times their population share**, but that overrepresentation did not change meaningfully after 2022

- **No meaningful change in incidents on tribal lands**
  - Pre: **0.000983**
  - Post: **0.000725**
  - Paired t-test: **p = 0.299**
  - State-level decrease share: **16% of states decreased**
  - No clear post-2022 pattern

### Interpretation Notes
- This component does **not** show a statistically significant post-2022 shift.
- The findings are best interpreted as **relative stability**, not improvement or deterioration.
- Expanded jurisdiction may matter in practice, but the effect is not clearly detectable here.

---

## Component 3: Changes in Victim-Offender Relationship Proportions Over Time

### Policy Context
VAWA expanded domestic violence protections to dating partners, addressing gaps affecting some non-married intimate partners.

### Research Question
Do the post-2022 data show a meaningful change in the share of domestic violence incidents involving nonmarried partners?

### Results
- **Within-state share declined significantly**
  - Mean change: **-0.0303**
  - Median change: **-0.0288**
  - Paired t-test: **p = 0.0**
  - Wilcoxon signed-rank test: **W = 94, p = 0.000**

- **Decline is widespread across states**
  - **44 states (86%) decreased**
  - **7 states (14%) increased**

- **Relative to spouse incidents, unmarried partner incidents also declined**
  - Regression: **Post_2022 = -0.1701, p = 0.002**

- **Absolute rates also decreased**
  - Project notes indicate that absolute rates of unmarried partner incidents went down, not just the share

### Interpretation Notes
- The observed decline is statistically strong and geographically widespread.
- However, interpretation is not straightforward:
  - A decline may reflect fewer incidents
  - It may also reflect changes in reporting, classification, or enforcement patterns
- Because the policy also aimed to improve recognition and protection for dating partners, this result should **not** be treated as a simple success or failure signal.

---

## Component 4: Proportion of DV Incidents That Involved a Firearm

### Policy Context
The 2022 reauthorization expanded firearm prohibitions for individuals convicted of qualifying domestic violence offenses, including certain dating partners.

### Research Question
Did the number of incidents involving both domestic violence and a firearm change meaningfully after 2022?

### Results
- **No meaningful change in total incidents**
  - Regression coefficient: **Post_2022 = 0.72, p = 0.96**
  - Pre-2022 mean: **320.33**
  - Post-2022 mean: **321.29**

- **No widespread state-level effect**
  - **45% of states increased**
  - **55% of states decreased**

- **No significant within-state decline in adjusted rates**
  - Pre mean rate: **13.67**
  - Post mean rate: **13.33**
  - Fixed-effects result: **Post_2022 = -0.25, p = 0.60**

### Interpretation Notes
- The estimated effect is very small and not statistically significant.
- This component does **not** provide evidence of a measurable post-2022 shift in DV incidents involving firearms.

---

## Components 5–7 (Omitted from This KB Summary)

**Components 5, 6, and 7** are **not** included as separate numbered statistical sections in this document.

### Why they are omitted here
During research design, additional outcome areas were considered that align with **service capacity, rural and underserved access, and campus-related systems** under the 2022 reauthorization. For this knowledge-base summary, the team **does not present** standalone pre/post quantitative write-ups for those slots because:

- **Indicator and data alignment:** Constructing stable, comparable **state–year** measures that isolate those policy channels—separately from overlapping trends already captured in other components—did not meet the bar used for Components 1–4 and 8 in this abbreviated write-up.
- **Overlap with narrative policy docs:** Related themes remain described at a high level in `vawa_2022_reauthorization.md`, `vawa_overview.md`, and the components framework document `vawa_2022_components_and_research_design.md`. Omission here means **not separately tabulated in this file**, not “out of scope for VAWA.”

### Where the full analysis may still appear
Extended tables, appendices, or notebooks in the main project repository may carry additional detail. This chatbot KB focuses on **components with clear incident- or rate-based summaries** in `analysis_results.md` plus cross-component synthesis.

---

## Component 8: Has Incident Reporting Increased Over Time?

### Policy Context
This component is designed around a different policy logic: stronger systems may lead to **higher reporting**, even when violence itself is not increasing.

### Research Question
Did reported incidents increase in a way that is consistent with improved system response and reporting?

### Results
- **Total reported incidents increased**
  - Pre: **1,481,146**
  - Post: **3,811,064**
  - Change: **157.31%**
  - This result is descriptive only

- **Population-adjusted reporting rate increased, but not significantly in simple comparison**
  - Pre: **1372.53**
  - Post: **1526.27**
  - Change: **+11.2%**
  - p = **0.1809**

- **No systematic state-level increase**
  - **25 of 51 states increased** (**49%**)
  - Binomial test: **p = 0.61**

- **Within-state increase appears after controlling for state and year effects**
  - Fixed-effects model: **Post_2022 = +102.49, p = 0.0026**
  - Interpreted as **102 additional incidents per 100K women after 2022**
  - Project interpretation: about a **7.8% increase** in reporting rates after controlling for existing within-state and between-state trends

- **Trend was already rising before 2022**
  - 2021: **1125**
  - 2022: **1282**
  - 2023: **1396**
  - This suggests an upward trend, but not necessarily a sharp acceleration immediately after 2022

### Interpretation Notes
- This component is one of the strongest examples of why raw totals alone are not enough.
- The descriptive increase could reflect:
  - improved reporting systems
  - increased violence
  - broader reporting participation
  - or a combination of these factors
- The controlled model supports a statistically significant post-2022 increase in reporting rates, but the trend was already moving upward before 2022.

---

## Cross-Component State Performance

### Purpose
This analysis asks whether some states appear to be improving across multiple policy-relevant dimensions at the same time.

### Method Summary
- For each component, the team computed a **state-level change**.
- Each indicator was assigned a desired direction based on policy grounding or observed directional interpretation.
- State outcomes were converted into **binary improvement indicators**.
- These indicators were summed into an **improvement count**.
- A **standardized composite score** was then created using a z-score transformation.

### State Patterns
**Broad Improvement**
- High composite score and high improvement count
- Examples: **Vermont, New York**
- Interpretation: these states show multi-dimensional improvement across several indicators

**Targeted Improvement**
- High composite score but low improvement count
- Examples: **DC, South Dakota, California**
- Interpretation: strong movement in a smaller number of indicators

**Superficial Improvement**
- Low composite score but high improvement count
- Examples: **West Virginia, Texas**
- Interpretation: many improvements, but generally small in magnitude

**Deterioration**
- Low composite score and low improvement count
- Examples: **Florida, Mississippi**
- Interpretation: weak or negative movement across multiple indicators

---

## Overall Takeaways

- **Component 1** shows the clearest evidence of post-2022 change, with significant declines in both sexual assault share and adjusted rate.
- **Component 2** does not show a statistically meaningful shift in Native American victim representation or tribal land incident rates.
- **Component 3** shows a significant and widespread decline in incidents involving unmarried partners, though interpretation is substantively complex.
- **Component 4** shows no detectable change in domestic violence incidents involving firearms.
- **Component 8** provides mixed but important evidence: descriptive reporting increases are hard to interpret, but controlled models suggest a significant rise in reporting rates after 2022.

---

## Caveats

- These are **observational findings**, not causal estimates.
- State reporting systems, participation, and coding practices may vary over time.
- Some policy goals are not directly observable in incident-level administrative data.
- A statistically significant result does not, by itself, prove policy effectiveness.
- A non-significant result does not prove that no real-world change occurred.

For definitions, see `glossary.md`.  
For methods, see `methodology.md`.  
For the research question and policy-to-component mapping, see `vawa_2022_components_and_research_design.md`.  
For tool and KB scope (including data gaps such as LGBTQ+ indicators), see `limitations_scope.md`.