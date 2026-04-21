---
doc_type: limitations_scope
title: Limitations and Scope of the VAWA Insights Bot and Knowledge Base
citation_id: KB-LIMITATIONS-SCOPE-001
years_covered: varies_by_dataset
geo_level: state, county, tribal (where available)
metric_tags: limitations, scope, methodology, nibrs, vawa, disclaimer
source: internal_kb
---

## Purpose

This document states what the **VAWA Insights** system is for, what it is **not**, and how to interpret outputs when combining **structured metrics**, **knowledge-base text**, and **internal analysis summaries**.

---

## Scope of the Tool

### In scope
- Exploratory questions about **defined metrics** (e.g., rankings, comparisons, time patterns, risk profiles) when matching rows exist in the structured dataset.
- **Context and caveats** from curated markdown: VAWA overview, 2022 reauthorization themes, NIBRS limitations, methodology, glossary, FAQ, and post-2022 **associational** analysis summaries.

### Out of scope
- **Legal advice** or case-specific guidance (eligibility, orders, prosecution outcomes).
- **Real-time** crime counts or official FBI releases (the bot uses **project** datasets and KB text, not a live government API).
- **Causal claims:** Policy changes and observed metric changes are **not** treated as proven cause–effect relationships in this design.
- **Complete coverage of every 2022 VAWA provision:** Some themes are narrative-only or omitted from quantitative components when incident-based data do not support a clean indicator (see `vawa_2022_components_and_research_design.md`).

---

## Data and Metric Limitations

- Metrics reflect **reported** incidents and **project-defined** constructs; they are **not** measures of true population prevalence.
- **Participation, coding, and NIBRS transition** affect trends; see `nibrs_limitations.md`.
- **Geography and years** may be missing for some queries; absence of a row is not evidence that nothing occurred.
- The **chat UI sample CSV** may not list every derivative used in long-form **analysis_results.md**; when numbers conflict, prefer the **citation** attached to the structured tool output for that question.

---

## Knowledge Base Limitations

- Retrieval uses **keyword overlap** on chunked markdown. Relevant passages may be missed or lightly ranked; an upgrade path is embedding-based RAG.
- Documents are **summaries and student/project framing**, not a substitute for statutes, regulations, or agency guidance.
- **Sources** are cited in individual KB files; links may change over time.

---

## Components Not Fully Represented in Quantitative KB

- **LGBTQ+ inclusive access** provisions under VAWA are **important policy context** but are **not** modeled with a dedicated indicator in this KB set because **administrative incident extracts used here do not provide reliable, standardized identity fields** suitable for the same state–year comparison framework. That is a **data limitation**, not a statement about policy importance.

---

## Responsible Use

- Use outputs for **learning, hypothesis generation, and transparency-minded reporting**; validate conclusions for any external publication with primary data and methods documentation.
- For emergencies or personal safety, use **911** and qualified victim services—not this bot.

---

## Related Documents

- `faq.md` — User-oriented questions and definitions.
- `methodology.md` — Study design and modeling (when populated).
- `nibrs_limitations.md` — Administrative data caveats.
- `vawa_2022_components_and_research_design.md` — Research question and policy-to-component mapping.
