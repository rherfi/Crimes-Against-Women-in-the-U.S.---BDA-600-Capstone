---
doc_type: research_design
title: VAWA 2022 Components, Research Question, and Analytic Mapping
citation_id: KB-VAWA-COMPONENTS-FRAMEWORK-001
years_covered: pre_2022_vs_post_2022
geo_level: state
metric_tags: vawa, 2022, reauthorization, research_question, components, policy, methodology
source: internal_kb
---

## Research Question

**Are the patterns of crimes against women in the U.S. consistent with improvement on the components targeted by the 2022 reauthorization of VAWA?**

This question is addressed using **observable administrative and incident-derived patterns** at primarily **state level**, comparing **pre-2022** and **post-2022** periods. “Consistent with improvement” means **directionally aligned changes** (or stability where stability is the meaningful benchmark) on indicators chosen to **mirror** major reauthorization themes—not proof that VAWA **caused** those changes.

---

## Key Components of the 2022 Violence Against Women Act (Policy Themes)

The 2022 reauthorization emphasizes multiple interlocking areas. The list below is a **policy organizing frame** used alongside statistical components in this project.

1. **Sexual assault prevention and response** — Reauthorized federal funding for prevention and response programs related to sexual assault.
2. **Tribal jurisdiction** — Expanded Native American tribal jurisdiction to allow more prosecutorial authority over certain offenses involving non-Native offenders (subject to statutory scope).
3. **Dating and nonmarried partners** — Expanded domestic violence protections to include **nonmarried** intimate partners where statutory definitions apply.
4. **Firearms** — Restricted certain individuals with qualifying domestic violence convictions from possessing firearms (implementation varies by jurisdiction and enforcement).
5. **Victim services and housing** — Increased funding for victim services, including **shelters** and **transitional housing**.
6. **Rural and underserved communities** — Strengthened victim services in **rural** and **underserved** communities.
7. **Campus** — Strengthened victim services and **prevention** on **college campuses**.
8. **Reporting and response** — Improved **reporting** and **response** services for victims (systems-level expectation; may show up as reporting intensity or volume in data).
9. **Implementation** — Much of VAWA operates through **federal grants** and state-level implementation; observed patterns may lag statutory changes and depend on funding flows not directly visible in incident extracts.

---

## How Policy Themes Map to Quantitative “Components” in This Project

The detailed statistical write-up in `analysis_results.md` uses **numbered components**. The mapping is **conceptual** (which policy pillar each component was designed to speak to), not a one-to-one match to every statutory subsection.

| Policy theme (2022 VAWA) | Primary quantitative component in `analysis_results.md` |
|--------------------------|-------------------------------------------------------------|
| Sexual assault prevention/response | **Component 1** — Rates / share of sexual assault |
| Tribal jurisdiction / Native victims / tribal geography | **Component 2** — Native American victim representation; tribal land–related rates |
| Protections for nonmarried / dating partners | **Component 3** — Victim–offender relationship patterns (unmarried partner share) |
| Firearm prohibitions linked to domestic violence | **Component 4** — Domestic violence incidents involving a firearm |
| Reporting and response systems | **Component 8** — Reporting volume and population-adjusted reporting rates (with strong caveats on composition vs true prevalence) |

**Components 5–7** in the project numbering are **reserved** for additional service-, rural/underserved-, and campus-aligned outcomes where the team did **not** publish separate statistical sections in this KB file. See the **“Components 5–7 (Omitted from This KB Summary)”** section in `analysis_results.md` for the rationale.

---

## Key Component Left Out Due to Lack of Relevant Data

### LGBTQ+ inclusive access

The 2022 reauthorization includes **expanded protections for LGBTQ+ survivors** and expectations of **inclusive access** to federally funded programs.

**Why it is not a standalone quantitative component here**

Incident-based administrative data used for state–year modeling **do not provide reliable, standardized fields** for sexual orientation and gender identity across agencies and years in a way that supports the same comparative design as other components. Treating this as a **data gap** is important:

- It does **not** mean the policy theme is unimportant.
- It means this KB and the summarized analysis **cannot** fairly score “improvement” on that dimension from these extracts alone.

Narrative context may still appear in policy overview documents; see also `limitations_scope.md`.

---

## How to Read Results Alongside This Framework

- **Statistical significance** does not prove **policy effectiveness**; **non-significance** does not prove absence of real-world change.
- **Multiple channels** (reporting, classification, coverage) can move the same metric; see `nibrs_limitations.md` and (when populated) `methodology.md`.
- For **definitions of NIBRS fields**, use `glossary.md`. For **tool-facing metric names**, align with the structured dataset column names and `metric_definitions.md` when present.
