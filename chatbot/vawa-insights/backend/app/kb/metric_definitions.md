---
doc_type: methodology
title: Metric Definitions (Derived from NIBRS Data)
citation_id: metric_definitions_nibrs
years_covered: varies_by_dataset
geo_level: state, county
metric_tags: [dv_rate, sexual_assault_share, sexual_assault_rate, firearm_share, dating_partner_share, native_victim_share, tribal_land_incident_rate, reporting_rate, risk_index]
source: NIBRS_DataDictionary
---

## Overview

This document defines the core derived metrics used in the VAWA Insights Bot. All metrics are constructed from incident-level data in the FBI’s National Incident-Based Reporting System (NIBRS).  

NIBRS captures structured information about incidents, offenses, victims, offenders, relationships, and weapons :contentReference[oaicite:0]{index=0}. These fields are aggregated into interpretable indicators for analysis.

All metrics reflect **reported incidents**, not true underlying prevalence.

---

## Metric Definitions

### 1. Sexual Assault Share (`sexual_assault_share`)

**Definition:**  
The proportion of all reported incidents that involve a sexual assault offense.

**Construction:**
- Numerator: Incidents with a sexual assault offense
- Denominator: Total incidents

**NIBRS fields used:**
- `NIBRS_OFFENSE.OFFENSE_ID`
- `NIBRS_OFFENSE.INCIDENT_ID`
- `NIBRS_OFFENSE_TYPE.OFFENSE_TYPE_ID`
- `NIBRS_OFFENSE_TYPE.OFFENSE_NAME`

**Interpretation:**
- Measures the composition of reported crime types  
- Does not reflect total prevalence of sexual assault

---

### 2. Sexual Assault Rate (`sexual_assault_rate`)

**Definition:**  
Population-adjusted rate of sexual assault incidents.

**Construction:**
- Numerator: Sexual assault incidents  
- Denominator: Population (external to NIBRS)

**NIBRS fields used:**
- Same offense fields as above (`OFFENSE_TYPE`)
- `INCIDENT_ID` for aggregation

**Interpretation:**
- Enables cross-state comparison  
- Still dependent on reporting patterns

---

### 3. Domestic Violence Rate (`dv_rate`)

**Definition:**  
Rate of incidents classified as domestic violence per population.

**Construction:**
- Identified using:
  - Offense types (e.g., assault)
  - Victim-offender relationship (intimate or family)
- Normalized by population

**NIBRS fields used:**
- `NIBRS_OFFENSE_TYPE.OFFENSE_NAME`
- `NIBRS_VICTIM_OFFENDER_REL.RELATIONSHIP_ID`
- `NIBRS_RELATIONSHIP.RELATIONSHIP_NAME`
- `NIBRS_VICTIM.VICTIM_ID`
- `NIBRS_INCIDENT.INCIDENT_ID`

**Interpretation:**
- Sensitive to both incident occurrence and reporting behavior  
- Depends on correct relationship classification

---

### 4. Firearm Share (`firearm_share`)

**Definition:**  
Proportion of incidents involving a firearm.

**Construction:**
- Numerator: Incidents involving a firearm  
- Denominator: Total incidents (or DV subset)

**NIBRS fields used:**
- `NIBRS_WEAPON.WEAPON_ID`
- `NIBRS_WEAPON_TYPE.WEAPON_CODE`
- `NIBRS_WEAPON_TYPE.WEAPON_NAME`
- `NIBRS_OFFENSE.OFFENSE_ID`
- `NIBRS_OFFENSE.INCIDENT_ID`

**Interpretation:**
- Measures firearm involvement in incidents  
- Does not measure firearm ownership or access

---

### 5. Dating / Unmarried Partner Share (`dating_partner_share`)

**Definition:**  
Proportion of domestic violence incidents involving nonmarried partners.

**Construction:**
- Numerator: DV incidents involving dating/unmarried partners  
- Denominator: All DV incidents

**NIBRS fields used:**
- `NIBRS_VICTIM_OFFENDER_REL.RELATIONSHIP_ID`
- `NIBRS_RELATIONSHIP.RELATIONSHIP_NAME`
- `NIBRS_VICTIM.VICTIM_ID`
- `NIBRS_INCIDENT.INCIDENT_ID`

**Interpretation:**
- Reflects relationship composition within DV incidents  
- Sensitive to reporting and classification differences

---

### 6. Native Victim Share (`native_victim_share`)

**Definition:**  
Proportion of victims identified as Native American.

**Construction:**
- Numerator: Victims with Native American race classification  
- Denominator: All victims

**NIBRS fields used:**
- `NIBRS_VICTIM.RACE_ID`
- `NIBRS_VICTIM.ETHNICITY_ID`
- `NIBRS_VICTIM.VICTIM_ID`
- Reference: race lookup table (`REF_RACE`)

**Interpretation:**
- Used to assess representation within reported incidents  
- Often compared to population share for context

---

### 7. Tribal Land Incident Rate (`tribal_land_incident_rate`)

**Definition:**  
Rate of incidents occurring in tribal jurisdictions.

**Construction:**
- Based on location or jurisdiction indicators  
- Normalized by population or total incidents

**NIBRS fields used:**
- `NIBRS_OFFENSE.LOCATION_ID`
- `NIBRS_LOCATION_TYPE.LOCATION_NAME`
- `NIBRS_INCIDENT.AGENCY_ID`

**Interpretation:**
- Dependent on how location/jurisdiction is encoded  
- May underrepresent incidents affecting tribal communities

---

### 8. Reporting Rate (`reporting_rate`)

**Definition:**  
Number of reported incidents per population.

**Construction:**
- Numerator: Total reported incidents  
- Denominator: Population (external)

**NIBRS fields used:**
- `NIBRS_INCIDENT.INCIDENT_ID`
- `NIBRS_MONTH.DATA_YEAR`
- `NIBRS_MONTH.MONTH_NUM`

**Interpretation:**
- Captures reporting activity and system engagement  
- Increases may reflect improved reporting, increased incidents, or both

---

### 9. Risk Index (`risk_index`)

**Definition:**  
Composite indicator summarizing multiple risk-related metrics.

**Construction:**
- Combines standardized (z-scored) metrics such as:
  - DV rate  
  - firearm share  
  - reporting indicators  
- Aggregated into a single score

**NIBRS fields used:**
- Derived from multiple metrics above (no single table)

**Interpretation:**
- Relative measure of risk across geographies  
- Sensitive to metric selection and weighting choices

---

## Interpretation Notes

- All metrics are based on **reported incidents**, not actual prevalence.
- Differences across states may reflect:
  - reporting practices  
  - agency participation  
  - classification differences  
- “Share” metrics describe **composition**, not total magnitude.
- “Rate” metrics allow comparison but depend on accurate population data.

---

## Caveats

- NIBRS coverage varies by jurisdiction and year.
- Underreporting is a known limitation, especially for sensitive crimes.
- Victim-offender relationships and demographic fields may be incomplete.
- Derived metrics depend on classification and filtering decisions made in this project.

For methodology, see `methodology.md`.  
For results, see `analysis_results.md`.