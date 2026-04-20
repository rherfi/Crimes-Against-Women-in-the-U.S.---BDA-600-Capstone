from pathlib import Path
import io
import contextlib

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


SCRIPT_DIR = Path(__file__).resolve().parent
STAT_ANALYSIS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = STAT_ANALYSIS_DIR.parent

DATA_PATH = PROJECT_ROOT / "aggregated_crime_and_census_data.csv"
OUTPUT_DIR = STAT_ANALYSIS_DIR / "output"
OUTPUT_FILE = OUTPUT_DIR / "C8_reporting_results.txt"


def run_analysis():
    df = pd.read_csv(DATA_PATH).copy()
    df["data_year"] = df["data_year"].astype(int)

    # Build panel
    df["state"] = df["state_name"]
    df["year"] = df["data_year"]

    df["total_incidents"] = pd.to_numeric(df["total_incidents"], errors="coerce")
    df["fem_pop"] = pd.to_numeric(df["fem_pop"], errors="coerce")
    df.loc[df["fem_pop"] <= 0, "fem_pop"] = np.nan

    df["incident_rate"] = df["total_incidents"] / df["fem_pop"]

    # *** Corrected definition ***
    df["post_2022"] = (df["year"] >= 2023).astype(int)

    panel = df[[
        "state", "year", "total_incidents",
        "fem_pop", "incident_rate", "post_2022"
    ]].copy()

    print(panel.head())

    # =========================
    # Q1: Yearly totals
    # =========================
    yearly = panel.groupby("year")["total_incidents"].sum()
    print("\nYearly totals:")
    print(yearly)

    # =========================
    # Q2: State-level changes
    # =========================
    state_means = (
        panel.groupby(["state", "post_2022"])["incident_rate"]
        .mean()
        .unstack()
        .dropna()
    )

    state_means["change"] = state_means[1] - state_means[0]

    print("\nState-level change summary:")
    print(state_means["change"].describe())

    print("\nMean change:", state_means["change"].mean())
    print("Median change:", state_means["change"].median())

    # =========================
    # Q3: Direction consistency
    # =========================
    prop_increase = (state_means["change"] > 0).mean()
    print("\nProportion of states with increase:", prop_increase)

    # =========================
    # Q4: Fixed effects model
    # =========================
    fe_df = panel.dropna(subset=["incident_rate"]).copy()

    model = smf.ols(
        "incident_rate ~ post_2022 + C(state) + C(year)",
        data=fe_df
    ).fit(cov_type="HC1")

    print("\nFE model:")
    print(model.summary())

    coef = model.params["post_2022"]
    pval = model.pvalues["post_2022"]
    ci = model.conf_int().loc["post_2022"]

    print("\nFE result:")
    print(f"Coefficient: {coef:.6f}")
    print(f"p-value: {pval:.6f}")
    print(f"95% CI: [{ci[0]:.6f}, {ci[1]:.6f}]")

    # =========================
    # Q5: Weighted model
    # =========================
    w_model = smf.wls(
        "incident_rate ~ post_2022 + C(state) + C(year)",
        data=fe_df,
        weights=fe_df["fem_pop"]
    ).fit(cov_type="HC1")

    print("\nWeighted FE model:")
    print(w_model.summary())

    coef = w_model.params["post_2022"]
    pval = w_model.pvalues["post_2022"]
    ci = w_model.conf_int().loc["post_2022"]

    print("\nWeighted FE result:")
    print(f"Coefficient: {coef:.6f}")
    print(f"p-value: {pval:.6f}")
    print(f"95% CI: [{ci[0]:.6f}, {ci[1]:.6f}]")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        run_analysis()

    results_text = buffer.getvalue()

    print(results_text, end="")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(results_text)

    print(f"\nSaved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()