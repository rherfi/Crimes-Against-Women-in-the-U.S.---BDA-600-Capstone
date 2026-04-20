from pathlib import Path
import io
import contextlib

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
STAT_ANALYSIS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = STAT_ANALYSIS_DIR.parent

DATA_PATH = PROJECT_ROOT / "aggregated_crime_and_census_data.csv"
OUTPUT_DIR = STAT_ANALYSIS_DIR / "output"
OUTPUT_FILE = OUTPUT_DIR / "C4_DV_firearm_results.txt"


def run_analysis():
    # load data
    analysis_df = pd.read_csv(DATA_PATH).copy()

    # Use existing variable
    analysis_df["dv_firearm_unmarried"] = pd.to_numeric(
        analysis_df["firearm_nonmarried"], errors="coerce"
    ).fillna(0)

    # Correct policy timing (2023+)
    analysis_df["post_policy"] = (analysis_df["data_year"] >= 2023).astype(int)

    # Clean population
    analysis_df["fem_pop"] = pd.to_numeric(analysis_df["fem_pop"], errors="coerce")
    analysis_df.loc[analysis_df["fem_pop"] <= 0, "fem_pop"] = np.nan

    # Rate per 100k
    analysis_df["dv_firearm_unmarried_rate"] = (
        analysis_df["dv_firearm_unmarried"] / analysis_df["fem_pop"]
    ) * 100000

    # Quick check
    print(analysis_df[[
        "state_name", "data_year",
        "dv_firearm_unmarried",
        "dv_firearm_unmarried_rate"
    ]].head())

    # q1 Did the overall total number of incidents involving both DV and a firearm change after 2022?

    # Q1 descriptive summary
    q1_summary = (
        analysis_df.groupby("post_policy", as_index=False)
        .agg(
            mean_incidents=("dv_firearm_unmarried", "mean"),
            median_incidents=("dv_firearm_unmarried", "median"),
            total_incidents=("dv_firearm_unmarried", "sum"),
            n_state_years=("dv_firearm_unmarried", "size")
        )
    )

    print("Q1 descriptive summary:")
    print(q1_summary)

    # Q1 fixed effects model
    q1_df = analysis_df[
        ["dv_firearm_unmarried", "post_policy", "state_name", "data_year"]
    ].dropna().copy()

    q1_model = smf.ols(
        "dv_firearm_unmarried ~ post_policy + C(state_name) + C(data_year)",
        data=q1_df
    ).fit(cov_type="cluster", cov_kwds={"groups": q1_df["state_name"]})

    print("\nQ1 FE model:")
    print(q1_model.summary())

    print("\nQ1 post_policy coefficient:")
    print(q1_model.params["post_policy"])
    print("p-value:", q1_model.pvalues["post_policy"])
    print("95% CI:", q1_model.conf_int().loc["post_policy"].tolist())

    # q2 Is this effect widespread or driven by a few states?

    # State-level pre/post means for adjusted rate
    state_changes = (
        analysis_df.groupby(["state_name", "post_policy"], as_index=False)["dv_firearm_unmarried_rate"]
        .mean()
        .pivot(index="state_name", columns="post_policy", values="dv_firearm_unmarried_rate")
        .reset_index()
        .rename(columns={0: "pre_rate", 1: "post_rate"})
    )

    state_changes["change_rate"] = state_changes["post_rate"] - state_changes["pre_rate"]

    # Sort for inspection
    state_changes = state_changes.sort_values("change_rate", ascending=False).reset_index(drop=True)

    print("Top 10 increases:")
    print(state_changes.head(10))

    print("\nTop 10 decreases:")
    print(state_changes.tail(10))

    # Summary counts
    n_states = len(state_changes)
    n_increase = (state_changes["change_rate"] > 0).sum()
    n_decrease = (state_changes["change_rate"] < 0).sum()
    n_no_change = (state_changes["change_rate"] == 0).sum()

    print("\nStates with increase:", f"{n_increase} / {n_states} ({(n_increase/n_states)*100:.1f}%)")
    print("States with decrease:", f"{n_decrease} / {n_states} ({(n_decrease/n_states)*100:.1f}%)")
    print("States with no change:", f"{n_no_change} / {n_states} ({(n_no_change/n_states)*100:.1f}%)")

    print("\nMean change (adjusted rate, per 100k):", state_changes["change_rate"].mean())
    print("Median change (adjusted rate, per 100k):", state_changes["change_rate"].median())

    # Optional concentration check
    positive_changes = state_changes[state_changes["change_rate"] > 0].copy()

    if len(positive_changes) > 0:
        total_positive = positive_changes["change_rate"].sum()
        top5_share = positive_changes.head(5)["change_rate"].sum() / total_positive
        print("\nTop 5 states share of total increase:", round(top5_share, 3))

    # q3 Within states, did the rate of incidents involving both DV and a firearm change after 2022?

    # Q3 descriptive summary
    q3_summary = (
        analysis_df.groupby("post_policy", as_index=False)
        .agg(
            mean_rate=("dv_firearm_unmarried_rate", "mean"),
            median_rate=("dv_firearm_unmarried_rate", "median"),
            n_state_years=("dv_firearm_unmarried_rate", "size")
        )
    )

    print("Q3 descriptive summary:")
    print(q3_summary)

    # Q3 fixed effects model
    q3_df = analysis_df[
        ["dv_firearm_unmarried_rate", "post_policy", "state_name", "data_year"]
    ].dropna().copy()

    q3_model = smf.ols(
        "dv_firearm_unmarried_rate ~ post_policy + C(state_name) + C(data_year)",
        data=q3_df
    ).fit(cov_type="cluster", cov_kwds={"groups": q3_df["state_name"]})

    print("\nQ3 FE model:")
    print(q3_model.summary())

    print("\nQ3 post_policy coefficient:")
    print(q3_model.params["post_policy"])
    print("p-value:", q3_model.pvalues["post_policy"])
    print("95% CI:", q3_model.conf_int().loc["post_policy"].tolist())

    paired_t = stats.ttest_rel(
        state_changes["post_rate"],
        state_changes["pre_rate"],
        nan_policy="omit"
    )

    wilcoxon_res = stats.wilcoxon(
        state_changes["post_rate"],
        state_changes["pre_rate"]
    )

    print("Paired t-test p-value:", paired_t.pvalue)
    print("Wilcoxon p-value:", wilcoxon_res.pvalue)


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