from pathlib import Path
import io
import contextlib

import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, binomtest


SCRIPT_DIR = Path(__file__).resolve().parent
STAT_ANALYSIS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = STAT_ANALYSIS_DIR.parent

DATA_PATH = PROJECT_ROOT / "aggregated_crime_and_census_data.csv"
OUTPUT_DIR = STAT_ANALYSIS_DIR / "output"
OUTPUT_FILE = OUTPUT_DIR / "C2_NatAmer_results.txt"


def run_analysis():
    # load data
    df = pd.read_csv(DATA_PATH)

    # Ensure correct types
    df["data_year"] = df["data_year"].astype(int)

    # Drop rows where key denominators are missing or zero
    df = df[(df["total_incidents"] > 0) & (df["nat_amer_pop"] > 0)].copy()

    # exclude 2022 - treat as transition year
    df = df[df["data_year"] != 2022].copy()

    df["post_2022"] = (df["data_year"] >= 2023).astype(int)

    # Q1/Q3 working copy excluding Florida
    df_no_fl = df[df["state_name"] != "Florida"].copy()

    print(df_no_fl["state_name"].nunique())
    print(sorted(df_no_fl["data_year"].unique()))

    # recreate metrics with new florida-less data
    # Q1 variable: Native American share of victims
    df_no_fl["nat_amer_victim_share"] = np.where(
        df_no_fl["total_incidents"] > 0,
        df_no_fl["nat_amer_victims"] / df_no_fl["total_incidents"],
        np.nan,
    )

    # Q3 variable: overrepresentation ratio
    df_no_fl["nat_amer_overrepresentation_ratio"] = np.where(
        df_no_fl["nat_amer_pop"] > 0,
        df_no_fl["nat_amer_victim_share"] / df_no_fl["nat_amer_pop"],
        np.nan,
    )

    # Optional check
    print(df_no_fl[[
        "nat_amer_victim_share",
        "nat_amer_pop",
        "nat_amer_overrepresentation_ratio"
    ]].head())

    # q1 Did the share of victims who are Native American increase within states after 2022?
    # Overall pre/post means
    q1 = df_no_fl.groupby("post_2022")["nat_amer_victim_share"].mean()
    print("Q1 overall means:")
    print(q1)

    # State-level paired comparison
    state_q1 = (
        df_no_fl.groupby(["state_name", "post_2022"])["nat_amer_victim_share"]
        .mean()
        .unstack()
    )

    # Keep only states with both pre and post values
    state_q1 = state_q1.dropna(subset=[0, 1])

    q1_ttest = ttest_rel(state_q1[1], state_q1[0], nan_policy="omit")

    print("\nQ1 paired t-test:")
    print(q1_ttest)

    print("\nQ1 state-level change summary:")
    state_q1["change"] = state_q1[1] - state_q1[0]
    print(state_q1["change"].describe())

    # Q2 direction test
    num_increase = (state_q1["change"] > 0).sum()
    num_total_q2 = state_q1["change"].notna().sum()
    q2_direction_test = binomtest(num_increase, num_total_q2, p=0.5)

    print(f"States with increase: {num_increase}/{num_total_q2}")
    print("Q2 binomial test:")
    print(q2_direction_test)

    print("\nLargest increases:")
    print(state_q1["change"].sort_values(ascending=False).head(10))

    print("\nLargest decreases:")
    print(state_q1["change"].sort_values().head(10))

    # q3 Did Native Americans become more overrepresented among victims
    # relative to their population share after 2022?
    # Overall pre/post means
    q3 = df_no_fl.groupby("post_2022")["nat_amer_overrepresentation_ratio"].mean()
    print("Q3 overall means:")
    print(q3)

    # State-level paired comparison
    state_q3 = (
        df_no_fl.groupby(["state_name", "post_2022"])["nat_amer_overrepresentation_ratio"]
        .mean()
        .unstack()
    )

    # Keep only states with both pre and post values
    state_q3 = state_q3.dropna(subset=[0, 1])

    q3_ttest = ttest_rel(state_q3[1], state_q3[0], nan_policy="omit")

    print("\nQ3 paired t-test:")
    print(q3_ttest)

    print("\nQ3 state-level change summary:")
    state_q3["change"] = state_q3[1] - state_q3[0]
    print(state_q3["change"].describe())

    # q4 did incidents on tribal lands decrease after 2022?
    # q4 setup
    df_q4 = df.copy()

    df_q4["tribal_incident_rate"] = np.where(
        df_q4["total_incidents"] > 0,
        df_q4["on_tribal_lands"] / df_q4["total_incidents"],
        np.nan,
    )

    print(df_q4[[
        "state_name",
        "data_year",
        "on_tribal_lands",
        "total_incidents",
        "tribal_incident_rate"
    ]].head())

    # Overall pre/post means
    q4 = df_q4.groupby("post_2022")["tribal_incident_rate"].mean()
    print("Q4 overall means:")
    print(q4)

    # State-level paired comparison
    state_q4 = (
        df_q4.groupby(["state_name", "post_2022"])["tribal_incident_rate"]
        .mean()
        .unstack()
    )

    # Keep only states with both pre and post values
    state_q4 = state_q4.dropna(subset=[0, 1])

    q4_ttest = ttest_rel(state_q4[1], state_q4[0], nan_policy="omit")

    print("\nQ4 paired t-test:")
    print(q4_ttest)

    state_q4["change"] = state_q4[1] - state_q4[0]

    num_decrease = (state_q4["change"] < 0).sum()
    num_total_q4 = state_q4["change"].notna().sum()

    q4_direction_test = binomtest(num_decrease, num_total_q4, p=0.5)

    print(f"\nStates with decrease: {num_decrease}/{num_total_q4}")
    print("Q4 direction binomial test:")
    print(q4_direction_test)

    print("\nQ4 state-level change summary:")
    print(state_q4["change"].describe())

    # compact output block
    print("=== Q1 ===")
    print(q1)
    print(q1_ttest)

    print("\n=== Q2 ===")
    print(f"States with increase: {num_increase}/{num_total_q2}")
    print(q2_direction_test)

    print("\n=== Q3 ===")
    print(q3)
    print(q3_ttest)

    print("\n=== Q4 ===")
    print(q4)
    print(q4_ttest)
    print(f"States with decrease: {num_decrease}/{num_total_q4}")
    print(q4_direction_test)


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