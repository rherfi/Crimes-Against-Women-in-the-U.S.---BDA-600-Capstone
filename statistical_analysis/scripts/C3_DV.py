from pathlib import Path
import io
import contextlib

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import zscore


SCRIPT_DIR = Path(__file__).resolve().parent
STAT_ANALYSIS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = STAT_ANALYSIS_DIR.parent

DATA_PATH = PROJECT_ROOT / "aggregated_crime_and_census_data.csv"
OUTPUT_DIR = STAT_ANALYSIS_DIR / "output"
OUTPUT_FILE = OUTPUT_DIR / "C3_DV_results.txt"


def detect_outliers_iqr(series):
    s = series.dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (series < lower) | (series > upper)


def run_analysis():
    df = pd.read_csv(DATA_PATH).copy()
    df["data_year"] = df["data_year"].astype(int)

    # --- Base cleaning ---
    df = df[df["fem_pop"] > 0].copy()

    # --- Derived variables ---
    df["share_nonmarried_dv"] = (
        df["victim_offender_nonmarried_partner"] / df["dv_total"]
    )

    df["nonmarried_to_spouse_ratio"] = (
        df["victim_offender_nonmarried_partner"] /
        df["victim_offender_rel_spouse"]
    )

    # --- Denominator guards ---
    df.loc[df["dv_total"] <= 0, "share_nonmarried_dv"] = np.nan
    df.loc[df["dv_total"] < 10, "share_nonmarried_dv"] = np.nan

    df.loc[df["victim_offender_rel_spouse"] <= 0, "nonmarried_to_spouse_ratio"] = np.nan
    df.loc[df["victim_offender_rel_spouse"] < 10, "nonmarried_to_spouse_ratio"] = np.nan

    df["post_2022"] = (df["data_year"] >= 2023).astype(int)

    # --- Outliers ---
    df["outlier_share"] = detect_outliers_iqr(df["share_nonmarried_dv"])
    df["outlier_ratio"] = detect_outliers_iqr(df["nonmarried_to_spouse_ratio"])

    df["z_share"] = zscore(df["share_nonmarried_dv"], nan_policy="omit")
    df["z_ratio"] = zscore(df["nonmarried_to_spouse_ratio"], nan_policy="omit")

    df["outlier_share_z"] = df["z_share"].abs() > 3
    df["outlier_ratio_z"] = df["z_ratio"].abs() > 3

    print("Share outliers:")
    print(
        df[df["outlier_share"] | df["outlier_share_z"]]
        [["state_name", "data_year", "share_nonmarried_dv"]]
        .sort_values(by="share_nonmarried_dv", ascending=False)
        .head(20)
    )

    print("\nRatio outliers:")
    print(
        df[df["outlier_ratio"] | df["outlier_ratio_z"]]
        [["state_name", "data_year", "nonmarried_to_spouse_ratio"]]
        .sort_values(by="nonmarried_to_spouse_ratio", ascending=False)
        .head(20)
    )

    # =========================
    # Q1 REGRESSION (KEEP 2022)
    # =========================
    q1_df = df.dropna(subset=["share_nonmarried_dv"]).copy()

    q1_model = smf.ols(
        "share_nonmarried_dv ~ post_2022 + C(state_name) + C(data_year)",
        data=q1_df
    ).fit(cov_type="cluster", cov_kwds={"groups": q1_df["state_name"]})

    print(q1_model.summary())

    coef = q1_model.params["post_2022"]
    pval = q1_model.pvalues["post_2022"]
    ci = q1_model.conf_int().loc["post_2022"]

    print("\nQ1 result:")
    print(f"post_2022 coefficient: {coef:.6f}")
    print(f"p-value: {pval:.6f}")
    print(f"95% CI: [{ci[0]:.6f}, {ci[1]:.6f}]")

    # =========================
    # Q1 STATE SUMMARY (DROP 2022 + dropna)
    # =========================
    summary_df = df[df["data_year"] != 2022].copy()
    summary_df = summary_df.dropna(subset=["share_nonmarried_dv"])

    q1_state = (
        summary_df.groupby(["state_name", "post_2022"])["share_nonmarried_dv"]
        .mean()
        .unstack()
        .dropna()
    )

    q1_state["change_share"] = q1_state[1] - q1_state[0]

    print(q1_state.sort_values("change_share", ascending=False).head(15))
    print(q1_state.sort_values("change_share").head(15))
    print(q1_state["change_share"].describe())

    num_inc = (q1_state["change_share"] > 0).sum()
    num_dec = (q1_state["change_share"] < 0).sum()

    print(f"States with increase: {num_inc}")
    print(f"States with decrease: {num_dec}")
    print(f"Total states with usable data: {len(q1_state)}")

    t_stat, p_val = stats.ttest_rel(q1_state[1], q1_state[0])
    print(f"Paired t-test: t = {t_stat:.4f}, p = {p_val:.4f}")

    # =========================
    # Q3 REGRESSION (KEEP 2022)
    # =========================
    q3_df = df.dropna(subset=["nonmarried_to_spouse_ratio"]).copy()

    q3_model = smf.ols(
        "nonmarried_to_spouse_ratio ~ post_2022 + C(state_name) + C(data_year)",
        data=q3_df
    ).fit(cov_type="cluster", cov_kwds={"groups": q3_df["state_name"]})

    print(q3_model.summary())

    coef = q3_model.params["post_2022"]
    pval = q3_model.pvalues["post_2022"]
    ci = q3_model.conf_int().loc["post_2022"]

    print("\nQ3 ratio regression result:")
    print(f"post_2022 coefficient: {coef:.6f}")
    print(f"p-value: {pval:.6f}")
    print(f"95% CI: [{ci[0]:.6f}, {ci[1]:.6f}]")

    # =========================
    # Q3 STATE SUMMARY (DROP 2022 + dropna)
    # =========================
    summary_df = df[df["data_year"] != 2022].copy()
    summary_df = summary_df.dropna(subset=["nonmarried_to_spouse_ratio"])

    q3_state = (
        summary_df.groupby(["state_name", "post_2022"])["nonmarried_to_spouse_ratio"]
        .mean()
        .unstack()
        .dropna()
    )

    q3_state["change_ratio"] = q3_state[1] - q3_state[0]

    print(q3_state.sort_values("change_ratio", ascending=False).head(15))
    print(q3_state.sort_values("change_ratio").head(15))
    print(q3_state["change_ratio"].describe())

    num_inc = (q3_state["change_ratio"] > 0).sum()
    num_dec = (q3_state["change_ratio"] < 0).sum()

    print(f"States with ratio increase: {num_inc}")
    print(f"States with ratio decrease: {num_dec}")
    print(f"Total states with usable data: {len(q3_state)}")

    t_stat, p_val = stats.ttest_rel(q3_state[1], q3_state[0])
    print(f"Paired t-test: t = {t_stat:.4f}, p = {p_val:.4f}")

    w_stat, w_p = stats.wilcoxon(q3_state[1], q3_state[0])
    print(f"Wilcoxon signed-rank test: W = {w_stat:.4f}, p = {w_p:.4f}")


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