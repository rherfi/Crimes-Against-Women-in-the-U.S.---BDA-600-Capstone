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
OUTPUT_FILE = OUTPUT_DIR / "C1_sexassault_results.txt"


def detect_outliers_iqr(series):
    series = series.dropna()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper


def run_analysis():
    # load data
    df = pd.read_csv(DATA_PATH)

    # load metrics
    # "After 2022" = 2023 and later
    df["post_2022"] = (df["data_year"] >= 2023).astype(int)

    # Main analogous outcome: share of sexual assaults among all incidents
    df["share_sex_assault"] = df["sex_assaults"] / df["total_incidents"]

    # Avoid unstable shares when denominator is tiny
    df.loc[df["total_incidents"] < 10, "share_sex_assault"] = np.nan

    # Optional rate version for robustness
    df["sex_assault_rate"] = df["sex_assaults"] / df["fem_pop"]
    df.loc[df["fem_pop"] <= 0, "sex_assault_rate"] = np.nan

    # outlier detection
    lower_share, upper_share = detect_outliers_iqr(df["share_sex_assault"])
    outliers_share = df[
        (df["share_sex_assault"] < lower_share) |
        (df["share_sex_assault"] > upper_share)
    ]

    print("Share outliers:")
    print(
        outliers_share[["state_name", "data_year", "share_sex_assault"]]
        .sort_values(by="share_sex_assault", ascending=False)
        .head(20)
    )

    # q1 Did the overall share of sexual assault victims increase after 2022?
    # Two-way fixed effects model:
    # share_sex_assault ~ post_2022 + state FE + year FE
    q1_model = smf.ols(
        "share_sex_assault ~ post_2022 + C(state_name) + C(data_year)",
        data=df
    ).fit(cov_type="cluster", cov_kwds={"groups": df["state_name"]})

    print(q1_model.summary())

    # Extract the coefficient of interest
    q1_coef = q1_model.params.get("post_2022", np.nan)
    q1_pval = q1_model.pvalues.get("post_2022", np.nan)
    q1_ci = (
        q1_model.conf_int().loc["post_2022"]
        if "post_2022" in q1_model.params.index
        else [np.nan, np.nan]
    )

    print("\nQ1 result:")
    print(f"post_2022 coefficient: {q1_coef:.6f}")
    print(f"p-value: {q1_pval:.6f}")
    print(f"95% CI: [{q1_ci[0]:.6f}, {q1_ci[1]:.6f}]")

    # q2 Is the trend consistent across states, or limited to a few?
    # Compute each state's mean pre-2023 and post-2022 share
    state_pre_post = (
        df.groupby(["state_name", "post_2022"])["share_sex_assault"]
          .mean()
          .unstack()
          .rename(columns={0: "pre_mean_share", 1: "post_mean_share"})
          .reset_index()
    )

    # Keep only states that have both pre and post values
    state_pre_post = state_pre_post.dropna(
        subset=["pre_mean_share", "post_mean_share"]
    ).copy()

    # State-level change
    state_pre_post["change_share"] = (
        state_pre_post["post_mean_share"] - state_pre_post["pre_mean_share"]
    )

    # Sort to see where increases/decreases are concentrated
    state_changes_sorted = state_pre_post.sort_values("change_share", ascending=False)

    print("Top states with largest increases:")
    print(state_changes_sorted.head(10))

    print("\nTop states with largest decreases:")
    print(state_changes_sorted.tail(10))

    # Count how many states increased vs decreased
    num_increase = (state_pre_post["change_share"] > 0).sum()
    num_decrease = (state_pre_post["change_share"] < 0).sum()
    num_no_change = (state_pre_post["change_share"] == 0).sum()
    total_states = len(state_pre_post)

    print("\nState-level consistency summary:")
    print(f"States with increase: {num_increase}")
    print(f"States with decrease: {num_decrease}")
    print(f"States with no change: {num_no_change}")
    print(f"Total states in comparison: {total_states}")

    print("\nDistribution of state-level changes:")
    print(state_pre_post["change_share"].describe())

    # q3: Did the share of sexual assault victims increase within states after 2022?
    # Using the state-level pre/post means from Q2
    # Paired t-test and Wilcoxon signed-rank test
    pre_vals = state_pre_post["pre_mean_share"]
    post_vals = state_pre_post["post_mean_share"]

    # Paired t-test
    t_stat, t_p = stats.ttest_rel(post_vals, pre_vals, nan_policy="omit")

    # Wilcoxon signed-rank test
    # Use try/except in case all differences are zero or sample is too small
    try:
        w_stat, w_p = stats.wilcoxon(post_vals, pre_vals)
    except ValueError as e:
        w_stat, w_p = np.nan, np.nan
        print("Wilcoxon test could not be computed:", e)

    print("Q3 within-state tests:")
    print(f"Paired t-test statistic: {t_stat:.6f}, p-value: {t_p:.6f}")
    print(f"Wilcoxon statistic: {w_stat}, p-value: {w_p}")

    # Also report the average within-state change
    mean_change = state_pre_post["change_share"].mean()
    median_change = state_pre_post["change_share"].median()

    print(f"\nAverage within-state change: {mean_change:.6f}")
    print(f"Median within-state change: {median_change:.6f}")

    # Absolute measure: sexual assault rate per female population
    df["sex_assault_rate"] = df["sex_assaults"] / df["fem_pop"]

    # Clean invalid values
    df.loc[df["fem_pop"] <= 0, "sex_assault_rate"] = np.nan
    df = df.dropna(subset=["sex_assault_rate"]).copy()

    # Regression for absolute level
    rate_model = smf.ols(
        "sex_assault_rate ~ post_2022 + C(state_name) + C(data_year)",
        data=df
    ).fit(cov_type="cluster", cov_kwds={"groups": df["state_name"]})

    print(rate_model.summary())

    # Extract key result
    rate_coef = rate_model.params.get("post_2022", np.nan)
    rate_pval = rate_model.pvalues.get("post_2022", np.nan)
    rate_ci = rate_model.conf_int().loc["post_2022"]

    print("\nAbsolute rate result:")
    print(f"post_2022 coefficient: {rate_coef:.6f}")
    print(f"p-value: {rate_pval:.6f}")
    print(f"95% CI: [{rate_ci[0]:.6f}, {rate_ci[1]:.6f}]")


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