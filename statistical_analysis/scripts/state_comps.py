from pathlib import Path
import io
import contextlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
STAT_ANALYSIS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = STAT_ANALYSIS_DIR.parent

DATA_PATH = PROJECT_ROOT / "aggregated_crime_and_census_data.csv"
OUTPUT_DIR = STAT_ANALYSIS_DIR / "output"
OUTPUT_FILE = OUTPUT_DIR / "state_comps_results.txt"
PLOT_FILE = OUTPUT_DIR / "state_comps_scatter.png"


def safe_divide(numerator, denominator):
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    return np.where(denominator > 0, numerator / denominator, np.nan)


def compute_metrics(d):
    out = pd.DataFrame(index=d.index)

    # Directional metrics
    out["sa_rate"] = safe_divide(d["sex_assaults"], d["fem_pop"]) * 100000
    out["unmarried_dv_share"] = safe_divide(
        d["victim_offender_nonmarried_partner"], d["dv_total"]
    )
    out["firearm_dv_share"] = safe_divide(
        d["firearm_nonmarried"], d["victim_offender_nonmarried_partner"]
    )
    out["reporting_rate"] = safe_divide(d["total_incidents"], d["fem_pop"]) * 100000

    # Non-directional / system-activation metrics
    out["nat_amer_victim_rate"] = safe_divide(
        d["nat_amer_victims"], d["nat_amer_pop"]
    ) * 100000
    out["tribal_land_rate"] = safe_divide(
        d["on_tribal_lands"], d["fem_pop"]
    ) * 100000

    return out


def run_analysis():
    df = pd.read_csv(DATA_PATH).copy()
    df["data_year"] = df["data_year"].astype(int)

    # Match notebook split exactly:
    # pre = <= 2021, post = 2023 and 2024 only
    pre_df = df[df["data_year"] <= 2021].copy()
    post_df = df[df["data_year"].isin([2023, 2024])].copy()

    pre_metrics = compute_metrics(pre_df)
    post_metrics = compute_metrics(post_df)

    pre_state = pre_metrics.groupby(pre_df["state_name"]).mean()
    post_state = post_metrics.groupby(post_df["state_name"]).mean()

    delta = post_state - pre_state
    delta.columns = [f"{c}_delta" for c in delta.columns]

    results = delta.copy()

    negative_metrics = [
        "sa_rate_delta",
        "unmarried_dv_share_delta",
        "firearm_dv_share_delta",
    ]
    positive_metrics = [
        "reporting_rate_delta",
    ]
    nondirectional_metrics = [
        "nat_amer_victim_rate_delta",
        "tribal_land_rate_delta",
    ]

    # Binary directional improvement count
    results["improvement_count"] = 0
    for col in negative_metrics:
        results["improvement_count"] += (results[col] < 0).astype(int)
    for col in positive_metrics:
        results["improvement_count"] += (results[col] > 0).astype(int)

    # Composite score
    z_df = pd.DataFrame(index=results.index)

    for col in negative_metrics:
        std = results[col].std()
        if pd.notna(std) and std != 0:
            z_df[col] = -((results[col] - results[col].mean()) / std)
        else:
            z_df[col] = np.nan

    for col in positive_metrics:
        std = results[col].std()
        if pd.notna(std) and std != 0:
            z_df[col] = (results[col] - results[col].mean()) / std
        else:
            z_df[col] = np.nan

    for col in nondirectional_metrics:
        std = results[col].std()
        if pd.notna(std) and std != 0:
            z_df[col] = abs((results[col] - results[col].mean()) / std)
        else:
            z_df[col] = np.nan

    results["composite_score"] = z_df.mean(axis=1, skipna=True)

    results["rank_count"] = results["improvement_count"].rank(
        ascending=False, method="min"
    )
    results["rank_composite"] = results["composite_score"].rank(
        ascending=False, method="min"
    )

    results = results.sort_values("composite_score", ascending=False)

    print("Top 10 states (composite score):")
    print(results[["improvement_count", "composite_score"]].head(10))

    print("\nBottom 10 states:")
    print(results[["improvement_count", "composite_score"]].tail(10))

    print("\nBinary improvement metrics used:")
    print([c.replace("_delta", "") for c in negative_metrics + positive_metrics])

    print("\nNon-directional composite-only metrics used:")
    print([c.replace("_delta", "") for c in nondirectional_metrics])

    # The notebook continues with state-type classification and a scatter plot
    mean_count = results["improvement_count"].mean()

    def classify_state(row):
        if row["composite_score"] > 0 and row["improvement_count"] >= mean_count:
            return "Broad Improvement"
        elif row["composite_score"] > 0 and row["improvement_count"] < mean_count:
            return "Targeted Improvement"
        elif row["composite_score"] <= 0 and row["improvement_count"] >= mean_count:
            return "Superficial Improvement"
        else:
            return "Deterioration"

    results["state_type"] = results.apply(classify_state, axis=1)

    print(results["state_type"].value_counts())

    np.random.seed(42)
    plt.figure(figsize=(11, 8))

    for state_type, group in results.groupby("state_type"):
        y_jitter = group["improvement_count"] + np.random.normal(0, 0.08, len(group))
        plt.scatter(
            group["composite_score"],
            y_jitter,
            label=state_type,
            alpha=0.75
        )

    plt.axvline(0, linestyle="--")
    plt.axhline(mean_count, linestyle="--")

    plt.xlim(
        results["composite_score"].min() - 0.1,
        results["composite_score"].max() + 0.1
    )
    plt.ylim(
        results["improvement_count"].min() - 0.5,
        results["improvement_count"].max() + 0.5
    )

    plt.xlabel("Composite Score")
    plt.ylabel("Improvement Count")
    plt.title("State Classification: Breadth vs Composite Improvement")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=300, bbox_inches="tight")
    plt.close()


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
    print(f"Saved plot to: {PLOT_FILE}")


if __name__ == "__main__":
    main()