import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------
# FIX: JSON compatibility
# -----------------------------
def convert_numpy(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


# -----------------------------
# Variable classification
# -----------------------------
def classify_variable(col):
    col = col.lower()

    if col == "data_year":
        return "time"
    if "rate" in col or "per_100k" in col:
        return "rate"
    if "%" in col:
        return "percentage"
    if "pop" in col:
        return "population"
    if "total" in col or "victim" in col or "firearm" in col:
        return "count"

    return "numeric"


# -----------------------------
# Numeric summary
# -----------------------------
def summarize_numeric(series, col):
    s = series.dropna()

    if len(s) == 0:
        return {"column": col}

    skew = s.skew()

    return {
        "column": col,
        "variable_type": classify_variable(col),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std()),
        "min": float(s.min()),
        "max": float(s.max()),
        "skew": float(skew),
        "high_skew": bool(abs(skew) > 1),
        "extreme_skew": bool(abs(skew) > 2),
    }


# -----------------------------
# Correlations
# -----------------------------
def build_correlations(df, numeric_cols):
    corr = df[numeric_cols].corr()
    results = []

    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            v1, v2 = numeric_cols[i], numeric_cols[j]
            val = corr.loc[v1, v2]

            results.append({
                "var1": v1,
                "var2": v2,
                "correlation": float(val),
                "abs_correlation": float(abs(val)),
                "strength": "strong" if abs(val) > 0.7 else "moderate" if abs(val) > 0.4 else "weak"
            })

    results.sort(key=lambda x: x["abs_correlation"], reverse=True)
    return results


# -----------------------------
# STATE POLICY ANALYSIS
# -----------------------------
def compute_policy_changes(df):
    results = []

    states = df["state_name"].unique()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove("data_year")

    for state in states:
        sub = df[df["state_name"] == state]

        pre = sub[sub["data_year"] == 2021]
        post = sub[sub["data_year"].isin([2023, 2024])]

        if pre.empty or post.empty:
            continue

        pre_row = pre.iloc[0]
        post_mean = post.mean(numeric_only=True)

        for col in numeric_cols:
            pre_val = pre_row[col]
            post_val = post_mean[col]

            if pd.isna(pre_val) or pd.isna(post_val):
                continue

            abs_change = post_val - pre_val

            if pre_val == 0:
                pct = None
                reliable = False
            else:
                pct = abs_change / pre_val
                reliable = pre_val >= 10

            results.append({
                "state": state,
                "variable": col,
                "pre_2022": float(pre_val),
                "post_2022_avg": float(post_val),
                "absolute_change": float(abs_change),
                "percent_change": float(pct) if pct is not None else None,
                "percent_change_reliable": bool(reliable),
                "direction": "increase" if abs_change > 0 else "decrease"
            })

    return results


# -----------------------------
# VARIABLE SUMMARY
# -----------------------------
def summarize_policy_changes(results):
    df = pd.DataFrame(results)
    summary = []

    for var in df["variable"].unique():
        sub = df[df["variable"] == var]
        valid = sub[sub["percent_change"].notna()]

        summary.append({
            "variable": var,
            "mean_percent_change": float(valid["percent_change"].mean()),
            "median_percent_change": float(valid["percent_change"].median()),
            "states_increased": int((valid["absolute_change"] > 0).sum()),
            "states_decreased": int((valid["absolute_change"] < 0).sum())
        })

    return summary


# -----------------------------
# NATIONAL TRENDS
# -----------------------------
def compute_national_trends(df):
    results = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove("data_year")

    pre = df[df["data_year"] == 2021]
    post = df[df["data_year"].isin([2023, 2024])]

    for col in numeric_cols:
        var_type = classify_variable(col)

        if var_type == "count":
            pre_val = pre[col].sum()
            post_val = post.groupby("data_year")[col].sum().mean()
        else:
            pre_val = pre[col].mean()
            post_val = post[col].mean()

        if pre_val == 0:
            pct = None
            reliable = False
        else:
            pct = (post_val - pre_val) / pre_val
            reliable = pre_val >= 10

        results.append({
            "variable": col,
            "national_pre": float(pre_val),
            "national_post": float(post_val),
            "percent_change": float(pct) if pct is not None else None,
            "percent_change_reliable": bool(reliable)
        })

    return results


# -----------------------------
# MAIN
# -----------------------------
def main():
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir.parent.parent / "aggregated_crime_and_census_data.csv"
    default_output = script_dir.parent / "output"

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(default_input))
    parser.add_argument("--output_dir", default=str(default_output))
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # FIX: pandas warning
    categorical_cols = df.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()

    numeric_summary = [summarize_numeric(df[c], c) for c in numeric_cols]
    correlations = build_correlations(df, numeric_cols)

    policy_results = compute_policy_changes(df)
    policy_summary = summarize_policy_changes(policy_results)
    national_trends = compute_national_trends(df)

    eda_output = {
        "numeric_summary": numeric_summary,
        "correlations_top_100": correlations[:100],
        "policy_analysis": {
            "state_level": policy_results,
            "variable_summary": policy_summary,
            "national_trends": national_trends
        }
    }

    with open(output_dir / "eda_summary.json", "w") as f:
        json.dump(eda_output, f, indent=2, default=convert_numpy)

    pd.DataFrame(policy_results).to_csv(output_dir / "policy_state_level.csv", index=False)
    pd.DataFrame(policy_summary).to_csv(output_dir / "policy_summary.csv", index=False)
    pd.DataFrame(national_trends).to_csv(output_dir / "national_trends.csv", index=False)

    print("Done.")


if __name__ == "__main__":
    main()