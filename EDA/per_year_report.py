"""
Per-year EDA report for FBI/NIBRS-derived incidents (female victims only).

Runs ONE year at a time and writes a GitHub-friendly report plus tables/figures.

Outputs (per year):
- <OUT_BASE>/<YEAR>/REPORT.md
- <OUT_BASE>/<YEAR>/tables/*.csv
- <OUT_BASE>/<YEAR>/figures/*.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG: Edit this block to run new years / adapt to schema differences
# =============================================================================
CONFIG: Dict = {
    # Input file for a single year (relative to repo root by default)
    "input_csv": "MASTER_2022.csv",

    # Year label used in output paths and report title
    "year_label": "2022",

    # Output base directory (per-year folders created beneath this)
    # relative to repo root
    "out_base": "EDA/output",

    # Subfolders (relative to per-year folder)
    "tables_dirname": "tables",
    "figures_dirname": "figures",

    # Missing-value tokens in raw CSV that should be treated as NA
    "na_tokens": ["N/A", "NA", "NULL", "null", "None", ""],

    # Core columns (change if schema differs)
    "cols": {
        "state": "state_name",
        "weapon": "weapon_name",
        "relationship": "relationship_name",
        "injury": "injury_name",
        "race": "race_desc",
        "ethnicity": "ethnicity_name",
        "offense_name": "offense_name",
        "offense_category": "offense_category_name",
        "location": "location_name",
        # NOTE: excluded from profiling outputs per your request:
        # "crime_against": "crime_against",
        # "offense_group": "offense_group",
        # "reporting_type": "reporting_type",
        # "activity_type": "activity_type_name",
        # "bias_desc": "bias_desc",
        # keep bias_category available for schema completeness (but not profiled below)
        "bias_category": "bias_category",
        "age_low": "age_range_low_num",
        "age_high": "age_range_high_num",
    },

    # Derived indicators
    "indicators": {
        # Keyword rules are applied with case-insensitive substring matching
        "firearm_keywords": ["firearm", "gun", "handgun", "rifle", "shotgun", "automatic"],
        # IPV (proxy based on relationship_name)
        "ipv_keywords": [
            "spouse", "husband", "wife",
            "boyfriend", "girlfriend",
            "intimate", "partner",
            "ex-spouse", "ex boyfriend", "ex girlfriend",
        ],
        # Injury: treat any non-missing injury_name as "injury present"
        "injury_present_if_nonmissing": True,
    },

    # Distributions to compute (top N categories saved + plotted)
    "distributions": {
        "top_n_table": 25,
        "top_n_plot": 15,
        # Which categorical columns to profile (by CONFIG["cols"] keys)
        # Removed per your request:
        # - bias_desc
        # - crime_against
        # - offense_group
        # - reporting_type
        # - activity_type
        "categorical_keys": [
            "race",
            "ethnicity",
            "relationship",
            "weapon",
            "injury",
            "offense_name",
            "offense_category",
            "location",
            "bias_category",
        ],
    },

    # State comparison settings
    "state_summary": {
        "top_states_by_volume": 15,
    },

    # Plot settings
    "plots": {
        "dpi": 200,
        "figsize": (10, 6),
        "age_hist_bins": 30,
    },
}
# =============================================================================


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_fig(path: Path, dpi: int) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def col_exists(df: pd.DataFrame, col: Optional[str]) -> bool:
    return bool(col) and (col in df.columns)


def to_lower_safe(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.lower()


def flag_contains_any(series: pd.Series, keywords: List[str]) -> pd.Series:
    s = to_lower_safe(series)
    return s.apply(lambda x: int(any(k in x for k in keywords)))


def derive_age_numeric(df: pd.DataFrame, col_low: str, col_high: str) -> pd.Series:
    """
    Derive numeric age using age range midpoint when both bounds exist,
    else use low bound when only low exists. Returns float series.
    """
    if col_low not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")

    low = pd.to_numeric(df[col_low], errors="coerce")
    high = pd.to_numeric(df[col_high], errors="coerce") if col_high in df.columns else pd.Series(np.nan, index=df.index)

    age = pd.Series(np.nan, index=df.index, dtype="float64")
    both = low.notna() & high.notna()
    age.loc[both] = (low.loc[both] + high.loc[both]) / 2.0

    only_low = low.notna() & high.isna()
    age.loc[only_low] = low.loc[only_low]

    return age


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum().sort_values(ascending=False)
    miss_pct = (miss / len(df) * 100.0).round(2)
    out = pd.DataFrame(
        {
            "missing_count": miss,
            "missing_pct": miss_pct,
            "dtype": df.dtypes.astype(str),
        }
    )
    out = out[out["missing_count"] > 0]
    return out


def value_counts_table(df: pd.DataFrame, col: str, top_n: int) -> pd.DataFrame:
    vc = df[col].value_counts(dropna=False).head(top_n)
    pct = (vc / len(df) * 100.0).round(2)

    out = pd.DataFrame({"count": vc, "pct": pct})
    out.index.name = col
    out = out.reset_index()

    # Ensure the category column is a string and NaNs are labeled
    out[col] = out[col].where(out[col].notna(), "MISSING").astype(str)
    return out


def barplot_top(
    df_counts: pd.DataFrame,
    category_col: str,
    count_col: str,
    title: str,
    out_path: Path,
    dpi: int,
    figsize: Tuple[int, int],
) -> None:
    plt.figure(figsize=figsize)

    # Force category labels to be strings (handle NaN explicitly)
    y = df_counts[category_col].copy()
    y = y.where(y.notna(), "MISSING").astype(str)

    # Force counts to numeric
    x = pd.to_numeric(df_counts[count_col], errors="coerce").fillna(0)

    plt.barh(y, x)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("Count")
    plt.ylabel(category_col)
    save_fig(out_path, dpi=dpi)


def histogram(series: pd.Series, title: str, out_path: Path, dpi: int, figsize: Tuple[int, int], bins: int) -> None:
    s = series.dropna()
    plt.figure(figsize=figsize)
    plt.hist(s, bins=bins)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    save_fig(out_path, dpi=dpi)


def resolve_repo_root() -> Path:
    """
    Resolve repo root robustly based on this script location.
    Assumes this script lives in: <repo_root>/EDA/per_year_report.py
    """
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    # If script is inside EDA/, parent is repo root. If you move it, adjust here.
    return script_dir.parent


def resolve_path_from_repo_root(path_like: str) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return resolve_repo_root() / p


def run_from_config(cfg: Dict) -> None:
    year = str(cfg["year_label"])

    cols = cfg["cols"]
    ind_cfg = cfg["indicators"]
    dist_cfg = cfg["distributions"]
    state_cfg = cfg["state_summary"]
    plot_cfg = cfg["plots"]

    # Resolve IO paths from repo root
    input_path = resolve_path_from_repo_root(cfg["input_csv"])
    out_base = resolve_path_from_repo_root(cfg["out_base"])

    out_dir = out_base / year
    tables_dir = out_dir / cfg["tables_dirname"]
    figures_dir = out_dir / cfg["figures_dirname"]
    ensure_dir(tables_dir)
    ensure_dir(figures_dir)

    df = pd.read_csv(
        input_path,
        na_values=cfg["na_tokens"],
        keep_default_na=True,
        low_memory=False,
    )

    # Derived indicators (only if source column exists)
    if col_exists(df, cols.get("weapon")):
        df["firearm_flag"] = flag_contains_any(df[cols["weapon"]], ind_cfg["firearm_keywords"])
    else:
        df["firearm_flag"] = 0

    if col_exists(df, cols.get("relationship")):
        df["ipv_flag"] = flag_contains_any(df[cols["relationship"]], ind_cfg["ipv_keywords"])
    else:
        df["ipv_flag"] = 0

    if ind_cfg.get("injury_present_if_nonmissing", True) and col_exists(df, cols.get("injury")):
        df["injury_flag"] = df[cols["injury"]].notna().astype(int)
    else:
        df["injury_flag"] = 0

    df["age_numeric"] = derive_age_numeric(df, cols["age_low"], cols["age_high"])

    # Overall summary
    overall = pd.DataFrame(
        [
            {
                "year": year,
                "n_incidents": int(len(df)),
                "firearm_rate": float(df["firearm_flag"].mean()),
                "ipv_proxy_rate": float(df["ipv_flag"].mean()),
                "injury_rate": float(df["injury_flag"].mean()),
                "mean_age": float(df["age_numeric"].mean()),
                "median_age": float(df["age_numeric"].median()),
                "pct_age_missing": float(df["age_numeric"].isna().mean() * 100.0),
            }
        ]
    )
    overall.to_csv(tables_dir / "overall_summary.csv", index=False)

    # Missingness
    miss = missing_report(df)
    miss.to_csv(tables_dir / "missing_report.csv", index=True)

    # Categorical distributions (top N)
    top_n_table = int(dist_cfg["top_n_table"])
    top_n_plot = int(dist_cfg["top_n_plot"])
    dpi = int(plot_cfg["dpi"])
    figsize = tuple(plot_cfg["figsize"])
    cat_keys = dist_cfg["categorical_keys"]

    for key in cat_keys:
        col = cols.get(key)
        if not col_exists(df, col):
            continue

        tab = value_counts_table(df, col, top_n=top_n_table)
        tab.to_csv(tables_dir / f"{col}_top{top_n_table}.csv", index=False)

        plot_tab = tab.head(top_n_plot)
        if len(plot_tab) > 0:
            barplot_top(
                plot_tab,
                category_col=col,
                count_col="count",
                title=f"{year}: Top {min(top_n_plot, len(tab))} {col}",
                out_path=figures_dir / f"{col}_top{top_n_plot}.png",
                dpi=dpi,
                figsize=figsize,
            )

    # Age histogram
    histogram(
        df["age_numeric"],
        title=f"{year}: Age (derived from age range midpoint)",
        out_path=figures_dir / "age_hist.png",
        dpi=dpi,
        figsize=figsize,
        bins=int(plot_cfg["age_hist_bins"]),
    )

    # State summary + plots (still includes firearm/IPV/injury rates)
    if col_exists(df, cols.get("state")):
        state_col = cols["state"]
        top_states = int(state_cfg["top_states_by_volume"])

        state = (
            df.groupby(state_col, dropna=False)
            .agg(
                n_incidents=(state_col, "count"),
                firearm_rate=("firearm_flag", "mean"),
                ipv_proxy_rate=("ipv_flag", "mean"),
                injury_rate=("injury_flag", "mean"),
                mean_age=("age_numeric", "mean"),
                pct_age_missing=("age_numeric", lambda x: float(x.isna().mean() * 100.0)),
            )
            .reset_index()
            .sort_values("n_incidents", ascending=False)
        )
        state.to_csv(tables_dir / "state_summary.csv", index=False)

        top = state.head(top_states).copy()

        # Volume
        plt.figure(figsize=figsize)
        plt.barh(top[state_col].where(top[state_col].notna(), "MISSING").astype(str), top["n_incidents"].astype(float))
        plt.gca().invert_yaxis()
        plt.title(f"{year}: Top {top_states} states by incident volume")
        plt.xlabel("Incident count")
        plt.ylabel("State")
        save_fig(figures_dir / "top_states_by_volume.png", dpi=dpi)

        # Firearm rate
        plt.figure(figsize=figsize)
        plt.barh(top[state_col].where(top[state_col].notna(), "MISSING").astype(str), top["firearm_rate"].astype(float))
        plt.gca().invert_yaxis()
        plt.title(f"{year}: Firearm involvement rate (top {top_states} states by volume)")
        plt.xlabel("Rate")
        plt.ylabel("State")
        save_fig(figures_dir / "top_states_firearm_rate.png", dpi=dpi)

        # IPV proxy rate
        plt.figure(figsize=figsize)
        plt.barh(top[state_col].where(top[state_col].notna(), "MISSING").astype(str), top["ipv_proxy_rate"].astype(float))
        plt.gca().invert_yaxis()
        plt.title(f"{year}: IPV proxy rate (top {top_states} states by volume)")
        plt.xlabel("Rate")
        plt.ylabel("State")
        save_fig(figures_dir / "top_states_ipv_proxy_rate.png", dpi=dpi)

    # Markdown report (GitHub-friendly)
    rel_tables = tables_dir.relative_to(out_dir).as_posix()
    rel_figs = figures_dir.relative_to(out_dir).as_posix()

    lines: List[str] = []
    lines.append(f"# EDA Report: {year}")
    lines.append("")
    lines.append("## Overall summary")
    lines.append("")
    lines.append("```")
    lines.append(overall.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- Tables: `{rel_tables}/`")
    lines.append(f"- Figures: `{rel_figs}/`")
    lines.append("")
    lines.append("## Figures preview")
    lines.append("")

    fig_files = sorted(figures_dir.glob("*.png"))
    for fp in fig_files:
        rel = fp.relative_to(out_dir).as_posix()
        lines.append(f"### {Path(rel).name}")
        lines.append(f"![{Path(rel).name}]({rel})")
        lines.append("")

    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] Input:  {input_path}")
    print(f"[OK] Report: {out_dir / 'REPORT.md'}")
    print(f"[OK] Tables: {tables_dir}")
    print(f"[OK] Figures:{figures_dir}")


if __name__ == "__main__":
    run_from_config(CONFIG)