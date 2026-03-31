"""
Association Analysis: Factors associated with motor development
===============================================================
Analyzes which training and treatment factors are associated with
motor milestone scores in children with cerebral palsy.

Outputs:
  - Multiple linear regression with coefficients, CIs and p-values
  - Spearman correlations
  - Group comparisons (Mann-Whitney U)
  - Visualizations

Usage:
    cd src
    python ../association_analysis.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

SRC = Path(__file__).resolve().parent / "src"
sys.path.append(str(SRC))

from src.connect_db import get_connection
from src.dataloader import load_data
from src.preprocessing.master_preprocessing import build_master_feature_table

IMAGES_DIR = Path(__file__).resolve().parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────

TARGETS = [
    "milestone_score_setvalue",
    "delta_milestone_score_setvalue",
]

CONTINUOUS_FEATURES = [
    "total_home_training_hours",
    "total_other_training_hours",
    "neurohab_hours",
    "active_total_hours",
]

BINARY_FEATURES = [
    "has_any_device",
    "has_any_medical_treatment",
]

CONTROL_VARS = [
    "gmfcs_int",
]

FEATURE_LABELS = {
    "total_home_training_hours":  "Home training (hrs/yr)",
    "total_other_training_hours": "Other training (hrs/yr)",
    "neurohab_hours":             "Intensive therapy (hrs/yr)",
    "active_total_hours":         "Total active hours (hrs/yr)",
    "has_any_device":             "Uses any device",
    "has_any_medical_treatment":  "Any medical treatment",
    "gmfcs_int":                  "GMFCS level",
}

TARGET_LABELS = {
    "milestone_score_setvalue":        "Motor milestone score",
    "delta_milestone_score_setvalue":  "Δ Motor milestone score",
}

# ─── Data preparation ─────────────────────────────────────────────────────────

def prepare_data(master) -> pd.DataFrame:
    df = master.to_pandas()
    all_cols = CONTINUOUS_FEATURES + BINARY_FEATURES + CONTROL_VARS + TARGETS
    existing = [c for c in all_cols if c in df.columns]
    df = df[existing].copy()
    df[CONTINUOUS_FEATURES + CONTROL_VARS] = df[CONTINUOUS_FEATURES + CONTROL_VARS].fillna(0)
    return df


# ─── 1. Spearman correlations ─────────────────────────────────────────────────

def run_spearman(df: pd.DataFrame) -> pd.DataFrame:
    features = CONTINUOUS_FEATURES + BINARY_FEATURES + CONTROL_VARS
    rows = []

    for target in TARGETS:
        if target not in df.columns:
            continue
        target_data = df[target].dropna()

        for feat in features:
            if feat not in df.columns:
                continue
            common = df[[feat, target]].dropna()
            if len(common) < 5:
                continue
            r, p = stats.spearmanr(common[feat], common[target])
            rows.append({
                "target":  TARGET_LABELS.get(target, target),
                "feature": FEATURE_LABELS.get(feat, feat),
                "rho":     round(r, 3),
                "p_value": round(p, 4),
                "n":       len(common),
            })

    return pd.DataFrame(rows)


# ─── 2. Multiple linear regression ───────────────────────────────────────────

def run_regression(df: pd.DataFrame, target: str) -> pd.DataFrame | None:
    features = CONTINUOUS_FEATURES + BINARY_FEATURES + CONTROL_VARS
    features = [f for f in features if f in df.columns]

    subset = df[features + [target]].dropna()
    if len(subset) < len(features) + 5:
        print(f"  ⚠️  Too few observations for regression on {target} (n={len(subset)})")
        return None

    X = sm.add_constant(subset[features])
    y = subset[target]

    model = sm.OLS(y, X).fit()

    rows = []
    for feat in features:
        coef  = model.params[feat]
        ci_lo, ci_hi = model.conf_int().loc[feat]
        pval  = model.pvalues[feat]
        rows.append({
            "feature":  FEATURE_LABELS.get(feat, feat),
            "coef":     round(coef, 5),
            "ci_lower": round(ci_lo, 5),
            "ci_upper": round(ci_hi, 5),
            "p_value":  round(pval, 4),
            "sig":      "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "",
        })

    result_df = pd.DataFrame(rows)
    result_df["n"]        = len(subset)
    result_df["adj_r2"]   = round(model.rsquared_adj, 4)
    result_df["target"]   = TARGET_LABELS.get(target, target)
    return result_df


# ─── 3. Group comparisons (Mann-Whitney U) ────────────────────────────────────

def run_group_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for target in TARGETS:
        if target not in df.columns:
            continue

        for feat in BINARY_FEATURES:
            if feat not in df.columns:
                continue

            common = df[[feat, target]].dropna()
            group1 = common[common[feat] == 1][target]
            group0 = common[common[feat] == 0][target]

            if len(group1) < 3 or len(group0) < 3:
                continue

            stat, p = stats.mannwhitneyu(group1, group0, alternative="two-sided")

            # Effect size: rank-biserial correlation
            n1, n0 = len(group1), len(group0)
            effect_size = 1 - (2 * stat) / (n1 * n0)

            rows.append({
                "target":       TARGET_LABELS.get(target, target),
                "feature":      FEATURE_LABELS.get(feat, feat),
                "mean_yes":     round(group1.mean(), 4),
                "mean_no":      round(group0.mean(), 4),
                "mean_diff":    round(group1.mean() - group0.mean(), 4),
                "n_yes":        n1,
                "n_no":         n0,
                "U_stat":       round(stat, 1),
                "p_value":      round(p, 4),
                "effect_size":  round(effect_size, 3),
                "sig":          "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "",
            })

    return pd.DataFrame(rows)


# ─── Printing ─────────────────────────────────────────────────────────────────

def print_spearman(spearman_df: pd.DataFrame):
    sep = "=" * 70
    print(f"\n{sep}")
    print("  SPEARMAN CORRELATIONS")
    print(sep)
    for target in spearman_df["target"].unique():
        sub = spearman_df[spearman_df["target"] == target].sort_values("rho", key=abs, ascending=False)
        print(f"\n  Target: {target}")
        print(f"  {'Feature':<35} {'ρ':>7} {'p':>8} {'n':>5}")
        print(f"  {'-'*55}")
        for _, row in sub.iterrows():
            sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
            print(f"  {row['feature']:<35} {row['rho']:>7.3f} {row['p_value']:>8.4f}{sig:>4}  (n={row['n']})")


def print_regression(reg_df: pd.DataFrame, target: str):
    sep = "=" * 70
    label = TARGET_LABELS.get(target, target)
    n   = reg_df["n"].iloc[0]
    r2  = reg_df["adj_r2"].iloc[0]
    print(f"\n{sep}")
    print(f"  MULTIPLE LINEAR REGRESSION — {label}")
    print(f"  n={n},  Adjusted R²={r2}")
    print(sep)
    print(f"  {'Feature':<35} {'β':>10} {'95% CI':>22} {'p':>8}")
    print(f"  {'-'*78}")
    for _, row in reg_df.iterrows():
        ci_str = f"[{row['ci_lower']:+.4f}, {row['ci_upper']:+.4f}]"
        print(f"  {row['feature']:<35} {row['coef']:>+10.5f}  {ci_str:>22}  {row['p_value']:>7.4f} {row['sig']}")
    print(f"\n  Significance: * p<0.05  ** p<0.01  *** p<0.001")


def print_group_comparisons(group_df: pd.DataFrame):
    sep = "=" * 70
    print(f"\n{sep}")
    print("  GROUP COMPARISONS (Mann-Whitney U)")
    print(sep)
    for target in group_df["target"].unique():
        sub = group_df[group_df["target"] == target]
        print(f"\n  Target: {target}")
        print(f"  {'Feature':<28} {'Mean(yes)':>10} {'Mean(no)':>10} {'Diff':>8} {'n(yes)':>7} {'n(no)':>7} {'p':>8} {'r':>7}")
        print(f"  {'-'*90}")
        for _, row in sub.iterrows():
            print(
                f"  {row['feature']:<28}"
                f" {row['mean_yes']:>10.4f}"
                f" {row['mean_no']:>10.4f}"
                f" {row['mean_diff']:>+8.4f}"
                f" {row['n_yes']:>7}"
                f" {row['n_no']:>7}"
                f" {row['p_value']:>8.4f}{row['sig']:>3}"
                f" {row['effect_size']:>7.3f}"
            )
    print(f"\n  r = rank-biserial effect size  |r|>0.1 small, >0.3 medium, >0.5 large")


# ─── Visualizations ───────────────────────────────────────────────────────────

def plot_coefficient_forest(reg_results: dict):
    """Forest plot of regression coefficients with 95% CI for each target."""
    n_targets = len(reg_results)
    if n_targets == 0:
        return

    fig, axes = plt.subplots(1, n_targets, figsize=(7 * n_targets, 6), sharey=True)
    if n_targets == 1:
        axes = [axes]

    for ax, (target, df) in zip(axes, reg_results.items()):
        if df is None:
            continue
        label = TARGET_LABELS.get(target, target)
        y_pos = range(len(df))

        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        for i, (_, row) in enumerate(df.iterrows()):
            color = "steelblue" if row["p_value"] < 0.05 else "lightgray"
            ax.plot([row["ci_lower"], row["ci_upper"]], [i, i], color=color, linewidth=2)
            ax.scatter(row["coef"], i, color=color, s=60, zorder=5)

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(df["feature"].tolist())
        ax.set_xlabel("Regression coefficient (β)")
        ax.set_title(f"{label}\n(adj. R²={df['adj_r2'].iloc[0]:.3f}, n={df['n'].iloc[0]})")

    plt.suptitle("Regression Coefficients with 95% CI\n(blue = p<0.05)", fontsize=12)
    plt.tight_layout()
    path = IMAGES_DIR / "regression_forest_plot.png"
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path.name}")
    plt.show()


def plot_spearman_heatmap(spearman_df: pd.DataFrame):
    """Heatmap of Spearman ρ values across features and targets."""
    pivot = spearman_df.pivot(index="feature", columns="target", values="rho")
    pvals = spearman_df.pivot(index="feature", columns="target", values="p_value")

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 3), max(4, len(pivot) * 0.6)))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Annotate cells with ρ and significance stars
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            rho = pivot.values[i, j]
            p   = pvals.values[i, j]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            if not np.isnan(rho):
                ax.text(j, i, f"{rho:.2f}{sig}", ha="center", va="center",
                        fontsize=9, color="black" if abs(rho) < 0.6 else "white")

    plt.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_title("Spearman Correlations\n(* p<0.05  ** p<0.01  *** p<0.001)")
    plt.tight_layout()
    path = IMAGES_DIR / "spearman_heatmap.png"
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path.name}")
    plt.show()


def plot_group_boxplots(df: pd.DataFrame):
    """Box plots comparing score distributions by binary feature groups."""
    n_feats   = len(BINARY_FEATURES)
    n_targets = len(TARGETS)
    existing_targets = [t for t in TARGETS if t in df.columns]

    fig, axes = plt.subplots(
        len(existing_targets), n_feats,
        figsize=(5 * n_feats, 4 * len(existing_targets)),
        squeeze=False
    )

    for row_i, target in enumerate(existing_targets):
        for col_i, feat in enumerate(BINARY_FEATURES):
            ax = axes[row_i][col_i]
            if feat not in df.columns:
                ax.set_visible(False)
                continue

            common = df[[feat, target]].dropna()
            g0 = common[common[feat] == 0][target]
            g1 = common[common[feat] == 1][target]

            bp = ax.boxplot(
                [g0, g1],
                patch_artist=True,
                widths=0.5,
                medianprops=dict(color="black", linewidth=2)
            )
            bp["boxes"][0].set_facecolor("#AED6F1")
            bp["boxes"][1].set_facecolor("#A9DFBF")

            _, p = stats.mannwhitneyu(g1, g0, alternative="two-sided") if len(g1) >= 3 and len(g0) >= 3 else (None, 1.0)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

            ax.set_xticks([1, 2])
            ax.set_xticklabels([f"No\n(n={len(g0)})", f"Yes\n(n={len(g1)})"])
            ax.set_title(f"{FEATURE_LABELS.get(feat, feat)}\np={p:.4f} {sig}")
            ax.set_ylabel(TARGET_LABELS.get(target, target))
            ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")

    plt.suptitle("Motor Score by Group (Mann-Whitney U)", fontsize=13)
    plt.tight_layout()
    path = IMAGES_DIR / "group_comparisons.png"
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path.name}")
    plt.show()


def plot_scatter_continuous(df: pd.DataFrame):
    """Scatter plots of continuous features vs targets with Spearman ρ annotation."""
    existing_targets = [t for t in TARGETS if t in df.columns]
    existing_feats   = [f for f in CONTINUOUS_FEATURES if f in df.columns]

    n_rows = len(existing_targets)
    n_cols = len(existing_feats)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

    for row_i, target in enumerate(existing_targets):
        for col_i, feat in enumerate(existing_feats):
            ax = axes[row_i][col_i]
            common = df[[feat, target]].dropna()

            ax.scatter(common[feat], common[target], alpha=0.5, color="steelblue", s=30)

            if len(common) >= 5:
                rho, p = stats.spearmanr(common[feat], common[target])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                ax.annotate(
                    f"ρ={rho:.2f}{sig}\np={p:.3f}\nn={len(common)}",
                    xy=(0.97, 0.97), xycoords="axes fraction",
                    ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
                )

            ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
            ax.set_xlabel(FEATURE_LABELS.get(feat, feat))
            ax.set_ylabel(TARGET_LABELS.get(target, target))

    plt.suptitle("Continuous Features vs Motor Scores\n(* p<0.05  ** p<0.01  *** p<0.001)", fontsize=12)
    plt.tight_layout()
    path = IMAGES_DIR / "scatter_continuous.png"
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path.name}")
    plt.show()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    sep = "=" * 70

    print(sep)
    print("  Loading data …")
    conn = get_connection()
    data = load_data(conn)

    print("  Building master feature table …")
    master = build_master_feature_table(data)
    print(f"  Master table: {master.shape[0]} rows × {master.shape[1]} columns")

    df = prepare_data(master)
    print(f"  Working dataset: {len(df)} rows")

    # ── 1. Spearman ──────────────────────────────────────────────────────────
    spearman_df = run_spearman(df)
    print_spearman(spearman_df)

    # ── 2. Multiple linear regression ────────────────────────────────────────
    reg_results = {}
    for target in TARGETS:
        if target not in df.columns:
            continue
        reg_df = run_regression(df, target)
        reg_results[target] = reg_df
        if reg_df is not None:
            print_regression(reg_df, target)

    # ── 3. Group comparisons ─────────────────────────────────────────────────
    group_df = run_group_comparisons(df)
    print_group_comparisons(group_df)

    # ── 4. Plots ─────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  Generating plots …")
    print(sep)

    plot_spearman_heatmap(spearman_df)
    plot_coefficient_forest(reg_results)
    plot_group_boxplots(df)
    plot_scatter_continuous(df)

    plt.show()
    print("\nDone.")


if __name__ == "__main__":
    main()