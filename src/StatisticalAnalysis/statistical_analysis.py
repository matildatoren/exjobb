"""
Association Analysis: Factors associated with motor development
===============================================================
Improvements over v1:
  - Spearman CIs via Fisher's z-transformation
  - One-sample Wilcoxon signed-rank test (is change != 0?)
  - Feature correlation matrix (multicollinearity check)
  - Mixed-effects model with random intercept per child (correct for repeated measures)
  - Standardized regression coefficients
  - Post-hoc power analysis
  - Stratified Spearman by GMFCS level

Usage:
    cd src
    python ../statistical_analysis_v3.py
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import FTestAnovaPower

SRC = Path(__file__).resolve().parent / "src"
sys.path.append(str(SRC))

from src.connect_db import get_connection
from src.dataloader import load_data
from src.preprocessing.master_preprocessing import build_master_feature_table

IMAGES_DIR = Path(__file__).resolve().parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────

TARGETS = [
    "delta_impairment_score_setvalue",
    "delta_motorical_score",
]

CONTINUOUS_FEATURES = [
    "total_home_training_hours",
    "total_other_training_hours",
    "neurohab_hours",
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
    "has_any_device":             "Uses any device",
    "has_any_medical_treatment":  "Any medical treatment",
    "gmfcs_int":                  "GMFCS level",
}

TARGET_LABELS = {
    "delta_impairment_score_setvalue": "Δ Impairment score",
    "delta_motorical_score":           "Δ Motorical score",
}

MIN_GROUP_N  = 5
OUTLIER_Z    = 3.0
ALPHA        = 0.05


# ─── Data preparation ─────────────────────────────────────────────────────────

def prepare_data(master) -> pd.DataFrame:
    """
    Returns a long-format dataframe keeping all age rows per child.
    introductory_id is preserved for mixed-effects grouping.
    """
    df = master.to_pandas()
    all_cols = ["introductory_id", "age"] + CONTINUOUS_FEATURES + BINARY_FEATURES + CONTROL_VARS + TARGETS
    existing = [c for c in all_cols if c in df.columns]
    df = df[existing].copy()

    hour_cols    = [c for c in CONTINUOUS_FEATURES if c in df.columns]
    has_training = df[hour_cols].fillna(0).sum(axis=1) > 0
    n_dropped    = (~has_training).sum()
    print(f"  Dropping {n_dropped} rows with no training reported")
    df = df[has_training].reset_index(drop=True)

    df[CONTINUOUS_FEATURES + CONTROL_VARS] = df[CONTINUOUS_FEATURES + CONTROL_VARS].fillna(0)
    return df


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _spearman_ci(r: float, n: int, alpha: float = ALPHA):
    """95% CI for Spearman ρ via Fisher z-transformation."""
    if n < 4:
        return np.nan, np.nan
    z      = np.arctanh(r)
    se     = 1 / np.sqrt(n - 3)
    z_crit = norm.ppf(1 - alpha / 2)
    return np.tanh(z - z_crit * se), np.tanh(z + z_crit * se)


def _standardize(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Return a copy with specified columns z-scored."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            std = out[c].std()
            if std > 0:
                out[c] = (out[c] - out[c].mean()) / std
    return out


# ─── 0. Descriptives ──────────────────────────────────────────────────────────

def print_descriptives(df: pd.DataFrame):
    sep = "=" * 70
    print(f"\n{sep}")
    print("  DESCRIPTIVE STATISTICS")
    print(sep)

    print("\n  Targets:")
    print(f"  {'Target':<35} {'n':>5} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
    print(f"  {'-'*70}")
    for target in TARGETS:
        if target not in df.columns:
            continue
        col = df[target].dropna()
        print(f"  {TARGET_LABELS.get(target,target):<35} {len(col):>5}"
              f" {col.mean():>8.3f} {col.std():>8.3f}"
              f" {col.min():>8.3f} {col.max():>8.3f}")

    print("\n  Continuous features:")
    print(f"  {'Feature':<35} {'n':>5} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
    print(f"  {'-'*70}")
    for feat in CONTINUOUS_FEATURES + CONTROL_VARS:
        if feat not in df.columns:
            continue
        col = df[feat].dropna()
        print(f"  {FEATURE_LABELS.get(feat,feat):<35} {len(col):>5}"
              f" {col.mean():>8.3f} {col.std():>8.3f}"
              f" {col.min():>8.3f} {col.max():>8.3f}")

    print("\n  Binary features:")
    print(f"  {'Feature':<35} {'n(yes)':>8} {'n(no)':>8} {'%yes':>8}")
    print(f"  {'-'*60}")
    for feat in BINARY_FEATURES:
        if feat not in df.columns:
            continue
        n_yes = int(df[feat].sum())
        n_no  = int((df[feat] == 0).sum())
        pct   = 100 * n_yes / max(n_yes + n_no, 1)
        print(f"  {FEATURE_LABELS.get(feat,feat):<35} {n_yes:>8} {n_no:>8} {pct:>7.1f}%")

    print(f"\n  Unique children: {df['introductory_id'].nunique()}")
    print(f"  Total rows (child × age): {len(df)}")


# ─── 1. One-sample Wilcoxon (is change ≠ 0?) ─────────────────────────────────

def run_wilcoxon_change(df: pd.DataFrame):
    sep = "=" * 70
    print(f"\n{sep}")
    print("  ONE-SAMPLE WILCOXON SIGNED-RANK TEST  (H₀: median change = 0)")
    print(sep)
    print(f"  {'Target':<35} {'n':>5} {'median':>8} {'W':>10} {'p':>8}")
    print(f"  {'-'*65}")

    for target in TARGETS:
        if target not in df.columns:
            continue
        col = df[target].dropna()
        col = col[col != 0]           # exclude structural zeros
        if len(col) < 5:
            continue
        stat, p = stats.wilcoxon(col, alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {TARGET_LABELS.get(target,target):<35}"
              f" {len(col):>5} {col.median():>8.4f}"
              f" {stat:>10.1f} {p:>7.4f}{sig}")

    print(f"\n  Significance: * p<0.05  ** p<0.01  *** p<0.001")


# ─── 2. Spearman with CIs and FDR ────────────────────────────────────────────

def run_spearman(df: pd.DataFrame) -> pd.DataFrame:
    features = CONTINUOUS_FEATURES + BINARY_FEATURES + CONTROL_VARS
    rows = []

    for target in TARGETS:
        if target not in df.columns:
            continue
        for feat in features:
            if feat not in df.columns:
                continue
            common = df[[feat, target]].dropna()
            if len(common) < 5:
                continue
            r, p = stats.spearmanr(common[feat], common[target])
            ci_lo, ci_hi = _spearman_ci(r, len(common))
            rows.append({
                "target":  TARGET_LABELS.get(target, target),
                "feature": FEATURE_LABELS.get(feat, feat),
                "rho":     round(r, 3),
                "ci_lo":   round(ci_lo, 3),
                "ci_hi":   round(ci_hi, 3),
                "p_value": round(p, 4),
                "n":       len(common),
            })

    result = pd.DataFrame(rows)

    # FDR correction per target
    for target in result["target"].unique():
        mask = result["target"] == target
        _, p_corr, _, _ = multipletests(result.loc[mask, "p_value"], method="fdr_bh")
        result.loc[mask, "p_fdr"] = p_corr.round(4)

    return result


def print_spearman(spearman_df: pd.DataFrame):
    sep = "=" * 70
    print(f"\n{sep}")
    print("  SPEARMAN CORRELATIONS  (with 95% CI and FDR correction)")
    print(sep)
    for target in spearman_df["target"].unique():
        sub = spearman_df[spearman_df["target"] == target].sort_values(
            "rho", key=abs, ascending=False)
        print(f"\n  Target: {target}")
        print(f"  {'Feature':<35} {'ρ':>7} {'95% CI':>16} {'p':>8} {'p_fdr':>8} {'n':>5}")
        print(f"  {'-'*78}")
        for _, row in sub.iterrows():
            sig     = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
            sig_fdr = "†" if row["p_fdr"] < 0.05 else ""
            ci_str  = f"[{row['ci_lo']:+.3f}, {row['ci_hi']:+.3f}]"
            print(f"  {row['feature']:<35} {row['rho']:>7.3f} {ci_str:>16}"
                  f" {row['p_value']:>7.4f}{sig:<3} {row['p_fdr']:>7.4f}{sig_fdr:<2}  (n={row['n']})")
    print(f"\n  sig = uncorrected  |  † = FDR-corrected p<0.05")


# ─── 3. Stratified Spearman by GMFCS ─────────────────────────────────────────

def run_spearman_by_gmfcs(df: pd.DataFrame):
    sep = "=" * 70
    print(f"\n{sep}")
    print("  SPEARMAN CORRELATIONS — STRATIFIED BY GMFCS LEVEL")
    print(sep)

    if "gmfcs_int" not in df.columns:
        print("  gmfcs_int not found — skipping.")
        return

    features = CONTINUOUS_FEATURES
    levels   = sorted(df["gmfcs_int"].dropna().unique())

    for target in TARGETS:
        if target not in df.columns:
            continue
        label = TARGET_LABELS.get(target, target)
        print(f"\n  Target: {label}")
        print(f"  {'Feature':<30} " + " ".join(f"{'GMFCS '+str(int(l)):>14}" for l in levels))
        print(f"  {'-'*75}")

        for feat in features:
            if feat not in df.columns:
                continue
            feat_label = FEATURE_LABELS.get(feat, feat)
            row_str = f"  {feat_label:<30}"
            for lvl in levels:
                sub    = df[df["gmfcs_int"] == lvl][[feat, target]].dropna()
                if len(sub) < 5:
                    row_str += f"  {'n<5':>12}"
                    continue
                r, p   = stats.spearmanr(sub[feat], sub[target])
                sig    = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                row_str += f"  {r:>+6.3f}{sig:<3}(n={len(sub)})"
            print(row_str)


# ─── 4. Feature correlation matrix ───────────────────────────────────────────

def plot_feature_correlation_matrix(df: pd.DataFrame):
    """Heatmap of Spearman correlations between all features — multicollinearity check."""
    features = CONTINUOUS_FEATURES + BINARY_FEATURES + CONTROL_VARS
    existing = [f for f in features if f in df.columns]
    labels   = [FEATURE_LABELS.get(f, f) for f in existing]

    corr_matrix = np.zeros((len(existing), len(existing)))
    features = CONTINUOUS_FEATURES + BINARY_FEATURES + CONTROL_VARS
    existing = [f for f in features if f in df.columns]
    labels   = [FEATURE_LABELS.get(f, f) for f in existing]

    corr_matrix = df[existing].corr(method="spearman").values

    fig, ax = plt.subplots(figsize=(max(6, len(existing) * 0.9), max(5, len(existing) * 0.9)))
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(len(existing)):
        for j in range(len(existing)):
            val = corr_matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if abs(val) < 0.6 else "white")

    plt.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_title("Feature Correlation Matrix (Spearman)\nHigh values indicate multicollinearity", fontsize=11)
    plt.tight_layout()
    path = IMAGES_DIR / "feature_correlation_matrix.png"
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path.name}")
    plt.show()


# ─── 5. Mixed-effects model ───────────────────────────────────────────────────

def run_mixed_effects(df: pd.DataFrame, target: str) -> None:
    """
    Linear mixed-effects model with random intercept per child.
    Correctly handles the repeated-measures structure (multiple ages per child).
    """
    features = CONTINUOUS_FEATURES + BINARY_FEATURES + CONTROL_VARS
    features = [f for f in features if f in df.columns]

    subset = df[["introductory_id"] + features + [target]].dropna()
    if len(subset) < len(features) + 10:
        print(f"  ⚠️  Too few observations for mixed model on {target} (n={len(subset)})")
        return

    formula = f"{target} ~ " + " + ".join(features)
    model   = smf.mixedlm(formula, subset, groups=subset["introductory_id"])

    try:
        result = model.fit(reml=True, method="lbfgs")
    except Exception as e:
        print(f"  ⚠️  Mixed model failed: {e}")
        return

    sep  = "=" * 70
    line = "-" * 78
    label = TARGET_LABELS.get(target, target)
    n_obs      = len(subset)
    n_children = subset["introductory_id"].nunique()

    print(f"\n{sep}")
    print(f"  MIXED-EFFECTS MODEL — {label}")
    print(f"  n_obs={n_obs},  n_children={n_children}  (random intercept per child)")
    print(sep)
    print(f"  {'Feature':<35} {'β':>10} {'SE':>8} {'z':>8} {'p':>8}")
    print(f"  {line}")

    for feat in features:
        if feat not in result.params.index:
            continue
        coef = result.params[feat]
        se   = result.bse[feat]
        z    = result.tvalues[feat]
        p    = result.pvalues[feat]
        sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {FEATURE_LABELS.get(feat,feat):<35} {coef:>+10.5f}"
              f" {se:>8.5f} {z:>8.3f} {p:>7.4f} {sig}")

    print(f"\n  Log-likelihood: {result.llf:.3f}")
    print(f"  Significance: * p<0.05  ** p<0.01  *** p<0.001")


# ─── 6. OLS regression with standardized coefficients ────────────────────────

def run_regression(df: pd.DataFrame, target: str) -> pd.DataFrame | None:
    features = CONTINUOUS_FEATURES + BINARY_FEATURES + CONTROL_VARS
    features = [f for f in features if f in df.columns]

    subset = df[features + [target]].dropna()
    if len(subset) < len(features) + 5:
        print(f"  ⚠️  Too few observations for regression on {target} (n={len(subset)})")
        return None

    # Unstandardized model
    X     = sm.add_constant(subset[features])
    y     = subset[target]
    model = sm.OLS(y, X).fit()

    # Standardized model for beta comparison
    std_subset = _standardize(subset, features + [target])
    X_std      = sm.add_constant(std_subset[features])
    y_std      = std_subset[target]
    model_std  = sm.OLS(y_std, X_std).fit()

    rows = []
    for feat in features:
        coef     = model.params[feat]
        ci_lo, ci_hi = model.conf_int().loc[feat]
        pval     = model.pvalues[feat]
        beta_std = model_std.params.get(feat, np.nan)
        rows.append({
            "feature":   FEATURE_LABELS.get(feat, feat),
            "coef":      round(coef, 5),
            "ci_lower":  round(ci_lo, 5),
            "ci_upper":  round(ci_hi, 5),
            "beta_std":  round(beta_std, 4),
            "p_value":   round(pval, 4),
            "sig":       "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "",
        })

    result_df = pd.DataFrame(rows)
    result_df["n"]      = len(subset)
    result_df["adj_r2"] = round(model.rsquared_adj, 4)
    result_df["target"] = TARGET_LABELS.get(target, target)
    return result_df


def print_regression(reg_df: pd.DataFrame, target: str):
    sep   = "=" * 70
    label = TARGET_LABELS.get(target, target)
    n     = reg_df["n"].iloc[0]
    r2    = reg_df["adj_r2"].iloc[0]
    print(f"\n{sep}")
    print(f"  OLS REGRESSION — {label}")
    print(f"  n={n},  Adjusted R²={r2}  (NOTE: does not account for repeated measures)")
    print(sep)
    print(f"  {'Feature':<35} {'β':>10} {'β_std':>8} {'95% CI':>22} {'p':>8}")
    print(f"  {'-'*85}")
    for _, row in reg_df.iterrows():
        ci_str = f"[{row['ci_lower']:+.4f}, {row['ci_upper']:+.4f}]"
        print(f"  {row['feature']:<35} {row['coef']:>+10.5f}"
              f" {row['beta_std']:>8.4f}  {ci_str:>22}  {row['p_value']:>7.4f} {row['sig']}")
    print(f"\n  β_std = standardized coefficient (z-scored features and target)")
    print(f"  Significance: * p<0.05  ** p<0.01  *** p<0.001")


# ─── 7. Power analysis ────────────────────────────────────────────────────────

def run_power_analysis(df: pd.DataFrame):
    sep = "=" * 70
    print(f"\n{sep}")
    print("  POST-HOC POWER ANALYSIS")
    print(sep)

    n_pred  = len(CONTINUOUS_FEATURES) + len(BINARY_FEATURES) + len(CONTROL_VARS)
    n_obs   = len(df)

    print(f"\n  Sample: n={n_obs}, predictors={n_pred}")
    print(f"\n  Power to detect effect sizes (OLS, α={ALPHA}):")
    print(f"  {'Effect size (f²)':>20} {'Description':>20} {'Power':>8}")
    print(f"  {'-'*52}")

    power_tool = FTestAnovaPower()
    for f2, desc in [(0.02, "small"), (0.15, "medium"), (0.35, "large")]:
        try:
            power = power_tool.solve_power(
                effect_size=np.sqrt(f2),
                nobs=n_obs,
                alpha=ALPHA,
                k_groups=n_pred + 1,
            )
        except Exception:
            power = np.nan
        print(f"  {f2:>20.2f} {desc:>20} {power:>8.3f}")

    print(f"\n  Minimum detectable effect at 80% power:")
    for power_target in [0.80]:
        try:
            min_f2 = power_tool.solve_power(
                power=power_target,
                nobs=n_obs,
                alpha=ALPHA,
                k_groups=n_pred + 1,
            ) ** 2
            print(f"  f² ≥ {min_f2:.3f}  (equivalent to R² ≈ {min_f2/(1+min_f2):.3f})")
        except Exception:
            print("  Could not compute minimum detectable effect.")


# ─── 8. Group comparisons (Mann-Whitney U) ────────────────────────────────────

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
            if len(group1) < MIN_GROUP_N or len(group0) < MIN_GROUP_N:
                continue
            stat, p    = stats.mannwhitneyu(group1, group0, alternative="two-sided")
            n1, n0     = len(group1), len(group0)
            effect_size = 1 - (2 * stat) / (n1 * n0)
            rows.append({
                "target":      TARGET_LABELS.get(target, target),
                "feature":     FEATURE_LABELS.get(feat, feat),
                "mean_yes":    round(group1.mean(), 4),
                "mean_no":     round(group0.mean(), 4),
                "mean_diff":   round(group1.mean() - group0.mean(), 4),
                "n_yes":       n1,
                "n_no":        n0,
                "U_stat":      round(stat, 1),
                "p_value":     round(p, 4),
                "effect_size": round(effect_size, 3),
                "sig":         "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "",
            })
    return pd.DataFrame(rows)


def print_group_comparisons(group_df: pd.DataFrame):
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  GROUP COMPARISONS (Mann-Whitney U, min group n={MIN_GROUP_N})")
    print(sep)
    for target in group_df["target"].unique():
        sub = group_df[group_df["target"] == target]
        print(f"\n  Target: {target}")
        print(f"  {'Feature':<28} {'Mean(yes)':>10} {'Mean(no)':>10}"
              f" {'Diff':>8} {'n(yes)':>7} {'n(no)':>7} {'p':>8} {'r':>7}")
        print(f"  {'-'*90}")
        for _, row in sub.iterrows():
            print(f"  {row['feature']:<28}"
                  f" {row['mean_yes']:>10.4f} {row['mean_no']:>10.4f}"
                  f" {row['mean_diff']:>+8.4f} {row['n_yes']:>7} {row['n_no']:>7}"
                  f" {row['p_value']:>8.4f}{row['sig']:>3} {row['effect_size']:>7.3f}")
    print(f"\n  r = rank-biserial  |r|>0.1 small, >0.3 medium, >0.5 large")


# ─── Visualizations ───────────────────────────────────────────────────────────

def plot_spearman_heatmap(spearman_df: pd.DataFrame):
    pivot = spearman_df.pivot(index="feature", columns="target", values="rho")
    pvals = spearman_df.pivot(index="feature", columns="target", values="p_value")
    pfdr  = spearman_df.pivot(index="feature", columns="target", values="p_fdr")

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 3), max(4, len(pivot) * 0.7)))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            rho = pivot.values[i, j]
            p   = pvals.values[i, j]
            fdr = pfdr.values[i, j]
            if not np.isnan(rho):
                sig_raw = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                sig_fdr = "†" if fdr < 0.05 else ""
                ax.text(j, i, f"{rho:.2f}{sig_raw}\n{sig_fdr}",
                        ha="center", va="center", fontsize=9,
                        color="black" if abs(rho) < 0.6 else "white")

    plt.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_title("Spearman Correlations\n* p<0.05  ** p<0.01  *** p<0.001  † FDR<0.05")
    plt.tight_layout()
    path = IMAGES_DIR / "spearman_heatmap.png"
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path.name}")
    plt.show()


def plot_coefficient_forest(reg_results: dict):
    valid = {k: v for k, v in reg_results.items() if v is not None}
    if not valid:
        return

    fig, axes = plt.subplots(1, len(valid), figsize=(7 * len(valid), 6), sharey=False)
    if len(valid) == 1:
        axes = [axes]

    for ax, (target, df) in zip(axes, valid.items()):
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

    plt.suptitle("OLS Regression Coefficients with 95% CI\n(blue = p<0.05)", fontsize=12)
    plt.tight_layout()
    path = IMAGES_DIR / "regression_forest_plot.png"
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path.name}")
    plt.show()


def plot_scatter_continuous(df: pd.DataFrame):
    existing_targets = [t for t in TARGETS if t in df.columns]
    existing_feats   = [f for f in CONTINUOUS_FEATURES if f in df.columns]
    n_rows = len(existing_targets)
    n_cols = len(existing_feats)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

    for row_i, target in enumerate(existing_targets):
        for col_i, feat in enumerate(existing_feats):
            ax     = axes[row_i][col_i]
            common = df[[feat, target]].dropna()
            ax.scatter(common[feat], common[target], alpha=0.5, color="steelblue", s=30)
            if len(common) >= 5:
                rho, p = stats.spearmanr(common[feat], common[target])
                ci_lo, ci_hi = _spearman_ci(rho, len(common))
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                ax.annotate(
                    f"ρ={rho:.2f}{sig}\n95%CI [{ci_lo:+.2f},{ci_hi:+.2f}]\nn={len(common)}",
                    xy=(0.97, 0.97), xycoords="axes fraction",
                    ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
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


def plot_group_boxplots(df: pd.DataFrame):
    existing_targets = [t for t in TARGETS if t in df.columns]
    existing_feats   = [f for f in BINARY_FEATURES if f in df.columns]
    if not existing_feats:
        return

    fig, axes = plt.subplots(len(existing_targets), len(existing_feats),
                             figsize=(5 * len(existing_feats), 4 * len(existing_targets)),
                             squeeze=False)

    for row_i, target in enumerate(existing_targets):
        for col_i, feat in enumerate(existing_feats):
            ax     = axes[row_i][col_i]
            common = df[[feat, target]].dropna()
            g0     = common[common[feat] == 0][target]
            g1     = common[common[feat] == 1][target]

            bp = ax.boxplot([g0, g1], patch_artist=True, widths=0.5,
                            medianprops=dict(color="black", linewidth=2))
            bp["boxes"][0].set_facecolor("#AED6F1")
            bp["boxes"][1].set_facecolor("#A9DFBF")

            if len(g1) >= MIN_GROUP_N and len(g0) >= MIN_GROUP_N:
                _, p = stats.mannwhitneyu(g1, g0, alternative="two-sided")
                sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                title_str = f"{FEATURE_LABELS.get(feat,feat)}\np={p:.4f} {sig}"
            else:
                title_str = f"{FEATURE_LABELS.get(feat,feat)}\n(group too small)"

            ax.set_xticks([1, 2])
            ax.set_xticklabels([f"No\n(n={len(g0)})", f"Yes\n(n={len(g1)})"])
            ax.set_title(title_str)
            ax.set_ylabel(TARGET_LABELS.get(target, target))
            ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")

    plt.suptitle("Motor Score by Group (Mann-Whitney U)", fontsize=13)
    plt.tight_layout()
    path = IMAGES_DIR / "group_comparisons.png"
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
    print(f"  Working dataset: {len(df)} rows, {df['introductory_id'].nunique()} children")

    # ── 0. Descriptives ───────────────────────────────────────────────────────
    print_descriptives(df)

    # ── 1. Wilcoxon: is change ≠ 0? ──────────────────────────────────────────
    run_wilcoxon_change(df)

    # ── 2. Spearman with CIs and FDR ─────────────────────────────────────────
    spearman_df = run_spearman(df)
    print_spearman(spearman_df)

    # ── 3. Stratified Spearman by GMFCS ──────────────────────────────────────
    run_spearman_by_gmfcs(df)

    # ── 4. OLS regression with standardized betas ────────────────────────────
    reg_results = {}
    for target in TARGETS:
        if target not in df.columns:
            continue
        reg_df = run_regression(df, target)
        reg_results[target] = reg_df
        if reg_df is not None:
            print_regression(reg_df, target)

    # ── 5. Mixed-effects model (correct for repeated measures) ───────────────
    for target in TARGETS:
        if target in df.columns:
            run_mixed_effects(df, target)

    # ── 6. Group comparisons ─────────────────────────────────────────────────
    group_df = run_group_comparisons(df)
    if not group_df.empty:
        print_group_comparisons(group_df)

    # ── 7. Power analysis ────────────────────────────────────────────────────
    run_power_analysis(df)

    # ── 8. Plots ─────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  Generating plots …")
    print(sep)

    plot_spearman_heatmap(spearman_df)
    plot_coefficient_forest(reg_results)
    plot_feature_correlation_matrix(df)
    plot_group_boxplots(df)
    plot_scatter_continuous(df)

    plt.show()
    print("\nDone.")


if __name__ == "__main__":
    main()