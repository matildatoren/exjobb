"""
Association Analysis v4 — Extended
====================================
New analyses vs v3:
  1. Baseline-controlled delta regression
  2. Time interval stratification (which age transition)
  3. Training × GMFCS interaction term
  4. Responder analysis (logistic regression)
  5. Dose-response threshold (quartile analysis)
  6. Intraclass Correlation Coefficient (ICC)
  7. Training category comparison (Kruskal-Wallis)

Usage:
    cd src
    python ../statistical_analysis_v4.py
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

SRC = Path(__file__).resolve().parent / "src"
sys.path.append(str(SRC))

from src.connect_db import get_connection
from src.dataloader import load_data
from src.preprocessing.master_preprocessing import build_master_feature_table

FIGURES_DIR = Path(__file__).resolve().parents[2] / "outputs" / "statistical_analysis"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────

TARGETS = [
    "delta_impairment_score_setvalue",
    "delta_motorical_score",
]

BASELINE_COLS = {
    "delta_impairment_score_setvalue": "impairment_score_setvalue",
    "delta_motorical_score":           "motorical_score",
}

CONTINUOUS_FEATURES = [
    "total_home_training_hours",
    "total_other_training_hours",
    "neurohab_hours",
]

CATEGORY_FEATURES = [
    "cat_neurodevelopmental_reflex",
    "cat_motor_learning_task",
    "cat_technology_assisted",
    "cat_suit_based",
    "cat_physical_conditioning",
    "cat_complementary",
]

BINARY_FEATURES = [
    "has_any_device",
    "has_any_medical_treatment",
]

CONTROL_VARS = ["gmfcs_int"]

FEATURE_LABELS = {
    "total_home_training_hours":     "Home training (hrs/yr)",
    "total_other_training_hours":    "Other training (hrs/yr)",
    "neurohab_hours":                "Intensive therapy (hrs/yr)",
    "has_any_device":                "Uses any device",
    "has_any_medical_treatment":     "Any medical treatment",
    "gmfcs_int":                     "GMFCS level",
    "cat_neurodevelopmental_reflex": "Neurodevelopmental/reflex",
    "cat_motor_learning_task":       "Motor learning",
    "cat_technology_assisted":       "Technology assisted",
    "cat_suit_based":                "Suit based",
    "cat_physical_conditioning":     "Physical conditioning",
    "cat_complementary":             "Complementary",
}

TARGET_LABELS = {
    "delta_impairment_score_setvalue": "Δ Impairment score",
    "delta_motorical_score":           "Δ Motorical score",
}

ALPHA    = 0.05
MIN_N    = 5

FILTER_INTRODUCTORY_IDS = [
        "c0990a55-916e-47ba-b29a-aee83d9f33c9",
        "65ab3206-7371-4471-845c-6d238050494f",
        "c8f4ec50-18b6-47ed-92a3-919da180a10d",
        "8dba1f55-9e79-4e62-90c3-02e9609d3feb",
        "f1856ef8-2fe0-480d-9635-cfc0be308458",
        "771d12c3-bc1a-4a97-ad27-00d35b24f87e",
        "1019fb0a-480d-4bef-b8f9-493b9dfe253b",
        "6e7aeec2-2846-433d-a4ac-0e753da08530",
        "e30d335e-3a7a-484d-951d-f8e3f17ccfb3",
        "578adb11-a12f-4121-a567-afe67c25640b",
        "0a584ba1-cdf4-4251-9168-5f8ccc0240e3",
        "7e42b31a-c597-4418-9bf6-a8c3286d049f",
        "89e4bf27-9a6f-45e8-a415-ef53f23f7931",
        "16f3f961-07a2-4099-8498-1bad9c2faa19",
        "44cd783c-b33d-4553-89cd-2a73b59e1982",
        "d2703a20-7b4a-4624-b31a-306eebe4caa0",
        "1d0afd8d-6945-488a-964c-724e95db6696",
        "f9231c8d-2ade-4c0e-a878-a9524ccc3d65",
        "cd26a009-6e51-4372-b151-b7d2bb8b7183",
        "df67e7ea-0b50-408b-9342-4c29d0efa839",
        "30302f7a-c470-47bf-8f0e-d104b3065d99",
        "1950325f-99da-47b4-b49d-735253ba0aaa",
]


# ─── Data preparation ─────────────────────────────────────────────────────────

def prepare_data(master) -> pd.DataFrame:
    df = master.to_pandas()

    # Filter to specific introductory IDs if list is non-empty
    if FILTER_INTRODUCTORY_IDS:
        df = df[df["introductory_id"].isin(FILTER_INTRODUCTORY_IDS)]
        print(f"  Filtered to {len(df)} rows for {len(FILTER_INTRODUCTORY_IDS)} introductory IDs")

    # Keep baseline scores too for baseline-controlled analysis
    baseline_cols = list(BASELINE_COLS.values())
    all_cols = (["introductory_id", "age"]
                + CONTINUOUS_FEATURES + CATEGORY_FEATURES
                + BINARY_FEATURES + CONTROL_VARS
                + TARGETS + baseline_cols)
    existing = [c for c in all_cols if c in df.columns]
    df = df[existing].copy()

    hour_cols    = [c for c in CONTINUOUS_FEATURES if c in df.columns]
    has_training = df[hour_cols].fillna(0).sum(axis=1) > 0
    n_dropped    = (~has_training).sum()
    print(f"  Dropping {n_dropped} rows with no training reported")
    df = df[has_training].reset_index(drop=True)

    # Drop outlier
    if "delta_motorical_score" in df.columns:
        n_before = len(df)
        df = df[df["delta_motorical_score"] != 25.9].reset_index(drop=True)
        if len(df) < n_before:
            print(f"  Dropping 1 outlier (delta_motorical_score = 25.9)")

    fill_cols = [c for c in df.columns if c not in ["introductory_id", "gmfcs_int"]]
    df[fill_cols] = df[fill_cols].fillna(0)

    return df


# ─── Helper ───────────────────────────────────────────────────────────────────

def _sig(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def _standardize(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            std = out[c].std()
            if std > 0:
                out[c] = (out[c] - out[c].mean()) / std
    return out


# ════════════════════════════════════════════════════════════════════════════
# 1. BASELINE-CONTROLLED DELTA REGRESSION
# ════════════════════════════════════════════════════════════════════════════

def run_baseline_controlled_regression(df: pd.DataFrame):
    """
    Regress delta score on training variables while controlling for baseline score.
    This removes the floor/ceiling effect where low-baseline children have
    more room to improve.
    """
    sep = "=" * 70
    print(f"\n{sep}")
    print("  BASELINE-CONTROLLED REGRESSION")
    print("  (controls for starting score to remove floor/ceiling effect)")
    print(sep)

    features = CONTINUOUS_FEATURES + BINARY_FEATURES + CONTROL_VARS

    for target in TARGETS:
        baseline_col = BASELINE_COLS.get(target)
        if target not in df.columns or baseline_col not in df.columns:
            continue

        label    = TARGET_LABELS.get(target, target)
        feat_use = [f for f in features if f in df.columns]
        cols     = feat_use + [target, baseline_col]
        subset   = df[cols].dropna()

        if len(subset) < len(feat_use) + 5:
            print(f"  ⚠️  Too few observations for {label}")
            continue

        X     = sm.add_constant(subset[feat_use + [baseline_col]])
        y     = subset[target]
        model = sm.OLS(y, X).fit()

        n  = len(subset)
        r2 = round(model.rsquared_adj, 4)
        print(f"\n  Target: {label}  |  n={n}  |  Adj. R²={r2}")
        print(f"  {'Feature':<35} {'β':>10} {'95% CI':>22} {'p':>8}")
        print(f"  {'-'*78}")

        for feat in feat_use + [baseline_col]:
            coef     = model.params[feat]
            ci_lo, ci_hi = model.conf_int().loc[feat]
            pval     = model.pvalues[feat]
            ci_str   = f"[{ci_lo:+.4f}, {ci_hi:+.4f}]"
            fname    = FEATURE_LABELS.get(feat, feat) if feat != baseline_col else "Baseline score"
            print(f"  {fname:<35} {coef:>+10.5f}  {ci_str:>22}  {pval:>7.4f} {_sig(pval)}")

    print(f"\n  Significance: * p<0.05  ** p<0.01  *** p<0.001")


# ════════════════════════════════════════════════════════════════════════════
# 2. TIME INTERVAL STRATIFICATION
# ════════════════════════════════════════════════════════════════════════════

def run_time_interval_analysis(df: pd.DataFrame):
    """
    Check whether delta scores differ by which age transition they cover
    (year1→2, year2→3, year3→4). Children develop fastest early.
    """
    sep = "=" * 70
    print(f"\n{sep}")
    print("  TIME INTERVAL ANALYSIS")
    print("  (does the age of transition affect how much children improve?)")
    print(sep)

    if "age" not in df.columns:
        print("  age column not found — skipping.")
        return

    for target in TARGETS:
        if target not in df.columns:
            continue
        label    = TARGET_LABELS.get(target, target)
        col      = df[target].dropna()
        ages     = df.loc[col.index, "age"]
        groups   = {age: col[ages == age].dropna() for age in sorted(ages.unique())}
        groups   = {k: v for k, v in groups.items() if len(v) >= MIN_N}

        if len(groups) < 2:
            continue

        print(f"\n  Target: {label}")
        print(f"  {'Age':>6} {'n':>5} {'median':>8} {'mean':>8} {'std':>8}")
        print(f"  {'-'*40}")
        for age, vals in groups.items():
            print(f"  {int(age):>6} {len(vals):>5} {vals.median():>8.4f}"
                  f" {vals.mean():>8.4f} {vals.std():>8.4f}")

        stat, p = stats.kruskal(*groups.values())
        print(f"\n  Kruskal-Wallis: H={stat:.3f}, p={p:.4f} {_sig(p)}")
        print(f"  Interpretation: {'significant difference between age intervals' if p < ALPHA else 'no significant difference between age intervals'}")

    # Plot
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(6 * len(TARGETS), 5), squeeze=False)
    for ax, target in zip(axes[0], TARGETS):
        if target not in df.columns:
            ax.set_visible(False)
            continue
        label  = TARGET_LABELS.get(target, target)
        subset = df[[target, "age"]].dropna()
        data   = [subset[subset["age"] == age][target].values
                  for age in sorted(subset["age"].unique())]
        labels = [f"Year {int(a)}\n(n={len(d)})"
                  for a, d in zip(sorted(subset["age"].unique()), data)]

        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", linewidth=2))
        colors = ["#AED6F1", "#A9DFBF", "#F9E79F", "#F1948A"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

        ax.set_xticklabels(labels)
        ax.set_ylabel(label)
        ax.set_title(f"{label} by age interval")
        ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")

    plt.suptitle("Score Change by Age Interval", fontsize=13)
    plt.tight_layout()
    path = FIGURES_DIR / "time_interval_analysis.png"
    plt.savefig(path, dpi=150)
    print(f"\n  Saved: {path.name}")
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
# 3. TRAINING × GMFCS INTERACTION
# ════════════════════════════════════════════════════════════════════════════

def run_interaction_analysis(df: pd.DataFrame):
    """
    Test whether the effect of training on outcomes differs by GMFCS level.
    Formally tests what the stratified Spearman analysis hinted at.
    """
    sep = "=" * 70
    print(f"\n{sep}")
    print("  TRAINING × GMFCS INTERACTION")
    print("  (does training effect differ by severity level?)")
    print(sep)

    if "gmfcs_int" not in df.columns:
        print("  gmfcs_int not found — skipping.")
        return

    training_vars = CONTINUOUS_FEATURES

    for target in TARGETS:
        if target not in df.columns:
            continue
        label = TARGET_LABELS.get(target, target)
        print(f"\n  Target: {label}")
        print(f"  {'Training variable':<30} {'β_main':>10} {'β_interact':>12} {'p_interact':>12}")
        print(f"  {'-'*66}")

        for feat in training_vars:
            if feat not in df.columns:
                continue
            feat_label = FEATURE_LABELS.get(feat, feat)
            cols   = [feat, "gmfcs_int", target]
            subset = df[cols].dropna()
            if len(subset) < 15:
                continue

            # Center variables for interaction
            subset = subset.copy()
            subset[feat]         = subset[feat] - subset[feat].mean()
            subset["gmfcs_int"]  = subset["gmfcs_int"] - subset["gmfcs_int"].mean()
            subset["interaction"] = subset[feat] * subset["gmfcs_int"]

            X     = sm.add_constant(subset[[feat, "gmfcs_int", "interaction"]])
            model = sm.OLS(subset[target], X).fit()

            b_main     = model.params[feat]
            b_interact = model.params["interaction"]
            p_interact = model.pvalues["interaction"]
            print(f"  {feat_label:<30} {b_main:>+10.5f} {b_interact:>+12.5f}"
                  f" {p_interact:>11.4f} {_sig(p_interact)}")

    print(f"\n  β_interact: positive = stronger effect at higher GMFCS (more severe)")
    print(f"  Significance: * p<0.05  ** p<0.01  *** p<0.001")

    # Visualize: scatter colored by GMFCS for best predictor
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(7 * len(TARGETS), 5), squeeze=False)
    feat_to_plot = "neurohab_hours"

    for ax, target in zip(axes[0], TARGETS):
        if target not in df.columns or feat_to_plot not in df.columns:
            ax.set_visible(False)
            continue

        label  = TARGET_LABELS.get(target, target)
        subset = df[[feat_to_plot, "gmfcs_int", target]].dropna()
        gmfcs_colors = {1: "#3498db", 2: "#2ecc71", 3: "#f39c12", 4: "#e74c3c", 5: "#8e44ad"}

        for lvl, grp in subset.groupby("gmfcs_int"):
            color = gmfcs_colors.get(int(lvl), "gray")
            ax.scatter(grp[feat_to_plot], grp[target], color=color,
                       alpha=0.6, s=40, label=f"GMFCS {int(lvl)}")
            if len(grp) >= 5:
                z = np.polyfit(grp[feat_to_plot], grp[target], 1)
                x_line = np.linspace(grp[feat_to_plot].min(), grp[feat_to_plot].max(), 50)
                ax.plot(x_line, np.polyval(z, x_line), color=color, linewidth=1.5, alpha=0.8)

        ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
        ax.set_xlabel(FEATURE_LABELS.get(feat_to_plot, feat_to_plot))
        ax.set_ylabel(label)
        ax.set_title(f"Intensive therapy vs {label}\nby GMFCS level")
        ax.legend(fontsize=8)

    plt.suptitle("Training × GMFCS Interaction", fontsize=13)
    plt.tight_layout()
    path = FIGURES_DIR / "interaction_analysis.png"
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path.name}")
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
# 4. RESPONDER ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def run_responder_analysis(df: pd.DataFrame):
    """
    Dichotomize delta into improved (>0) vs not improved (<=0).
    Use logistic regression and Fisher's exact test.
    More clinically interpretable and robust to outliers.
    """
    sep = "=" * 70
    print(f"\n{sep}")
    print("  RESPONDER ANALYSIS")
    print("  (improved = delta > 0, not improved = delta <= 0)")
    print(sep)

    features = CONTINUOUS_FEATURES + BINARY_FEATURES + CONTROL_VARS
    fig_rows  = len(TARGETS)
    fig, axes = plt.subplots(fig_rows, len(CONTINUOUS_FEATURES),
                             figsize=(5 * len(CONTINUOUS_FEATURES), 5 * fig_rows),
                             squeeze=False)

    for row_i, target in enumerate(TARGETS):
        if target not in df.columns:
            continue

        label     = TARGET_LABELS.get(target, target)
        df_target = df.copy()
        df_target["responder"] = (df_target[target] > 0).astype(int)
        df_target = df_target.dropna(subset=[target])
        df_target = df_target[df_target[target] != 0]  # exclude structural zeros

        n_resp    = df_target["responder"].sum()
        n_nonresp = (df_target["responder"] == 0).sum()
        print(f"\n  Target: {label}")
        print(f"  Responders (improved): {n_resp}  |  Non-responders: {n_nonresp}")

        # Logistic regression
        feat_use = [f for f in features if f in df_target.columns]
        subset   = df_target[feat_use + ["responder"]].dropna()
        if len(subset) >= len(feat_use) + 5 and subset["responder"].nunique() == 2:
            X     = sm.add_constant(subset[feat_use])
            model = sm.Logit(subset["responder"], X).fit(disp=0)

            print(f"\n  Logistic regression (outcome = improved vs not)")
            print(f"  {'Feature':<35} {'OR':>8} {'95% CI':>20} {'p':>8}")
            print(f"  {'-'*74}")
            for feat in feat_use:
                coef     = model.params[feat]
                ci_lo, ci_hi = model.conf_int().loc[feat]
                pval     = model.pvalues[feat]
                OR       = np.exp(coef)
                OR_lo    = np.exp(ci_lo)
                OR_hi    = np.exp(ci_hi)
                ci_str   = f"[{OR_lo:.3f}, {OR_hi:.3f}]"
                print(f"  {FEATURE_LABELS.get(feat,feat):<35} {OR:>8.3f} {ci_str:>20}  {pval:>7.4f} {_sig(pval)}")
            print(f"\n  OR = odds ratio  |  OR>1 = associated with improvement")

        # Boxplots: training hours by responder status
        for col_i, feat in enumerate(CONTINUOUS_FEATURES):
            if feat not in df_target.columns:
                continue
            ax   = axes[row_i][col_i]
            resp = df_target[df_target["responder"] == 1][feat].dropna()
            non  = df_target[df_target["responder"] == 0][feat].dropna()

            bp = ax.boxplot([non, resp], patch_artist=True, widths=0.5,
                            medianprops=dict(color="black", linewidth=2))
            bp["boxes"][0].set_facecolor("#AED6F1")
            bp["boxes"][1].set_facecolor("#A9DFBF")

            if len(resp) >= 3 and len(non) >= 3:
                _, p = stats.mannwhitneyu(resp, non, alternative="two-sided")
                sig  = _sig(p)
                ax.set_title(f"{FEATURE_LABELS.get(feat,feat)}\np={p:.3f} {sig}", fontsize=9)
            else:
                ax.set_title(FEATURE_LABELS.get(feat, feat), fontsize=9)

            ax.set_xticks([1, 2])
            ax.set_xticklabels([f"Not improved\n(n={len(non)})", f"Improved\n(n={len(resp)})"])
            ax.set_ylabel(FEATURE_LABELS.get(feat, feat), fontsize=8)
            ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")

    plt.suptitle("Training Hours: Responders vs Non-Responders", fontsize=13)
    plt.tight_layout()
    path = FIGURES_DIR / "responder_analysis.png"
    plt.savefig(path, dpi=150)
    print(f"\n  Saved: {path.name}")
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
# 5. DOSE-RESPONSE THRESHOLD (QUARTILE ANALYSIS)
# ════════════════════════════════════════════════════════════════════════════

def run_quartile_analysis(df: pd.DataFrame):
    """
    Split training hours into quartiles and compare outcome across groups.
    Reveals whether there is a threshold below which no effect is seen.
    """
    sep = "=" * 70
    print(f"\n{sep}")
    print("  DOSE-RESPONSE THRESHOLD — QUARTILE ANALYSIS")
    print(sep)

    n_feats  = len(CONTINUOUS_FEATURES)
    n_targets = len(TARGETS)
    fig, axes = plt.subplots(n_targets, n_feats,
                             figsize=(5 * n_feats, 5 * n_targets), squeeze=False)

    for row_i, target in enumerate(TARGETS):
        if target not in df.columns:
            continue
        label = TARGET_LABELS.get(target, target)

        for col_i, feat in enumerate(CONTINUOUS_FEATURES):
            if feat not in df.columns:
                continue
            ax     = axes[row_i][col_i]
            subset = df[[feat, target]].dropna()
            subset = subset[subset[feat] > 0]  # users only

            if len(subset) < 20:
                ax.set_visible(False)
                continue

            subset = subset.copy()
            subset["quartile"] = pd.qcut(subset[feat], q=4,
                                         labels=["Q1\n(lowest)", "Q2", "Q3", "Q4\n(highest)"])

            groups     = [subset[subset["quartile"] == q][target].values
                          for q in subset["quartile"].cat.categories]
            group_labs = list(subset["quartile"].cat.categories)
            ns         = [len(g) for g in groups]

            stat, p = stats.kruskal(*[g for g in groups if len(g) >= 3])

            bp = ax.boxplot(groups, patch_artist=True, widths=0.5,
                            medianprops=dict(color="black", linewidth=2))
            colors = ["#D6EAF8", "#AED6F1", "#5DADE2", "#2E86C1"]
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)

            ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(group_labs, ns)], fontsize=8)
            ax.set_ylabel(label, fontsize=9)
            ax.set_title(f"{FEATURE_LABELS.get(feat,feat)}\nKruskal-Wallis p={p:.3f} {_sig(p)}",
                         fontsize=9)
            ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")

            print(f"\n  {label} × {FEATURE_LABELS.get(feat,feat)} quartiles:")
            for q, g in zip(group_labs, groups):
                if len(g) > 0:
                    print(f"    {q}: n={len(g)}, median={np.median(g):.4f}, mean={np.mean(g):.4f}")
            print(f"  Kruskal-Wallis: H={stat:.3f}, p={p:.4f} {_sig(p)}")

    plt.suptitle("Dose-Response by Training Quartile", fontsize=13)
    plt.tight_layout()
    path = FIGURES_DIR / "quartile_analysis.png"
    plt.savefig(path, dpi=150)
    print(f"\n  Saved: {path.name}")
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
# 6. INTRACLASS CORRELATION COEFFICIENT (ICC)
# ════════════════════════════════════════════════════════════════════════════

def run_icc_analysis(df: pd.DataFrame):
    """
    ICC(1,1): how consistent is a child's score with itself over time?
    High ICC = baseline dominates, little room for training to explain change.
    Low ICC = scores vary, more signal to explain.
    """
    sep = "=" * 70
    print(f"\n{sep}")
    print("  INTRACLASS CORRELATION COEFFICIENT (ICC)")
    print("  (how much of score variance is between-child vs within-child?)")
    print(sep)

    score_cols = [
        "milestone_score_setvalue",
        "impairment_score_setvalue",
        "combined_score_setvalue",
        "motorical_score",
    ]
    existing_scores = [c for c in score_cols if c in df.columns]

    if not existing_scores:
        print("  No score columns found — skipping.")
        return

    print(f"\n  {'Score':<35} {'ICC':>8} {'Between-child var':>20} {'Within-child var':>18} {'Interpretation'}")
    print(f"  {'-'*95}")

    for score in existing_scores:
        subset = df[["introductory_id", score]].dropna()
        if subset["introductory_id"].nunique() < 5:
            continue

        # One-way ANOVA ICC(1,1)
        groups      = [grp[score].values for _, grp in subset.groupby("introductory_id")
                       if len(grp) >= 2]
        if len(groups) < 5:
            continue

        grand_mean  = subset[score].mean()
        k           = np.mean([len(g) for g in groups])
        n_subjects  = len(groups)

        # MS between and within
        ss_between  = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        ss_within   = sum(np.sum((g - np.mean(g))**2) for g in groups)
        df_between  = n_subjects - 1
        df_within   = sum(len(g) - 1 for g in groups)

        ms_between  = ss_between / df_between if df_between > 0 else np.nan
        ms_within   = ss_within / df_within if df_within > 0 else np.nan

        icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within) if ms_between > 0 else np.nan

        var_between = max((ms_between - ms_within) / k, 0)
        var_within  = ms_within

        if icc >= 0.75:
            interp = "Excellent — baseline dominates"
        elif icc >= 0.5:
            interp = "Moderate — some within-child change"
        elif icc >= 0.25:
            interp = "Fair — substantial within-child variation"
        else:
            interp = "Poor — scores vary greatly within child"

        print(f"  {score:<35} {icc:>8.3f} {var_between:>20.4f} {var_within:>18.4f}  {interp}")

    print(f"\n  ICC interpretation: >0.75 excellent, 0.5–0.75 moderate,")
    print(f"  0.25–0.5 fair, <0.25 poor (Cicchetti 1994)")
    print(f"  High ICC means training can explain little additional variance.")


# ════════════════════════════════════════════════════════════════════════════
# 7. TRAINING CATEGORY COMPARISON
# ════════════════════════════════════════════════════════════════════════════

def run_category_comparison(df: pd.DataFrame):
    """
    Compare outcome across training categories using Kruskal-Wallis.
    Tests whether TYPE of training matters beyond total hours.
    """
    sep = "=" * 70
    print(f"\n{sep}")
    print("  TRAINING CATEGORY COMPARISON (Kruskal-Wallis)")
    print("  (does type of training matter beyond total hours?)")
    print(sep)

    existing_cats = [c for c in CATEGORY_FEATURES if c in df.columns]
    if not existing_cats:
        print("  No category columns found — skipping.")
        return

    for target in TARGETS:
        if target not in df.columns:
            continue
        label = TARGET_LABELS.get(target, target)
        print(f"\n  Target: {label}")
        print(f"  {'Category':<35} {'ρ':>7} {'p':>8} {'n_users':>8}")
        print(f"  {'-'*60}")

        results = []
        for cat in existing_cats:
            subset  = df[[cat, target]].dropna()
            users   = subset[subset[cat] > 0]
            if len(users) < MIN_N:
                continue
            r, p = stats.spearmanr(users[cat], users[target])
            results.append((cat, r, p, len(users)))

        # Sort by abs rho
        results.sort(key=lambda x: abs(x[1]), reverse=True)
        for cat, r, p, n in results:
            print(f"  {FEATURE_LABELS.get(cat,cat):<35} {r:>7.3f} {p:>8.4f}{_sig(p):>3}  {n:>6}")

        # Kruskal-Wallis: which category is associated with the best outcomes?
        print(f"\n  Comparing outcome between children who did vs did not use each category:")
        print(f"  {'Category':<35} {'Mean(users)':>12} {'Mean(non)':>12} {'p':>8}")
        print(f"  {'-'*72}")
        for cat in existing_cats:
            subset   = df[[cat, target]].dropna()
            users    = subset[subset[cat] > 0][target]
            non_users = subset[subset[cat] == 0][target]
            if len(users) < MIN_N or len(non_users) < MIN_N:
                continue
            _, p = stats.mannwhitneyu(users, non_users, alternative="two-sided")
            print(f"  {FEATURE_LABELS.get(cat,cat):<35}"
                  f" {users.mean():>12.4f} {non_users.mean():>12.4f}"
                  f" {p:>8.4f}{_sig(p):>3}")

    # Plot: correlation of each category with outcome
    if existing_cats:
        fig, axes = plt.subplots(1, len(TARGETS),
                                 figsize=(7 * len(TARGETS), 5), squeeze=False)
        for ax, target in zip(axes[0], TARGETS):
            if target not in df.columns:
                ax.set_visible(False)
                continue
            label   = TARGET_LABELS.get(target, target)
            rhos, labels_plot, colors = [], [], []
            for cat in existing_cats:
                subset = df[[cat, target]].dropna()
                users  = subset[subset[cat] > 0]
                if len(users) < MIN_N:
                    continue
                r, p = stats.spearmanr(users[cat], users[target])
                rhos.append(r)
                labels_plot.append(FEATURE_LABELS.get(cat, cat))
                colors.append("#2ecc71" if r > 0 else "#e74c3c")

            y_pos = range(len(rhos))
            ax.barh(list(y_pos), rhos, color=colors, alpha=0.7)
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(labels_plot, fontsize=9)
            ax.axvline(0, color="gray", linewidth=0.8)
            ax.set_xlabel("Spearman ρ (among users only)")
            ax.set_title(f"Training Category vs {label}")

        plt.suptitle("Training Category Associations (among users only)", fontsize=12)
        plt.tight_layout()
        path = FIGURES_DIR / "category_comparison.png"
        plt.savefig(path, dpi=150)
        print(f"\n  Saved: {path.name}")
        plt.show()


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

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
    print(f"  Working dataset: {len(df)} rows, {df['introductory_id'].nunique()} children\n")

    # ── 1. Baseline-controlled regression ────────────────────────────────────
    run_baseline_controlled_regression(df)

    # ── 2. Time interval stratification ──────────────────────────────────────
    run_time_interval_analysis(df)

    # ── 3. Training × GMFCS interaction ──────────────────────────────────────
    run_interaction_analysis(df)

    # ── 4. Responder analysis ─────────────────────────────────────────────────
    run_responder_analysis(df)

    # ── 5. Quartile dose-response ─────────────────────────────────────────────
    run_quartile_analysis(df)

    # ── 6. ICC ───────────────────────────────────────────────────────────────
    run_icc_analysis(df)

    # ── 7. Training category comparison ──────────────────────────────────────
    run_category_comparison(df)

    plt.show()
    print("\nDone.")


if __name__ == "__main__":
    main()