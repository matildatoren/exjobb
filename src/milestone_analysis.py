"""
milestone_analysis.py
─────────────────────
Analysis of whether children achieve motor milestones (e.g. crawling, walking)
earlier or later than predicted, and what therapy/training features drive that.

Usage
-----
    python milestone_analysis.py

Outputs (saved to ./output/)
------------------------------
    kaplan_meier.png            — KM curves by GMFCS and training hours
    residual_correlations.png   — correlation of training features with residual
    early_vs_late.png           — residual distribution + boxplot by group
    residual_regression.csv     — Ridge coefficients on residuals
    logistic_coefficients.csv   — logistic regression coefficients (early vs late)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter, CoxPHFitter

# ── resolve imports from repo root ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]   # adjust if needed
sys.path.append(str(ROOT))

from src.preprocessing.master_preprocessing import build_master_feature_table
from src.connect_db import get_connection
from src.dataloader import load_data

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "milestone_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# CONFIG — adjust these to your use case
# ════════════════════════════════════════════════════════════════════════════

# Score threshold above which a child is considered to have achieved the milestone
MILESTONE_THRESHOLD = 0.7

# GMFCS-based normative expected achievement age (age bucket units, not years).
# Replace with values from clinical literature relevant to your milestone.
GMFCS_EXPECTED_AGE: dict[int, float | None] = {
    1: 1.0,
    2: 1.5,
    3: 2.5,
    4: 3.5,
    5: None,  # not expected to achieve independently
}

# ID filter — set to a list of introductory_id strings to restrict the analysis,
# or leave as None to include all children.
#
# Examples:
#   INCLUDE_IDS = None                            # all children
#   INCLUDE_IDS = ["abc123", "def456"]            # only these two
#
INCLUDE_IDS: list[str] | None = None

INCLUDE_IDS = [
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
        "42475b28-2dfd-4114-ac53-d8619881dd2f",
        "7e68f3b3-509b-4352-8eb1-400c9407ac9b",
        "4be3b41c-a0b4-4e7b-ae49-896b37ea2052",
        "52dac13b-a335-449d-a7db-a58e40b5e213",
]
# Features used to analyze residuals
TRAINING_FEATURES = [
    "total_home_training_hours",
    "neurohab_hours",
    # "has_any_medical_device"
    "total_other_training_hours",
    #"active_total_hours",
    ]


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — Build survival dataset (one row per child)
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# ID FILTER
# ════════════════════════════════════════════════════════════════════════════

def filter_ids(master: pl.DataFrame) -> pl.DataFrame:
    """
    Apply INCLUDE_IDS filter to the master table.
    Prints a summary of how many children remain after filtering.
    """
    total = master["introductory_id"].n_unique()

    if INCLUDE_IDS is not None:
        master = master.filter(pl.col("introductory_id").is_in(INCLUDE_IDS))
        missing = [i for i in INCLUDE_IDS if i not in master["introductory_id"].to_list()]
        if missing:
            print(f"  Warning: these IDs were not found in the data: {missing}")

    remaining = master["introductory_id"].n_unique()
    print(f"  ID filter: {total} → {remaining} children"
          + (f"  (included: {INCLUDE_IDS})" if INCLUDE_IDS else " (all children included)"))
    return master


def build_survival_df(
    master: pl.DataFrame,
    threshold: float = MILESTONE_THRESHOLD,
) -> pd.DataFrame:
    """
    For each child, find the first age bucket at which milestone_score >= threshold.
    Children who never cross it are censored (event=0) at their last observed age.
    Baseline covariates are taken from age bucket 1 to avoid data leakage.
    """
    # First age at which threshold is crossed
    achieved = (
        master
        .filter(pl.col("milestone_score") >= threshold)
        .group_by("introductory_id")
        .agg(pl.col("age").min().alias("age_achieved"))
        .with_columns(pl.lit(1).cast(pl.Int32).alias("event"))
    )

    # Last observed age (for censored children)
    last_obs = (
        master
        .group_by("introductory_id")
        .agg(pl.col("age").max().alias("last_age"))
    )

    # Combine: achieved children get their achievement age, others get last_age
    survival = (
        last_obs
        .join(achieved, on="introductory_id", how="left")
        .with_columns([
            pl.when(pl.col("event") == 1)
              .then(pl.col("age_achieved"))
              .otherwise(pl.col("last_age"))
              .alias("duration"),
            pl.col("event").fill_null(0).cast(pl.Int32),
        ])
        .select(["introductory_id", "duration", "event"])
    )

    # Baseline covariates from age bucket 1
    baseline_cols = ["introductory_id", "gmfcs_int"] + TRAINING_FEATURES
    baseline = (
        master
        .filter(pl.col("age") == 1)
        .select([c for c in baseline_cols if c in master.columns])
    )

    df = survival.join(baseline, on="introductory_id", how="left").to_pandas()
    return df


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — Predicted achievement age
# ════════════════════════════════════════════════════════════════════════════

def add_gmfcs_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Option A: normative GMFCS-based expected achievement age.
    Children with GMFCS 5 (or unknown expected age) are excluded.
    """
    df = df.copy()
    df["predicted_age"] = df["gmfcs_int"].map(GMFCS_EXPECTED_AGE)
    return df.dropna(subset=["predicted_age"])


def add_model_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Option B: data-driven prediction using only GMFCS as baseline feature.
    Uses cross-validated predictions to avoid leakage.
    """
    df = df.copy()
    mask = df["gmfcs_int"].notna() & df["duration"].notna()
    X = df.loc[mask, ["gmfcs_int"]]
    y = df.loc[mask, "duration"]

    preds = cross_val_predict(
        GradientBoostingRegressor(n_estimators=100, random_state=42),
        X, y, cv=min(5, len(y)),
    )
    df.loc[mask, "predicted_age"] = preds
    return df.dropna(subset=["predicted_age"])


def add_residuals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ✅ scale training (interpretable units)
    if "active_total_hours" in df.columns:
        df["training_100h"] = df["active_total_hours"] / 100

    df["residual"] = df["duration"] - df["predicted_age"]
    df["achieved_early"] = (df["residual"] < 0).astype(int)

    return df



# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — Kaplan-Meier curves
# ════════════════════════════════════════════════════════════════════════════

def plot_kaplan_meier(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # — By GMFCS level
    ax = axes[0]
    for level in sorted(df["gmfcs_int"].dropna().unique()):
        mask = df["gmfcs_int"] == level
        if mask.sum() < 3:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(df.loc[mask, "duration"], df.loc[mask, "event"],
                label=f"GMFCS {int(level)} (n={mask.sum()})")
        kmf.plot_survival_function(ax=ax)
    ax.set_title("Time to milestone achievement\nby GMFCS level")
    ax.set_xlabel("Age bucket")
    ax.set_ylabel("Proportion not yet achieved")
    ax.legend(fontsize=8)

    # — By training hours (median split)
    ax = axes[1]
    col = "total_home_training_hours"
    if col in df.columns:
        median_h = df[col].median()
        groups = [
            ("High training hours", df[col] >= median_h),
            ("Low training hours",  df[col] <  median_h),
        ]
        for label, mask in groups:
            kmf = KaplanMeierFitter()
            kmf.fit(df.loc[mask, "duration"], df.loc[mask, "event"], label=label)
            kmf.plot_survival_function(ax=ax)
        ax.set_title("Time to milestone achievement\nby home training hours (median split)")
        ax.set_xlabel("Age bucket")
        ax.set_ylabel("Proportion not yet achieved")

    plt.tight_layout()
    out = OUTPUT_DIR / "kaplan_meier.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")

def plot_additional_analysis(df: pd.DataFrame) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # ── 1. Observed vs predicted ─────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(df["predicted_age"], df["duration"], alpha=0.7)

    min_val = min(df["predicted_age"].min(), df["duration"].min())
    max_val = max(df["predicted_age"].max(), df["duration"].max())

    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")

    ax.set_xlabel("Predicted age")
    ax.set_ylabel("Observed age")
    ax.set_title("Observed vs Predicted milestone age")
    ax.legend()

    plt.tight_layout()
    out = OUTPUT_DIR / "observed_vs_predicted.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


    # ── 2. Residual vs training ─────────────────────────────
    if "active_total_hours" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 5))

        sns.regplot(
            x="active_total_hours",
            y="residual",
            data=df,
            ax=ax,
            scatter_kws={"alpha": 0.7},
            line_kws={"color": "red"},
        )

        ax.axhline(0, linestyle="--", color="black")
        ax.set_title("Training vs residual\n(negative = earlier than expected)")
        ax.set_xlabel("Active training hours")
        ax.set_ylabel("Residual (actual − predicted)")

        plt.tight_layout()
        out = OUTPUT_DIR / "residual_vs_training.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved: {out}")


    # ── 3. Training distribution ─────────────────────────────
    if "active_total_hours" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 5))

        sns.histplot(df["active_total_hours"], bins=15, kde=True, ax=ax)

        ax.set_title("Distribution of training hours")
        ax.set_xlabel("Active training hours")

        plt.tight_layout()
        out = OUTPUT_DIR / "training_distribution.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved: {out}")


    # ── 4. Duration by GMFCS ─────────────────────────────
    if "gmfcs_int" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 5))

        sns.boxplot(x="gmfcs_int", y="duration", data=df, ax=ax)

        ax.set_title("Milestone age by GMFCS level")
        ax.set_xlabel("GMFCS level")
        ax.set_ylabel("Age at milestone")

        plt.tight_layout()
        out = OUTPUT_DIR / "gmfcs_vs_duration.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved: {out}")


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — Cox proportional hazards model
# ════════════════════════════════════════════════════════════════════════════

def run_cox(df: pd.DataFrame) -> None:
    feature_cols = ["gmfcs_int"] + TRAINING_FEATURES
    available = [c for c in feature_cols if c in df.columns]

    df_cox = df[["duration", "event"] + available].dropna()
    if len(df_cox) < 10:
        print("Not enough complete cases for Cox model — skipping.")
        return

    print("\n── Cox proportional hazards ──────────────────────────────────")
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df_cox, duration_col="duration", event_col="event")
    cph.print_summary()

    print("\n── Checking proportional hazards assumption ──────────────────")
    try:
        cph.check_assumptions(df_cox, p_value_threshold=0.05)
    except Exception as e:
        print(f"  Could not check assumptions: {e}")

    # Stratified by GMFCS (avoids PH assumption on gmfcs_int)
    non_gmfcs = [c for c in available if c != "gmfcs_int"]
    if non_gmfcs:
        print("\n── Cox stratified by GMFCS ───────────────────────────────────")
        cph_s = CoxPHFitter(penalizer=0.1)
        cph_s.fit(df_cox, duration_col="duration", event_col="event",
                  strata=["gmfcs_int"])
        cph_s.print_summary()


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — Residual analysis
# ════════════════════════════════════════════════════════════════════════════

def plot_residuals(df: pd.DataFrame) -> None:
    available = [c for c in TRAINING_FEATURES if c in df.columns]

    # — Correlation bar chart
    corr = (
        df[available + ["residual"]]
        .corr()["residual"]
        .drop("residual")
        .sort_values()
    )
    colors = ["#2ecc71" if v < 0 else "#e74c3c" for v in corr]

    fig, ax = plt.subplots(figsize=(8, max(4, len(corr) * 0.5)))
    corr.plot(kind="barh", ax=ax, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(
        "Correlation with residual (actual − predicted age)\n"
        "Green = earlier than predicted   Red = later than predicted"
    )
    ax.set_xlabel("Pearson correlation")
    plt.tight_layout()
    out = OUTPUT_DIR / "residual_correlations.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")

    # — Residual distribution + boxplot by early/late group
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    df["residual"].plot(kind="hist", bins=20, ax=ax, edgecolor="black", color="#3498db")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="On time")
    ax.set_xlabel("Actual age − Predicted age (age buckets)")
    ax.set_title("Distribution of residuals")
    ax.legend()

    ax = axes[1]
    col = "total_home_training_hours"
    if col in df.columns:
        df.boxplot(column=col, by="achieved_early", ax=ax,
                   boxprops=dict(color="#2c3e50"),
                   medianprops=dict(color="#e74c3c"))
        ax.set_xticklabels(["Later than predicted\n(residual ≥ 0)",
                             "Earlier than predicted\n(residual < 0)"])
        ax.set_title("Home training hours by achievement group")
        ax.set_xlabel("")
        plt.suptitle("")

    plt.tight_layout()
    out = OUTPUT_DIR / "early_vs_late.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ════════════════════════════════════════════════════════════════════════════
# STEP 6 — Ridge regression on residuals
# ════════════════════════════════════════════════════════════════════════════

def run_residual_regression(df: pd.DataFrame) -> None:
    available = [c for c in TRAINING_FEATURES if c in df.columns]
    df_clean = df[available + ["residual"]].dropna()

    if len(df_clean) < 5:
        print("Not enough data for residual regression — skipping.")
        return

    X = df_clean[available].fillna(0)
    y = df_clean["residual"]

    model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
    model.fit(X, y)

    coef_df = (
        pd.Series(model.coef_, index=available)
        .sort_values()
        .rename("coefficient")
        .to_frame()
    )
    coef_df["interpretation"] = coef_df["coefficient"].apply(
        lambda x: "earlier than predicted" if x < 0 else "later than predicted"
    )

    out = OUTPUT_DIR / "residual_regression.csv"
    coef_df.to_csv(out)
    print(f"\nSaved: {out}")
    print("\n── Ridge regression on residuals ─────────────────────────────")
    print(coef_df.to_string())
    print(f"\n  Best alpha: {model.alpha_:.2f}")


# ════════════════════════════════════════════════════════════════════════════
# STEP 7 — Logistic regression: what predicts beating the prediction?
# ════════════════════════════════════════════════════════════════════════════

def run_logistic(df: pd.DataFrame) -> None:
    available = [c for c in TRAINING_FEATURES if c in df.columns]
    df_clean = df[available + ["achieved_early"]].dropna()

    if len(df_clean) < 10 or df_clean["achieved_early"].nunique() < 2:
        print("Not enough data or class variation for logistic regression — skipping.")
        return

    X = df_clean[available].fillna(0)
    y = df_clean["achieved_early"]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=0.5, random_state=42, max_iter=1000)),
    ])

    cv = min(5, len(y))
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    print(f"\n── Logistic regression (early vs late achiever) ──────────────")
    print(f"  Cross-validated AUC: {scores.mean():.2f} ± {scores.std():.2f}")

    pipe.fit(X, y)
    coef_df = (
        pd.Series(pipe["model"].coef_[0], index=available)
        .sort_values()
        .rename("coefficient")
        .to_frame()
    )
    coef_df["interpretation"] = coef_df["coefficient"].apply(
        lambda x: "more likely earlier" if x > 0 else "more likely later"
    )

    out = OUTPUT_DIR / "logistic_coefficients.csv"
    coef_df.to_csv(out)
    print(f"Saved: {out}")
    print(coef_df.to_string())


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("Loading data...")
    conn = get_connection()
    data = load_data(conn)
    master = build_master_feature_table(data)
    print(f"Master table: {master.shape[0]} rows × {master.shape[1]} columns")

    print("\nApplying ID filter...")
    master = filter_ids(master)

    print("\nBuilding survival dataset...")
    df_surv = build_survival_df(master)
    print(f"  {len(df_surv)} children  |  "
          f"  {df_surv['event'].sum()} achieved milestone  |  "
          f"  {(df_surv['event'] == 0).sum()} censored")

    # ── Choose prediction method ─────────────────────────────────────────────
    # Option A: normative GMFCS-based prediction (recommended if you have norms)
    df = add_gmfcs_prediction(df_surv)
    # Option B: data-driven prediction — uncomment to use instead
    # df = add_model_prediction(df_surv)

    df = add_residuals(df)
    print(f"\nResidual summary:")
    print(df["residual"].describe().round(2))
    print(f"  Earlier than predicted: {df['achieved_early'].sum()} children")
    print(f"  Later than predicted:   {(df['achieved_early'] == 0).sum()} children")

    print("\nPlotting Kaplan-Meier curves...")
    plot_kaplan_meier(df_surv)

    print("\nRunning Cox model...")
    run_cox(df_surv)

    print("\nPlotting residual analysis...")
    plot_residuals(df)

    print("\nRunning Ridge regression on residuals...")
    run_residual_regression(df)

    print("\nRunning logistic regression (early vs late)...")
    run_logistic(df)

    print("\nPlotting residual analysis...")
    plot_residuals(df)

    print("\nPlotting additional analysis...")
    plot_additional_analysis(df)


    print(f"\nDone. All outputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()