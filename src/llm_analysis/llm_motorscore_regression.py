"""
LLM Motor Score Regression Analysis
=====================================
Merges the LLM-derived motor score (1–10) from the CSV output with the
master feature table and runs linear regression analyses.

The LLM motor score is used as the TARGET (absolute level, not delta),
since it already represents a snapshot assessment at a given age.

Usage:
    cd src
    python ../llm_motorscore_regression.py

Output:
    results/llm_motorscore_regression_results.csv
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import r2_score

SRC = Path(__file__).resolve().parent / "src"
sys.path.append(str(SRC))

from src.connect_db import get_connection
from src.dataloader import load_data
from src.preprocessing.master_preprocessing import build_master_feature_table

# ─── Paths ────────────────────────────────────────────────────────────────────

_ROOT         = Path(__file__).resolve().parents[2]
LLM_SCORE_CSV = _ROOT / "outputs" / "motorscore_analysis" / "llm_motorscore_results.csv"
RESULTS_DIR   = _ROOT / "outputs" / "llm_motorscore_regression" / "results"
FIGURES_DIR   = _ROOT / "outputs" / "llm_motorscore_regression" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────

TARGET = "llm_motor_score"


FILTER_INTRODUCTORY_IDS: list[str] = [
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

# Feature sets to test — comment/uncomment as needed
FEATURE_SETS: dict[str, list[str]] = {
    "training_hours_raw": [
        "total_home_training_hours",
        "total_other_training_hours",
        "neurohab_hours",
    ],
    "training_hours_log": [
        "log_total_home_training_hours",
        "log_total_other_training_hours",
        "log_neurohab_hours",
    ],
    "training_categories_log": [
        "log_cat_neurodevelopmental_reflex",
        "log_cat_motor_learning_task",
        "log_cat_technology_assisted",
        "log_cat_suit_based",
        "log_cat_physical_conditioning",
        "log_cat_complementary",
    ],
    "devices_and_medical": [
        "has_any_device",
        "has_any_medical_treatment",
    ],
    "full_model": [
        "log_total_home_training_hours",
        "log_total_other_training_hours",
        "log_neurohab_hours",
        "has_any_device",
        "has_any_medical_treatment",
        "gmfcs_int",
    ],
    "full_model_with_categories": [
        "log_total_home_training_hours",
        "log_total_other_training_hours",
        "log_neurohab_hours",
        "log_cat_neurodevelopmental_reflex",
        "log_cat_motor_learning_task",
        "log_cat_technology_assisted",
        "log_cat_suit_based",
        "log_cat_physical_conditioning",
        "log_cat_complementary",
        "has_any_device",
        "has_any_medical_treatment",
        "gmfcs_int",
    ],
}

FEATURE_LABELS = {
    "total_home_training_hours":          "Home training (hrs/yr)",
    "total_other_training_hours":         "Other training (hrs/yr)",
    "neurohab_hours":                     "Intensive therapy (hrs/yr)",
    "log_total_home_training_hours":      "log Home training",
    "log_total_other_training_hours":     "log Other training",
    "log_neurohab_hours":                 "log Intensive therapy",
    "log_active_total_hours":             "log Total active hours",
    "has_any_device":                     "Uses any device",
    "has_any_medical_treatment":          "Any medical treatment",
    "gmfcs_int":                          "GMFCS level",
    "log_cat_neurodevelopmental_reflex":  "log Neurodevelopmental/reflex",
    "log_cat_motor_learning_task":        "log Motor learning",
    "log_cat_technology_assisted":        "log Technology assisted",
    "log_cat_suit_based":                 "log Suit based",
    "log_cat_physical_conditioning":      "log Physical conditioning",
    "log_cat_complementary":              "log Complementary",
}

MIN_N = 8


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_llm_scores() -> pd.DataFrame:
    if not LLM_SCORE_CSV.exists():
        raise FileNotFoundError(
            f"LLM score CSV not found: {LLM_SCORE_CSV}\n"
            "Run src/llm_analysis/llm_motorscore_analysis.py first."
        )
    df = pd.read_csv(LLM_SCORE_CSV)
    print(f"  LLM score CSV loaded: {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    return df


def load_master() -> pd.DataFrame:
    conn   = get_connection()
    data   = load_data(conn)
    master = build_master_feature_table(data)
    return master.to_pandas()


def build_merged_dataset(llm_df: pd.DataFrame, master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge LLM scores with master features on (introductory_id, age).
    """
    # Ensure consistent types
    llm_df    = llm_df.copy()
    master_df = master_df.copy()

    llm_df["introductory_id"] = llm_df["introductory_id"].astype(str)
    llm_df["age"]             = llm_df["age"].astype(int)
    master_df["introductory_id"] = master_df["introductory_id"].astype(str)
    master_df["age"]             = master_df["age"].astype(int)

    merged = llm_df[["introductory_id", "age", "llm_motor_score", "confidence"]].merge(
        master_df,
        on=["introductory_id", "age"],
        how="inner",
    )

    if FILTER_INTRODUCTORY_IDS:
        merged = merged[merged["introductory_id"].isin(FILTER_INTRODUCTORY_IDS)]
        print(f"  Filtered to {len(FILTER_INTRODUCTORY_IDS)} IDs → {len(merged)} rows")

    print(f"  Merged dataset: {len(merged)} rows, {merged['introductory_id'].nunique()} children")
    return merged


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _sig(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def _cv_score(model, X: pd.DataFrame, y: pd.Series) -> dict:
    n  = len(X)
    cv = LeaveOneOut() if n <= 20 else 5
    r2   = cross_val_score(model, X, y, cv=cv, scoring="r2")
    mae  = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
    return {"cv_r2_mean": r2.mean(), "cv_r2_std": r2.std(), "cv_mae_mean": mae.mean()}


# ─── Analysis 1: OLS per feature set ─────────────────────────────────────────

def run_ols_per_feature_set(df: pd.DataFrame) -> list[dict]:
    sep = "=" * 70
    print(f"\n{sep}")
    print("  OLS REGRESSION — LLM MOTOR SCORE vs TRAINING FEATURES")
    print(sep)

    all_rows = []

    for set_name, features in FEATURE_SETS.items():
        available = [f for f in features if f in df.columns]
        if not available:
            print(f"\n  [{set_name}] — no features available, skipping.")
            continue

        subset = df[available + [TARGET]].dropna()
        subset[available] = subset[available].fillna(0)

        if len(subset) < MIN_N:
            print(f"\n  [{set_name}] — too few rows ({len(subset)}), skipping.")
            continue

        X     = sm.add_constant(subset[available])
        y     = subset[TARGET]
        model = sm.OLS(y, X).fit()

        print(f"\n  Feature set: {set_name}  |  n={len(subset)}  |  Adj. R²={model.rsquared_adj:.4f}  |  F-p={model.f_pvalue:.4f} {_sig(model.f_pvalue)}")
        print(f"  {'Feature':<40} {'β':>10} {'95% CI':>24} {'p':>8}")
        print(f"  {'-'*84}")

        for feat in available:
            coef       = model.params[feat]
            ci_lo, ci_hi = model.conf_int().loc[feat]
            pval       = model.pvalues[feat]
            ci_str     = f"[{ci_lo:+.4f}, {ci_hi:+.4f}]"
            fname      = FEATURE_LABELS.get(feat, feat)
            print(f"  {fname:<40} {coef:>+10.5f}  {ci_str:>24}  {pval:>7.4f} {_sig(pval)}")

            all_rows.append({
                "feature_set":   set_name,
                "feature":       feat,
                "feature_label": FEATURE_LABELS.get(feat, feat),
                "n":             len(subset),
                "beta":          round(coef, 6),
                "ci_low":        round(ci_lo, 6),
                "ci_high":       round(ci_hi, 6),
                "p_value":       round(pval, 6),
                "significant":   pval < 0.05,
                "adj_r2":        round(model.rsquared_adj, 4),
                "f_pvalue":      round(model.f_pvalue, 6),
            })

    print(f"\n  Significance: * p<0.05  ** p<0.01  *** p<0.001")
    return all_rows


# ─── Analysis 2: Spearman correlations ───────────────────────────────────────

def run_spearman_correlations(df: pd.DataFrame) -> list[dict]:
    sep = "=" * 70
    print(f"\n{sep}")
    print("  SPEARMAN CORRELATIONS — LLM MOTOR SCORE vs EACH FEATURE")
    print(sep)

    all_features = list({f for fs in FEATURE_SETS.values() for f in fs})
    available    = [f for f in all_features if f in df.columns]

    rows = []
    print(f"\n  {'Feature':<40} {'ρ':>8} {'p':>10} {'n':>6}")
    print(f"  {'-'*66}")

    results = []
    for feat in available:
        subset = df[[feat, TARGET]].dropna()
        subset = subset.fillna(0)
        if len(subset) < MIN_N:
            continue
        r, p = stats.spearmanr(subset[feat], subset[TARGET])
        results.append((feat, r, p, len(subset)))

    results.sort(key=lambda x: abs(x[1]), reverse=True)

    for feat, r, p, n in results:
        fname = FEATURE_LABELS.get(feat, feat)
        print(f"  {fname:<40} {r:>+8.4f} {p:>10.4f}{_sig(p):>3}  {n:>5}")
        rows.append({
            "feature":       feat,
            "feature_label": FEATURE_LABELS.get(feat, feat),
            "spearman_rho":  round(r, 6),
            "p_value":       round(p, 6),
            "n":             n,
            "significant":   p < 0.05,
        })

    return rows


# ─── Analysis 3: Random Forest cross-validated ───────────────────────────────

def run_random_forest(df: pd.DataFrame) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print("  RANDOM FOREST — FEATURE IMPORTANCE (full model)")
    print(sep)

    features  = FEATURE_SETS["full_model_with_categories"]
    available = [f for f in features if f in df.columns]
    subset    = df[available + [TARGET]].dropna()
    subset    = subset.copy()
    subset[available] = subset[available].fillna(0)

    if len(subset) < MIN_N:
        print(f"  Too few rows ({len(subset)}) — skipping.")
        return

    X = subset[available]
    y = subset[TARGET]

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    cv_res = _cv_score(rf, X, y)
    rf.fit(X, y)

    print(f"\n  n={len(subset)}  |  CV R²={cv_res['cv_r2_mean']:.4f} ± {cv_res['cv_r2_std']:.4f}  |  CV MAE={cv_res['cv_mae_mean']:.4f}")

    importance = (
        pd.Series(rf.feature_importances_, index=X.columns)
        .rename(index=FEATURE_LABELS)
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(8, max(4, len(importance) * 0.45)))
    importance.plot(kind="barh", ax=ax, color="steelblue", alpha=0.8)
    ax.set_title(f"Random Forest — Feature Importance\n(CV R²={cv_res['cv_r2_mean']:.3f}, n={len(subset)})")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    path = FIGURES_DIR / "llm_motorscore_rf_importance.png"
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path.name}")
    plt.show()


# ─── Analysis 4: Predicted vs actual for best OLS model ──────────────────────

def plot_predicted_vs_actual(df: pd.DataFrame) -> None:
    features  = FEATURE_SETS["full_model"]
    available = [f for f in features if f in df.columns]
    subset    = df[available + [TARGET]].dropna().copy()
    subset[available] = subset[available].fillna(0)

    if len(subset) < MIN_N:
        return

    X       = sm.add_constant(subset[available])
    y       = subset[TARGET]
    model   = sm.OLS(y, X).fit()
    y_pred  = model.predict(X)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y, y_pred, alpha=0.7, color="steelblue", edgecolors="white", s=60)
    lims = [min(y.min(), y_pred.min()) - 0.3, max(y.max(), y_pred.max()) + 0.3]
    ax.plot(lims, lims, "k--", linewidth=1, label="Perfect fit")
    ax.set_xlabel("Actual LLM motor score")
    ax.set_ylabel("Predicted LLM motor score")
    ax.set_title(f"OLS Full Model — Predicted vs Actual\n(Adj. R²={model.rsquared_adj:.3f}, n={len(subset)})")
    ax.legend()
    plt.tight_layout()
    path = FIGURES_DIR / "llm_motorscore_pred_vs_actual.png"
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path.name}")
    plt.show()


# ─── Analysis 5: GMFCS stratified correlations ───────────────────────────────

def run_gmfcs_stratified(df: pd.DataFrame) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print("  GMFCS-STRATIFIED SPEARMAN CORRELATIONS")
    print("  (does training matter more at certain severity levels?)")
    print(sep)

    if "gmfcs_int" not in df.columns:
        print("  gmfcs_int not found — skipping.")
        return

    features = [
        "log_total_home_training_hours",
        "log_total_other_training_hours",
        "log_neurohab_hours",
    ]
    available = [f for f in features if f in df.columns]

    for feat in available:
        fname = FEATURE_LABELS.get(feat, feat)
        print(f"\n  {fname}")
        print(f"  {'GMFCS':>8} {'ρ':>8} {'p':>10} {'n':>6}")
        print(f"  {'-'*36}")
        for lvl in sorted(df["gmfcs_int"].dropna().unique()):
            grp = df[df["gmfcs_int"] == lvl][[feat, TARGET]].dropna()
            if len(grp) < 4:
                continue
            r, p = stats.spearmanr(grp[feat], grp[TARGET])
            print(f"  {int(lvl):>8} {r:>+8.4f} {p:>10.4f}{_sig(p):>3}  {len(grp):>5}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    sep = "=" * 70
    print(sep)
    print("  LLM MOTOR SCORE REGRESSION ANALYSIS")
    print(sep)

    print("\n  Loading LLM scores …")
    llm_df = load_llm_scores()

    print("\n  Loading master feature table …")
    master_df = load_master()
    print(f"  Master table: {len(master_df)} rows × {len(master_df.columns)} columns")

    print("\n  Merging …")
    df = build_merged_dataset(llm_df, master_df)

    print(f"\n  LLM motor score distribution:")
    print(f"    min={df[TARGET].min():.1f}  max={df[TARGET].max():.1f}"
          f"  mean={df[TARGET].mean():.2f}  std={df[TARGET].std():.2f}")

    # ── Run analyses ────────────────────────────────────────────────────────
    ols_rows       = run_ols_per_feature_set(df)
    spearman_rows  = run_spearman_correlations(df)
    run_random_forest(df)
    plot_predicted_vs_actual(df)
    run_gmfcs_stratified(df)

    # ── Save results to CSV ─────────────────────────────────────────────────
    ols_df      = pd.DataFrame(ols_rows)
    spearman_df = pd.DataFrame(spearman_rows)

    ols_path      = RESULTS_DIR / "llm_motorscore_ols_results.csv"
    spearman_path = RESULTS_DIR / "llm_motorscore_spearman_results.csv"
    merged_path   = RESULTS_DIR / "llm_motorscore_merged_dataset.csv"

    ols_df.to_csv(ols_path, index=False)
    spearman_df.to_csv(spearman_path, index=False)
    df.to_csv(merged_path, index=False)

    print(f"\n{sep}")
    print("  Output files saved:")
    print(f"    {ols_path}")
    print(f"    {spearman_path}")
    print(f"    {merged_path}")
    print(sep)
    print("\nDone.")


if __name__ == "__main__":
    main()