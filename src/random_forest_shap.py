"""
random_forest_shap.py
─────────────────────
Random Forest regression with SHAP waterfall plots for feature importance.

Usage
-----
    python src/random_forest_shap.py

Outputs (saved to outputs/rf_shap/)
-------------------------------------
    waterfall_mean.png        — SHAP waterfall for the mean prediction
    waterfall_<i>.png         — SHAP waterfall for each sample (if N_WATERFALL_SAMPLES > 0)
    beeswarm.png              — SHAP beeswarm summary plot
    pred_vs_actual.png        — Predicted vs actual scatter
    feature_importance.png    — Mean |SHAP| bar chart
    cv_scores.csv             — Cross-validation R² and MAE
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.connect_db import get_connection
from src.dataloader import load_data
from src.preprocessing.master_preprocessing import build_master_feature_table

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "rf_shap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these to choose inputs, output, and which children to include
# ════════════════════════════════════════════════════════════════════════════

# Target variable (what the model predicts)
TARGET = "delta_motorical_score"

# Input features (column names from the master feature table)
INPUT_FEATURES: list[str] = [
    "active_total_hours",
    "log_total_home_training_hours",
    "log_total_other_training_hours",
    "log_neurohab_hours",
    "has_any_medical_treatment",
    "gmfcs_int",
]

# ID filter — set to a list of introductory_id strings to restrict the analysis,
# or leave as None to include all children.
#
# Examples:
#   INCLUDE_IDS = None                         # all children
#   INCLUDE_IDS = ["abc123", "def456"]         # only these two
#
INCLUDE_IDS: list[str] | None = [
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
]

# Random Forest hyperparameters
RF_PARAMS: dict = dict(
    n_estimators=300,
    max_depth=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

# How many individual waterfall plots to save (0 = skip, -1 = all samples)
N_WATERFALL_SAMPLES: int = 5


# ════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def filter_ids(df: pd.DataFrame) -> pd.DataFrame:
    total = df["introductory_id"].nunique()
    if INCLUDE_IDS is not None:
        df = df[df["introductory_id"].isin(INCLUDE_IDS)]
        missing = [i for i in INCLUDE_IDS if i not in df["introductory_id"].values]
        if missing:
            print(f"  Warning: IDs not found in data: {missing}")
    remaining = df["introductory_id"].nunique()
    suffix = " (all children included)" if INCLUDE_IDS is None else f"  (filter list: {len(INCLUDE_IDS)} IDs)"
    print(f"  ID filter: {total} → {remaining} children{suffix}")
    return df


def build_dataset(master) -> tuple[pd.DataFrame, pd.Series]:
    df = master.to_pandas()
    df = filter_ids(df)

    available_features = [c for c in INPUT_FEATURES if c in df.columns]
    missing_features = [c for c in INPUT_FEATURES if c not in df.columns]
    if missing_features:
        print(f"  Warning: features not found in master table (skipped): {missing_features}")

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in master table.")

    df = df[available_features + [TARGET]].dropna(subset=[TARGET])
    df[available_features] = df[available_features].fillna(0)

    X = df[available_features]
    y = df[TARGET]
    return X, y


# ════════════════════════════════════════════════════════════════════════════
# MODEL
# ════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    n = len(X)
    cv = LeaveOneOut() if n <= 20 else 5
    cv_label = "LOO-CV" if n <= 20 else "5-fold CV"

    model = RandomForestRegressor(**RF_PARAMS)

    r2_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    mae_scores = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")

    model.fit(X, y)
    train_r2 = r2_score(y, model.predict(X))

    print(f"\n  [{cv_label}, n={n}]")
    print(f"    CV R²   : {r2_scores.mean():.4f}  ±  {r2_scores.std():.4f}")
    print(f"    CV MAE  : {mae_scores.mean():.4f}  ±  {mae_scores.std():.4f}")
    print(f"    Train R²: {train_r2:.4f}  (in-sample)")

    scores_df = pd.DataFrame({
        "metric": ["CV R² mean", "CV R² std", "CV MAE mean", "CV MAE std", "Train R²"],
        "value": [r2_scores.mean(), r2_scores.std(), mae_scores.mean(), mae_scores.std(), train_r2],
    })
    scores_df.to_csv(OUTPUT_DIR / "cv_scores.csv", index=False)
    print(f"  Saved: cv_scores.csv")

    return model


# ════════════════════════════════════════════════════════════════════════════
# PLOTS
# ════════════════════════════════════════════════════════════════════════════

def plot_pred_vs_actual(model: RandomForestRegressor, X: pd.DataFrame, y: pd.Series) -> None:
    y_pred = model.predict(X)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y, y_pred, alpha=0.65, color="steelblue", edgecolors="white", s=60)
    lims = [min(y.min(), y_pred.min()) - 0.05, max(y.max(), y_pred.max()) + 0.05]
    ax.plot(lims, lims, "k--", linewidth=1, label="Perfect fit")
    ax.set_xlabel(f"Actual  {TARGET}")
    ax.set_ylabel(f"Predicted  {TARGET}")
    ax.set_title(f"Random Forest — Predicted vs Actual\n(train R² = {r2_score(y, y_pred):.3f})")
    ax.legend()
    plt.tight_layout()
    path = OUTPUT_DIR / "pred_vs_actual.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: pred_vs_actual.png")


def plot_shap_beeswarm(shap_values: shap.Explanation, X: pd.DataFrame) -> None:
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="dot", max_display=20, show=False)
    plt.title("Random Forest — SHAP Beeswarm")
    plt.tight_layout()
    path = OUTPUT_DIR / "beeswarm.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: beeswarm.png")


def plot_shap_feature_importance(shap_values: shap.Explanation) -> None:
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    importance = pd.Series(mean_abs, index=shap_values.feature_names).sort_values()

    fig, ax = plt.subplots(figsize=(7, max(3, len(importance) * 0.55)))
    importance.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Random Forest — Mean |SHAP| Feature Importance")
    ax.set_xlabel("Mean |SHAP value|")
    plt.tight_layout()
    path = OUTPUT_DIR / "feature_importance.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: feature_importance.png")


def plot_waterfall_mean(shap_values: shap.Explanation) -> None:
    """Waterfall plot for the mean SHAP explanation across all samples."""
    mean_explanation = shap.Explanation(
        values=shap_values.values.mean(axis=0),
        base_values=shap_values.base_values.mean(),
        data=shap_values.data.mean(axis=0),
        feature_names=shap_values.feature_names,
    )
    plt.figure()
    shap.plots.waterfall(mean_explanation, max_display=20, show=False)
    plt.title("Random Forest — SHAP Waterfall (mean over all samples)")
    plt.tight_layout()
    path = OUTPUT_DIR / "waterfall_mean.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: waterfall_mean.png")


def plot_waterfall_samples(shap_values: shap.Explanation, n: int) -> None:
    """Waterfall plots for individual samples."""
    if n == 0:
        return
    indices = range(len(shap_values)) if n == -1 else range(min(n, len(shap_values)))
    for i in indices:
        plt.figure()
        shap.plots.waterfall(shap_values[i], max_display=20, show=False)
        plt.title(f"Random Forest — SHAP Waterfall (sample {i})")
        plt.tight_layout()
        path = OUTPUT_DIR / f"waterfall_{i}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
    print(f"  Saved: waterfall_0 … waterfall_{list(indices)[-1]}.png")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("  Random Forest + SHAP")
    print("=" * 60)

    print("\n  Loading data …")
    conn = get_connection()
    data = load_data(conn)
    master = build_master_feature_table(data)
    print(f"  Master table: {master.shape[0]} rows × {master.shape[1]} columns")

    print("\n  Preparing dataset …")
    X, y = build_dataset(master)
    print(f"  Rows: {len(X)}  |  Features: {list(X.columns)}  |  Target: {TARGET}")
    print(f"  Target range: [{y.min():.3f}, {y.max():.3f}]  "
          f"mean: {y.mean():.4f}  std: {y.std():.4f}")

    print("\n  Training and evaluating Random Forest …")
    model = train_and_evaluate(X, y)

    print("\n  Computing SHAP values …")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    print("\n  Generating plots …")
    plot_pred_vs_actual(model, X, y)
    plot_shap_beeswarm(shap_values, X)
    plot_shap_feature_importance(shap_values)
    plot_waterfall_mean(shap_values)
    plot_waterfall_samples(shap_values, N_WATERFALL_SAMPLES)

    print(f"\n  Done. All outputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
