"""
Model: Predicting delta_milestone_score_setvalue
================================================
Inputs:
  - total_home_training_hours
  - total_other_training_hours
  - neurohab_hours
  - botox (binary, auto-detected from med_* columns)

Output:
  - delta_milestone_score_setvalue

Usage:
    cd src
    python ../model_milestone_delta.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import r2_score

# ── resolve src/ on the path ────────────────────────────────────────────────
SRC = Path(__file__).resolve().parent / "src"
sys.path.append(str(SRC))

from src.connect_db import get_connection
from src.dataloader import load_data
from src.preprocessing.master_preprocessing import build_master_feature_table

FIGURES_DIR = Path(__file__).resolve().parents[2] / "outputs" / "model_comparison_filtered"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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

TARGET = "delta_milestone_score_setvalue"
TRAINING_FEATURES = [
    #"age",
    #"gmfcs_int",
    #"total_home_training_hours",
    #"total_other_training_hours",
    #"active_total_hours",
    #"neurohab_hours",
    #"device_AFOs",
    #"device_Kinesiotaping",
    #"device_Handsplint",
    #"device_Standing frame",
    #"device_Turtlebrace",
    #"device_inga",
    #"has_any_device",
    #"med_Botulinum toxin (Botox) injections",
    #"med_No",
    #"med_Hand casting",
    #"med_Orthopedic surgery (e.g., tendon lengthening, hip surgery)",
    #"med_Leg casting",
    #"has_any_medical_treatment",
    "log_total_home_training_hours",
    "log_total_other_training_hours",
    #"log_active_total_hours",
    "log_neurohab_hours",
    # "cat_neurodevelopmental_reflex",
    # "cat_motor_learning_task",
    # "cat_technology_assisted",
    # "cat_suit_based",
    # "cat_physical_conditioning",
    # "cat_complementary",
    # "log_cat_neurodevelopmental_reflex",
    # "log_cat_motor_learning_task",
    # "log_cat_technology_assisted",
    # "log_cat_suit_based",
    # "log_cat_physical_conditioning",
    # "log_cat_complementary",
    ]


def build_dataset(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select features and target, drop rows where target is null."""
    feature_cols = TRAINING_FEATURES.copy()

    # convert to pandas
    df = master.to_pandas()

    # Filter to specific introductory IDs if list is non-empty
    if FILTER_INTRODUCTORY_IDS:
        df = df[df["introductory_id"].isin(FILTER_INTRODUCTORY_IDS)]
        print(f"  Filtered to {len(df)} rows for {len(FILTER_INTRODUCTORY_IDS)} introductory IDs")


    # keep only rows where the target exists
    df = df[feature_cols + [TARGET]].dropna(subset=[TARGET])

    # fill remaining NaN in features with 0
    df[feature_cols] = df[feature_cols].fillna(0)

    # ── Drop zero outcomes and zero-training rows ────────────────────────────
    n_before = len(df)
    df = df[
        (df[TARGET] != 0) &                        # remove zero outcome (likely missing second assessment)
        (df[feature_cols].sum(axis=1) > 0)         # remove zero training (likely missing data)
    ]
    n_after = len(df)
    print(f"  [build_dataset] Dropped {n_before - n_after} rows "
          f"(zero outcome or zero training), {n_after} remaining")
    # ─────────────────────────────────────────────────────────────────────────

    X = df[feature_cols]
    y = df[TARGET]
    return X, y


# ── model evaluation ─────────────────────────────────────────────────────────

def evaluate_model(name: str, model, X: pd.DataFrame, y: pd.Series) -> dict:
    """Cross-validate and return a result dict."""
    n = len(X)

    # Use LeaveOneOut when dataset is tiny (<= 20 rows), else 5-fold CV
    if n <= 20:
        cv = LeaveOneOut()
        cv_label = "LOO-CV"
    else:
        cv = 5
        cv_label = "5-fold CV"

    r2_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    mae_scores = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")

    # fit on full data for inspection
    model.fit(X, y)
    y_pred = model.predict(X)
    train_r2 = r2_score(y, y_pred)

    print(f"\n  [{name}]  ({cv_label}, n={n})")
    print(f"    CV R²  : {r2_scores.mean():.4f}  ±  {r2_scores.std():.4f}")
    print(f"    CV MAE : {mae_scores.mean():.4f}  ±  {mae_scores.std():.4f}")
    print(f"    Train R²: {train_r2:.4f}  (in-sample, informational only)")

    return {
        "name": name,
        "model": model,
        "cv_r2_mean": r2_scores.mean(),
        "cv_r2_std": r2_scores.std(),
        "cv_mae_mean": mae_scores.mean(),
        "train_r2": train_r2,
        "X": X,
        "y": y,
    }


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_predicted_vs_actual(result: dict):
    model = result["model"]
    X, y = result["X"], result["y"]
    y_pred = model.predict(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y, y_pred, alpha=0.6, color="steelblue", edgecolors="white", s=60)
    lims = [min(y.min(), y_pred.min()) - 0.05, max(y.max(), y_pred.max()) + 0.05]
    ax.plot(lims, lims, "k--", linewidth=1, label="Perfect fit")
    ax.set_xlabel("Actual Δ motorical score")
    ax.set_ylabel("Predicted Δ motorical score")
    ax.set_title(f"{result['name']} — Predicted vs Actual\n(train R² = {result['train_r2']:.3f})")
    ax.legend()
    plt.tight_layout()
    path = FIGURES_DIR / f"pred_vs_actual_{result['name'].replace(' ', '_')}.png"
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path.name}")
    plt.show()


def plot_feature_importance(result: dict):
    model = result["model"]
    X = result["X"]

    if not hasattr(model, "feature_importances_"):
        print(f"  [{result['name']}] No feature importances available — skipping plot.")
        return

    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values()

    fig, ax = plt.subplots(figsize=(7, max(3, len(X.columns) * 0.6)))
    importance.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"{result['name']} — Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    path = FIGURES_DIR / f"feature_importance_{result['name'].replace(' ', '_')}.png"
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path.name}")
    plt.show()


def plot_shap(result: dict):
    model = result["model"]
    X = result["X"]

    if not isinstance(model, (RandomForestRegressor, GradientBoostingRegressor)):
        print(f"  [{result['name']}] SHAP only for tree models — skipping.")
        return

    print(f"  Computing SHAP values for [{result['name']}] …")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="dot", max_display=10, show=False)
    plt.title(f"{result['name']} — SHAP Beeswarm")
    plt.tight_layout()
    path = FIGURES_DIR / f"shap_{result['name'].replace(' ', '_')}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path.name}")
    plt.show()


def plot_partial_dependence(result: dict):
    """Simple 1-D partial dependence for each feature."""
    model = result["model"]
    X = result["X"]

    n_cols = len(X.columns)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), sharey=False)
    if n_cols == 1:
        axes = [axes]

    for ax, col in zip(axes, X.columns):
        grid = np.linspace(X[col].min(), X[col].max(), 50)
        X_copy = X.copy()
        preds = []
        for val in grid:
            X_copy[col] = val
            preds.append(model.predict(X_copy).mean())

        ax.plot(grid, preds, color="darkorange", linewidth=2)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_xlabel(col)
        ax.set_ylabel("Avg predicted Δ motorical score")
        ax.set_title(col)

    fig.suptitle(f"{result['name']} — Partial Dependence", fontsize=13)
    plt.tight_layout()
    path = FIGURES_DIR / f"partial_dep_{result['name'].replace(' ', '_')}.png"
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path.name}")
    plt.show()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Loading data …")
    conn = get_connection()
    data = load_data(conn)

    print("  Building master feature table …")
    master = build_master_feature_table(data)

    print(f"\n  Master table shape: {master.shape[0]} rows × {master.shape[1]} columns")

    print("\n  Selecting features and target …")
    X, y = build_dataset(master)

    print(f"\n  Dataset ready: {len(X)} rows, {len(X.columns)} features")
    print(f"  Features : {list(X.columns)}")
    print(f"  Target   : {TARGET}")
    print(f"  Target range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"  Target mean : {y.mean():.4f},  std: {y.std():.4f}")

    # ── models to compare ────────────────────────────────────────────────────
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=3,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.05,
            min_samples_leaf=2,
            random_state=42,
        ),
    }

    print("\n" + "=" * 60)
    print("  Model evaluation")
    print("=" * 60)

    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(name, model, X, y)

    # ── summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    summary = pd.DataFrame([
        {
            "Model": r["name"],
            "CV R² (mean)": round(r["cv_r2_mean"], 4),
            "CV R² (std)": round(r["cv_r2_std"], 4),
            "CV MAE (mean)": round(r["cv_mae_mean"], 4),
        }
        for r in results.values()
    ]).sort_values("CV R² (mean)", ascending=False)
    print(summary.to_string(index=False))

    # pick best by CV R²
    best_name = summary.iloc[0]["Model"]
    best = results[best_name]
    print(f"\n  Best model: {best_name}  (CV R² = {best['cv_r2_mean']:.4f})")

    # ── linear regression coefficients (always interpretable) ────────────────
    lr = results["Linear Regression"]["model"]
    print("\n  Linear Regression coefficients:")
    for col, coef in zip(X.columns, lr.coef_):
        print(f"    {col:<40} {coef:+.5f}")
    print(f"    {'Intercept':<40} {lr.intercept_:+.5f}")

    # ── plots ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Generating plots …")
    print("=" * 60)

    for result in results.values():
        plot_predicted_vs_actual(result)
        plot_feature_importance(result)
        plot_partial_dependence(result)

    # Kör SHAP på bästa trädmodell
    tree_preference = ["Random Forest", "Gradient Boosting"]
    best_tree_name = next((n for n in tree_preference if n in results), None)
    if best_tree_name:
        plot_shap(results[best_tree_name])
    else:
        print("  Inga trädmodeller tillgängliga för SHAP.")

    plt.show()
    print("\nDone.")


if __name__ == "__main__":
    main()