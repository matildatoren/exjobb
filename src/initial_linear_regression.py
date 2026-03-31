import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

from preprocessing.preprocessing_it import (
    process_neurohab_hours_per_user_per_age,
    process_medical_treatments_per_user_per_age,
)

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = BASE_DIR / "images"


# -------------------------------------------------------
# Build dose-response dataset
# -------------------------------------------------------


def build_dose_response_dataset(
    motor_df: pl.DataFrame,
    home_df: pl.DataFrame,
    score_col: str = "milestone_score",
) -> pd.DataFrame:
    """
    Join delta score with total training hours per child per year.

    Args:
        motor_df:   Motor score DataFrame with score_col per child and age.
        home_df:    Home training data with therapy hours per child, age, category.
        score_col:  Name of the score column to compute delta from.

    Returns:
        pd.DataFrame with delta_score, total_training_hours(_hr), and
        one column per training category.
    """
    motor_df = motor_df.sort(["introductory_id", "age"])
    motor_df = motor_df.with_columns(
        (pl.col(score_col) - pl.col(score_col).shift(1))
        .over("introductory_id")
        .alias("delta_score")
    ).filter(pl.col("delta_score").is_not_null())

    total_hours = home_df.group_by(["introductory_id", "age"]).agg(
        pl.sum("total_hours").alias("total_training_hours")
    )

    category_hours = (
        home_df.group_by(["introductory_id", "age", "training_category"])
        .agg(pl.sum("total_hours").alias("hours"))
        .pivot(values="hours", index=["introductory_id", "age"], on="training_category")
        .fill_null(0)
    )

    df = (
        motor_df.join(total_hours, on=["introductory_id", "age"], how="left")
        .join(category_hours, on=["introductory_id", "age"], how="left")
        .fill_null(0)
    )

    df = df.with_columns(
        (pl.col("total_training_hours") / 60).alias("total_training_hours_hr")
    )

    return df.to_pandas()


# -------------------------------------------------------
# Build combined active hours dataset (no devices)
# -------------------------------------------------------


def build_active_hours_dataset(
    motor_df: pl.DataFrame,
    home_df: pl.DataFrame,
    neurohab_df: pl.DataFrame,
    medical_df: pl.DataFrame,
    score_col: str = "milestone_score",
) -> pd.DataFrame:
    """
    Combine home training, sports/other, and intensive therapy hours.
    Devices are excluded. Binary indicators for medical treatments included.

    Args:
        motor_df:   Motor score DataFrame with score_col per child and age.
        home_df:    Home training data.
        neurohab_df: Intensive therapy hours per child and age.
        medical_df: Binary medical treatment indicators per child and age.
        score_col:  Name of the score column to compute delta from.

    Returns:
        pd.DataFrame with delta_score, component hours, active_total,
        and medical treatment columns.
    """
    motor_df = motor_df.sort(["introductory_id", "age"])
    motor_df = motor_df.with_columns(
        (pl.col(score_col) - pl.col(score_col).shift(1))
        .over("introductory_id")
        .alias("delta_score")
    ).filter(pl.col("delta_score").is_not_null())

    home_hours = (
        home_df.filter(pl.col("training_category") == "home")
        .group_by(["introductory_id", "age"])
        .agg(pl.sum("total_hours").alias("home_hours"))
    )

    sports_hours = (
        home_df.filter(pl.col("training_category") == "other")
        .group_by(["introductory_id", "age"])
        .agg(pl.sum("total_hours").alias("sports_hours"))
    )

    neurohab_hours = neurohab_df.group_by(["introductory_id", "age"]).agg(
        pl.sum("total_hours").alias("neurohab_hours")
    )

    df = (
        motor_df.join(home_hours, on=["introductory_id", "age"], how="left")
        .join(sports_hours, on=["introductory_id", "age"], how="left")
        .join(neurohab_hours, on=["introductory_id", "age"], how="left")
        .join(medical_df, on=["introductory_id", "age"], how="left")
        .fill_null(0)
    )

    df = df.with_columns(
        (
            pl.col("home_hours") + pl.col("sports_hours") + pl.col("neurohab_hours")
        ).alias("active_total")
    )

    df = df.with_columns(
        [
            (pl.col("home_hours") / 60).alias("home_hours"),
            (pl.col("sports_hours") / 60).alias("sports_hours"),
            (pl.col("neurohab_hours") / 60).alias("neurohab_hours"),
            (pl.col("active_total") / 60).alias("active_total"),
        ]
    )

    return df.to_pandas()


# -------------------------------------------------------
# Regression helpers
# -------------------------------------------------------


def _fit_linear(X: pd.DataFrame, y: pd.Series):
    model = LinearRegression()
    model.fit(X, y)
    r2 = r2_score(y, model.predict(X))
    return model, r2


def fit_linear_dose_response(
    df: pd.DataFrame, feature: str = "total_training_hours_hr"
):
    subset = df[["delta_score", feature]].dropna()
    X, y = subset[[feature]], subset["delta_score"]
    return _fit_linear(X, y)


def fit_polynomial_dose_response(
    df: pd.DataFrame, feature: str = "total_training_hours_hr", degree: int = 2
):
    subset = df[["delta_score", feature]].dropna()
    X, y = subset[[feature]], subset["delta_score"]
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False), LinearRegression()
    )
    model.fit(X, y)
    r2 = r2_score(y, model.predict(X))
    return model, r2


# -------------------------------------------------------
# Run analysis
# -------------------------------------------------------


def run_analysis(
    motor_df: pl.DataFrame,
    home_df: pl.DataFrame,
    neurohab_df: pl.DataFrame,
    medical_df: pl.DataFrame,
    score_col: str = "milestone_score",
) -> dict:
    """
    Run all dose-response regressions for a given score column.

    Args:
        motor_df:   Motor score DataFrame containing score_col.
        home_df:    Home training data.
        neurohab_df: Intensive therapy hours.
        medical_df: Binary medical treatment indicators.
        score_col:  Which score column to analyse ("milestone_score",
                    "mms_normalized", or "combined_score").

    Returns:
        dict with active_df, dose_df, component_results, treatment_results,
        linear/poly model and R² values.
    """
    active_df = build_active_hours_dataset(
        motor_df, home_df, neurohab_df, medical_df, score_col=score_col
    )
    dose_df = build_dose_response_dataset(motor_df, home_df, score_col=score_col)

    known_cols = {
        "introductory_id", "age", score_col,
        "delta_score", "home_hours", "sports_hours",
        "neurohab_hours", "active_total", "cum_unique_milestones",
        "milestone_score", "mms_normalized", "combined_score",
    }
    treatment_cols = [c for c in active_df.columns if c not in known_cols]

    components = {
        "home_hours":     "Home training",
        "sports_hours":   "Sports / other",
        "neurohab_hours": "Intensive therapy",
        "active_total":   "Combined active total",
    }

    component_results = []
    for col, label in components.items():
        subset = active_df[["delta_score", col]].dropna()
        subset = subset[subset[col] >= 0]
        if len(subset) < 5 or subset[col].sum() == 0:
            continue
        model, r2 = _fit_linear(subset[[col]], subset["delta_score"])
        component_results.append(
            {
                "col":        col,
                "label":      label,
                "coeff":      model.coef_[0],
                "intercept":  model.intercept_,
                "r2":         r2,
                "n":          len(subset),
                "mean_hours": round(subset[col].mean(), 1),
            }
        )

    treatment_results = []
    for col in treatment_cols:
        received     = active_df[active_df[col] == 1]["delta_score"].dropna()
        not_received = active_df[active_df[col] == 0]["delta_score"].dropna()
        if len(received) == 0:
            continue
        treatment_results.append(
            {
                "col":        col,
                "n_received": len(received),
                "n_not":      len(not_received),
                "mean_yes":   round(received.mean(), 3),
                "mean_no":    round(not_received.mean(), 3),
                "mean_diff":  round(received.mean() - not_received.mean(), 3),
            }
        )

    linear_model, linear_r2 = fit_linear_dose_response(dose_df)
    poly_model,   poly_r2   = fit_polynomial_dose_response(dose_df)

    return {
        "active_df":         active_df,
        "dose_df":           dose_df,
        "component_results": component_results,
        "treatment_results": treatment_results,
        "treatment_cols":    treatment_cols,
        "linear_model":      linear_model,
        "linear_r2":         linear_r2,
        "poly_model":        poly_model,
        "poly_r2":           poly_r2,
        "score_col":         score_col,
    }


# -------------------------------------------------------
# Print summary
# -------------------------------------------------------


def print_summary(results: dict, title: str = "Motor Score"):
    sep  = "=" * 62
    line = "-" * 62

    print(f"\n{sep}")
    print(f"  DOSE-RESPONSE ANALYSIS — {title.upper()}")
    print(sep)

    print("\n  Active Hours Components (devices excluded)\n")
    header = f"  {'Component':<26} {'Coeff':>8} {'R²':>7} {'N':>6} {'Mean hrs':>10}"
    print(header)
    print(f"  {line}")
    for r in results["component_results"]:
        print(
            f"  {r['label']:<26} "
            f"{r['coeff']:>8.4f} "
            f"{r['r2']:>7.3f} "
            f"{r['n']:>6} "
            f"{r['mean_hours']:>10.1f}"
        )

    print(f"\n  {line}")
    print("  Overall Dose-Response (all home training incl. devices)\n")
    print(f"    Linear model   R² = {results['linear_r2']:.4f}")
    print(f"    Poly model     R² = {results['poly_r2']:.4f}")

    if results["treatment_results"]:
        print(f"\n  {line}")
        print("  Medical Treatments\n")
        header2 = f"  {'Treatment':<28} {'Yes (n)':>8} {'No (n)':>8} {'Diff':>8}"
        print(header2)
        print(f"  {line}")
        for t in results["treatment_results"]:
            print(
                f"  {t['col']:<28} {t['n_received']:>8} {t['n_not']:>8} "
                f"{t['mean_diff']:>+8.3f}"
            )

    print(f"\n{sep}\n")


# -------------------------------------------------------
# Figure 1 — 2×2 training component scatter plots
# -------------------------------------------------------


def plot_training_components(
    results: dict,
    title: str = "Motor Score",
    filename: str = "training_components.png",
):
    active_df = results["active_df"]
    panels    = results["component_results"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, r in enumerate(panels):
        ax  = axes[i]
        col = r["col"]

        subset = active_df[["delta_score", col]].dropna()
        subset = subset[subset[col] >= 0]

        ax.scatter(subset[col], subset["delta_score"], alpha=0.4, color="steelblue", s=30)

        if len(subset) >= 5 and subset[col].sum() > 0:
            model, _ = _fit_linear(subset[[col]], subset["delta_score"])
            x_range    = np.linspace(subset[col].min(), subset[col].max(), 100)
            x_range_df = pd.DataFrame(x_range, columns=[col])
            ax.plot(x_range, model.predict(x_range_df), color="orange", linewidth=2, label="Linear fit")

        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_xlabel("Training dose (hours / year)")
        ax.set_ylabel(f"Δ {title}")
        ax.set_title(r["label"])
        ax.annotate(
            f"R² = {r['r2']:.3f}\nn = {r['n']}",
            xy=(0.97, 0.97), xycoords="axes fraction",
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )

    for j in range(len(panels), 4):
        axes[j].set_visible(False)

    plt.suptitle(f"Dose-Response by Training Component — {title} (devices excluded)", fontsize=13)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / filename, dpi=150)
    print(f"Saved: {filename}")


# -------------------------------------------------------
# Figure 2 — Treatment box plots
# -------------------------------------------------------


def plot_treatment_effects(
    results: dict,
    title: str = "Motor Score",
    filename: str = "treatment_effects.png",
):
    active_df      = results["active_df"]
    treatment_cols = results["treatment_cols"]

    if not treatment_cols:
        print("No treatment columns found — skipping treatment plot.")
        return

    n_cols = len(treatment_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), sharey=True)
    if n_cols == 1:
        axes = [axes]

    for ax, col in zip(axes, treatment_cols):
        received     = active_df[active_df[col] == 1]["delta_score"].dropna()
        not_received = active_df[active_df[col] == 0]["delta_score"].dropna()

        bp = ax.boxplot([not_received, received], patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor("#AED6F1")
        bp["boxes"][1].set_facecolor("#A9DFBF")

        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"No\n(n={len(not_received)})", f"Yes\n(n={len(received)})"])
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("Received treatment")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")

        diff = received.mean() - not_received.mean()
        ax.annotate(
            f"Δ mean = {diff:+.2f}",
            xy=(0.5, 0.97), xycoords="axes fraction",
            ha="center", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )

    axes[0].set_ylabel(f"Δ {title}")
    plt.suptitle(f"Score Change by Medical Treatment — {title}", fontsize=13)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / filename, dpi=150)
    print(f"Saved: {filename}")


# -------------------------------------------------------
# Figure 3 — Overall dose-response (linear + poly overlay)
# -------------------------------------------------------


def plot_overall_dose_response(
    results: dict,
    title: str = "Motor Score",
    filename: str = "overall_dose_response.png",
):
    dose_df      = results["dose_df"]
    feature      = "total_training_hours_hr"
    linear_model = results["linear_model"]
    poly_model   = results["poly_model"]

    subset = dose_df[["delta_score", feature]].dropna()
    X, y   = subset[[feature]], subset["delta_score"]

    x_range    = np.linspace(X[feature].min(), X[feature].max(), 200).reshape(-1, 1)
    x_range_df = pd.DataFrame(x_range, columns=[feature])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X[feature], y, alpha=0.5, color="steelblue", s=30, label="Observed")
    ax.plot(
        x_range, linear_model.predict(x_range_df),
        color="orange", linewidth=2,
        label=f"Linear (R²={results['linear_r2']:.3f})",
    )
    ax.plot(
        x_range, poly_model.predict(x_range_df),
        color="red", linewidth=2, linestyle="--",
        label=f"Polynomial deg-2 (R²={results['poly_r2']:.3f})",
    )

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Total training hours / year")
    ax.set_ylabel(f"Δ {title}")
    ax.set_title(f"Overall Dose-Response: Training Hours vs {title} Change")
    ax.legend()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / filename, dpi=150)
    print(f"Saved: {filename}")


# -------------------------------------------------------
# Main — kör tre separata analyser
# -------------------------------------------------------

if __name__ == "__main__":
    from dataloader import load_data
    from connect_db import get_connection
    from preprocessing.motor_scores import (
        motorscore_milestones_setvalue,
        motorscore_impairments_setvalue,
        motorscore_combined,
    )
    from preprocessing.preprocessing_ht import process_training_per_type_per_year
    from preprocessing.preprocessing_it import (
        process_neurohab_hours_per_user_per_age,
        process_medical_treatments_per_user_per_age,
    )

    conn = get_connection()
    data = load_data(conn)

    # --- Bygg scores ---
    milestone_df  = motorscore_milestones_setvalue(data["motorical_development"])
    impairment_df = motorscore_impairments_setvalue(data["motorical_development"])
    combined_df   = motorscore_combined(milestone_df, impairment_df)

    # --- Bygg träningsdata ---
    home_df     = process_training_per_type_per_year(data["home_training"])
    neurohab_df = process_neurohab_hours_per_user_per_age(data["intensive_therapies"])
    medical_df  = process_medical_treatments_per_user_per_age(data["intensive_therapies"])

    # --- Definiera de tre analyserna ---
    analyses = [
        {
            "motor_df":  milestone_df,
            "score_col": "milestone_score",
            "title":     "Milestone Score",
            "key":       "milestones",
        },
        {
            "motor_df":  impairment_df,
            "score_col": "mms_normalized",
            "title":     "Impairment Score",
            "key":       "impairments",
        },
        {
            "motor_df":  combined_df,
            "score_col": "combined_score",
            "title":     "Combined Score",
            "key":       "combined",
        },
    ]

    # --- Kör och plotta varje analys ---
    for a in analyses:
        results = run_analysis(
            motor_df    = a["motor_df"],
            home_df     = home_df,
            neurohab_df = neurohab_df,
            medical_df  = medical_df,
            score_col   = a["score_col"],
        )

        print_summary(results, title=a["title"])

        plot_training_components(
            results,
            title    = a["title"],
            filename = f"training_components_{a['key']}.png",
        )
        plot_overall_dose_response(
            results,
            title    = a["title"],
            filename = f"overall_dose_response_{a['key']}.png",
        )
        plot_treatment_effects(
            results,
            title    = a["title"],
            filename = f"treatment_effects_{a['key']}.png",
        )

    plt.show()