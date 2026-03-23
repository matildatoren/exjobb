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
) -> pd.DataFrame:
    """
    Join delta motor score with total training hours per child per year.

    Calculates the change in motor score between consecutive age intervals
    and merges it with aggregated training hours — both as a total and broken
    down by training category (wide format).

    Args:
        motor_df (pl.DataFrame): Motor development data with motor scores per
            child and age.
        home_df (pl.DataFrame): Home training data with therapy hours per child,
            age, and training category.

    Returns:
        pd.DataFrame: Pandas DataFrame with one row per (child, age) containing
            delta_motor_score, total_training_hours, total_training_hours_hr
            (converted from minutes to hours), and one column per training category.
    """

    motor_df = motor_df.sort(["introductory_id", "age"])
    motor_df = motor_df.with_columns(
        (pl.col("milestone_score") - pl.col("milestone_score").shift(1))
        .over("introductory_id")
        .alias("delta_motor_score")
    ).filter(pl.col("delta_motor_score").is_not_null())

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

    # Convert total training hours from minutes to hours for interpretability
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
) -> pd.DataFrame:
    """
    Combine home training, sports/other training, and intensive therapy hours.

    Devices are explicitly excluded by filtering on training_category.
    The result includes a summed active_total column and binary indicators
    for medical treatments.

    Args:
        motor_df (pl.DataFrame): Motor development data with motor scores per
            child and age.
        home_df (pl.DataFrame): Home training data with therapy hours per child,
            age, and training category.
        neurohab_df (pl.DataFrame): Intensive therapy hours per child and age.
        medical_df (pl.DataFrame): Binary medical treatment indicators per child
            and age.

    Returns:
        pd.DataFrame: Pandas DataFrame with one row per (child, age) containing
            delta_motor_score, component hours (home, sports, neurohab),
            active_total (sum of all active components), and medical treatment
            columns. All hour columns are converted from minutes to hours.
    """

    motor_df = motor_df.sort(["introductory_id", "age"])
    motor_df = motor_df.with_columns(
        (pl.col("milestone_score") - pl.col("milestone_score").shift(1))
        .over("introductory_id")
        .alias("delta_motor_score")
    ).filter(pl.col("delta_motor_score").is_not_null())

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

    # Convert all hour columns from minutes to hours
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
    """
    Fit a simple linear regression model and return the model with its R² score.

    Args:
        X (pd.DataFrame): Feature matrix with one or more predictor columns.
        y (pd.Series): Target variable (delta motor score).

    Returns:
        tuple[LinearRegression, float]: Fitted model and its in-sample R² score.
    """
    model = LinearRegression()
    model.fit(X, y)
    r2 = r2_score(y, model.predict(X))
    return model, r2


def fit_linear_dose_response(
    df: pd.DataFrame, feature: str = "total_training_hours_hr"
):
    """
    Fit a linear dose-response model for a given training feature.

    Rows with missing values in either the feature or the target are dropped
    before fitting.

    Args:
        df (pd.DataFrame): Dose-response dataset containing delta_motor_score
            and the specified feature column.
        feature (str): Name of the predictor column. Defaults to
            "total_training_hours_hr".

    Returns:
        tuple[LinearRegression, float]: Fitted linear model and its R² score.
    """
    subset = df[["delta_motor_score", feature]].dropna()
    X, y = subset[[feature]], subset["delta_motor_score"]
    return _fit_linear(X, y)


def fit_polynomial_dose_response(
    df: pd.DataFrame, feature: str = "total_training_hours_hr", degree: int = 2
):
    """
    Fit a polynomial dose-response model for a given training feature.

    Uses a scikit-learn Pipeline combining PolynomialFeatures and
    LinearRegression. Rows with missing values are dropped before fitting.

    Args:
        df (pd.DataFrame): Dose-response dataset containing delta_motor_score
            and the specified feature column.
        feature (str): Name of the predictor column. Defaults to
            "total_training_hours_hr".
        degree (int): Degree of the polynomial expansion. Defaults to 2.

    Returns:
        tuple[Pipeline, float]: Fitted polynomial pipeline and its R² score.
    """
    subset = df[["delta_motor_score", feature]].dropna()
    X, y = subset[[feature]], subset["delta_motor_score"]
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False), LinearRegression()
    )
    model.fit(X, y)
    r2 = r2_score(y, model.predict(X))
    return model, r2


# -------------------------------------------------------
# Analyze active hours dose-response
# -------------------------------------------------------


def run_analysis(
    motor_df: pl.DataFrame,
    home_df: pl.DataFrame,
    neurohab_df: pl.DataFrame,
    medical_df: pl.DataFrame,
) -> dict:
    """
    Run all dose-response regressions and return the results as a dict.

    Builds both the active-hours dataset (devices excluded) and the full
    dose-response dataset, then fits linear regressions for each training
    component and computes mean differences for each medical treatment.
    Linear and polynomial overall models are also fitted.

    Args:
        motor_df (pl.DataFrame): Motor development data with motor scores per
            child and age.
        home_df (pl.DataFrame): Home training data with therapy hours per child,
            age, and training category.
        neurohab_df (pl.DataFrame): Intensive therapy hours per child and age.
        medical_df (pl.DataFrame): Binary medical treatment indicators per child
            and age.

    Returns:
        dict: Results dictionary consumed by print_summary() and the plot
            functions. Keys include:
            - "active_df": Active-hours pandas DataFrame.
            - "dose_df": Full dose-response pandas DataFrame.
            - "component_results": List of regression stats per training component.
            - "treatment_results": List of mean-difference stats per treatment.
            - "treatment_cols": List of medical treatment column names.
            - "linear_model": Fitted overall linear model.
            - "linear_r2": R² of the linear model.
            - "poly_model": Fitted overall polynomial model.
            - "poly_r2": R² of the polynomial model.
    """
    active_df = build_active_hours_dataset(motor_df, home_df, neurohab_df, medical_df)
    dose_df = build_dose_response_dataset(motor_df, home_df)

    # Identify medical treatment columns by excluding all known non-treatment columns
    known_cols = {
        "introductory_id",
        "age",
        "milestone_score",
        "delta_motor_score",
        "home_hours",
        "sports_hours",
        "neurohab_hours",
        "active_total",
        "cum_unique_milestones",
    }
    treatment_cols = [c for c in active_df.columns if c not in known_cols]

    components = {
        "home_hours": "Home training",
        "sports_hours": "Sports / other",
        "neurohab_hours": "Intensive therapy",
        "active_total": "Combined active total",
    }

    component_results = []
    for col, label in components.items():
        subset = active_df[["delta_motor_score", col]].dropna()
        subset = subset[subset[col] >= 0]
        if len(subset) < 5 or subset[col].sum() == 0:
            continue
        model, r2 = _fit_linear(subset[[col]], subset["delta_motor_score"])
        component_results.append(
            {
                "col": col,
                "label": label,
                "coeff": model.coef_[0],
                "intercept": model.intercept_,
                "r2": r2,
                "n": len(subset),
                "mean_hours": round(subset[col].mean(), 1),
            }
        )

    treatment_results = []
    for col in treatment_cols:
        received = active_df[active_df[col] == 1]["delta_motor_score"].dropna()
        not_received = active_df[active_df[col] == 0]["delta_motor_score"].dropna()
        if len(received) == 0:
            continue
        treatment_results.append(
            {
                "col": col,
                "n_received": len(received),
                "n_not": len(not_received),
                "mean_yes": round(received.mean(), 3),
                "mean_no": round(not_received.mean(), 3),
                "mean_diff": round(received.mean() - not_received.mean(), 3),
            }
        )

    linear_model, linear_r2 = fit_linear_dose_response(dose_df)
    poly_model, poly_r2 = fit_polynomial_dose_response(dose_df)

    return {
        "active_df": active_df,
        "dose_df": dose_df,
        "component_results": component_results,
        "treatment_results": treatment_results,
        "treatment_cols": treatment_cols,
        "linear_model": linear_model,
        "linear_r2": linear_r2,
        "poly_model": poly_model,
        "poly_r2": poly_r2,
    }


def print_summary(results: dict):
    """
    Print a formatted summary of all dose-response and treatment results.

    Outputs three sections to stdout:
    - A regression table for each active training component (coefficient, R², N, mean hours).
    - Overall linear vs. polynomial R² comparison for all home training.
    - A mean-difference table for each medical treatment.

    Args:
        results (dict): Results dictionary as returned by run_analysis().
    """

    sep = "=" * 62
    line = "-" * 62

    print(f"\n{sep}")
    print("  DOSE-RESPONSE ANALYSIS SUMMARY")
    print(sep)

    # --- Component regression table ---
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

    # --- Overall dose-response (linear vs poly) ---
    print(f"\n  {line}")
    print("  Overall Dose-Response (all home training incl. devices)\n")
    print(f"    Linear model   R² = {results['linear_r2']:.4f}")
    print(f"    Poly model     R² = {results['poly_r2']:.4f}")

    # --- Treatment effects table ---
    if results["treatment_results"]:
        print(f"\n  {line}")
        print("  Medical Treatments\n")
        header2 = f"  {'Treatment':<28} {'Yes (n)':>8} {'No (n)':>8} {'Diff':>8}"
        print(header2)
        print(f"  {line}")
        for t in results["treatment_results"]:
            diff_str = f"{t['mean_diff']:+.3f}"
            print(
                f"  {t['col']:<28} {t['n_received']:>8} {t['n_not']:>8} {diff_str:>8}"
            )

    print(f"\n{sep}\n")


# -------------------------------------------------------
# Figure 1 — 2×2 training component scatter plots
# -------------------------------------------------------


def plot_training_components(results: dict):
    """
    Plot a 2×2 grid of scatter plots with regression lines per training component.

    Each panel shows the relationship between training dose (hours/year) and
    delta motor score for one component (home, sports, neurohab, combined).
    Panels include a linear fit line and an R²/N annotation. Devices are excluded.

    Args:
        results (dict): Results dictionary as returned by run_analysis().

    Saves:
        training_components.png: Figure saved to the current working directory.
    """
    active_df = results["active_df"]
    panels = [r for r in results["component_results"]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, r in enumerate(panels):
        ax = axes[i]
        col = r["col"]

        subset = active_df[["delta_motor_score", col]].dropna()
        subset = subset[subset[col] >= 0]

        ax.scatter(
            subset[col], subset["delta_motor_score"], alpha=0.4, color="steelblue", s=30
        )

        if len(subset) >= 5 and subset[col].sum() > 0:
            model, _ = _fit_linear(subset[[col]], subset["delta_motor_score"])
            x_range = np.linspace(subset[col].min(), subset[col].max(), 100)
            x_range_df = pd.DataFrame(x_range, columns=[col])
            ax.plot(
                x_range,
                model.predict(x_range_df),
                color="orange",
                linewidth=2,
                label="Linear fit",
            )

        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_xlabel("Training dose (hours / year)")
        ax.set_ylabel("Δ Motor score")
        ax.set_title(r["label"])

        ax.annotate(
            f"R² = {r['r2']:.3f}\nn = {r['n']}",
            xy=(0.97, 0.97),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )

    for j in range(len(panels), 4):
        axes[j].set_visible(False)

    plt.suptitle("Dose-Response by Training Component (devices excluded)", fontsize=13)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "training_components.png", dpi=150)


# -------------------------------------------------------
# Figure 2 — Treatment box plots
# -------------------------------------------------------


def plot_treatment_effects(results: dict):
    """
    Plot side-by-side box plots comparing motor score change by medical treatment.

    One panel per treatment column shows the distribution of delta motor score
    for children who did and did not receive the treatment, along with sample
    sizes and the mean difference annotation.

    Args:
        results (dict): Results dictionary as returned by run_analysis().

    Saves:
        treatment_effects.png: Figure saved to the current working directory.
    """
    active_df = results["active_df"]
    treatment_cols = results["treatment_cols"]

    if not treatment_cols:
        print("No treatment columns found — skipping treatment plot.")
        return

    n_cols = len(treatment_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), sharey=True)
    if n_cols == 1:
        axes = [axes]

    for ax, col in zip(axes, treatment_cols):
        received = active_df[active_df[col] == 1]["delta_motor_score"].dropna()
        not_received = active_df[active_df[col] == 0]["delta_motor_score"].dropna()

        bp = ax.boxplot(
            [not_received, received],
            patch_artist=True,
            widths=0.5,
        )
        bp["boxes"][0].set_facecolor("#AED6F1")
        bp["boxes"][1].set_facecolor("#A9DFBF")

        ax.set_xticks([1, 2])
        ax.set_xticklabels(
            [f"No\n(n={len(not_received)})", f"Yes\n(n={len(received)})"]
        )
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("Received treatment")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")

        diff = received.mean() - not_received.mean()
        ax.annotate(
            f"Δ mean = {diff:+.2f}",
            xy=(0.5, 0.97),
            xycoords="axes fraction",
            ha="center",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )

    axes[0].set_ylabel("Δ Motor score")
    plt.suptitle("Motor Score Change by Medical Treatment", fontsize=13)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "treatment_effects.png", dpi=150)


# -------------------------------------------------------
# Figure 3 — Overall dose-response (linear + poly overlay)
# -------------------------------------------------------


def plot_overall_dose_response(results: dict):
    """
    Plot observed data with overlaid linear and polynomial dose-response fits.

    Shows how total home training hours (including devices) relate to change
    in motor score. Both the linear and degree-2 polynomial fits are displayed
    with their respective R² values in the legend.

    Args:
        results (dict): Results dictionary as returned by run_analysis().

    Saves:
        overall_dose_response.png: Figure saved to the current working directory.
    """
    dose_df = results["dose_df"]
    feature = "total_training_hours_hr"
    linear_model = results["linear_model"]
    poly_model = results["poly_model"]

    subset = dose_df[["delta_motor_score", feature]].dropna()
    X, y = subset[[feature]], subset["delta_motor_score"]

    x_range = np.linspace(X[feature].min(), X[feature].max(), 200).reshape(-1, 1)
    x_range_df = pd.DataFrame(x_range, columns=[feature])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X[feature], y, alpha=0.5, color="steelblue", s=30, label="Observed")
    ax.plot(
        x_range,
        linear_model.predict(x_range_df),
        color="orange",
        linewidth=2,
        label=f"Linear (R²={results['linear_r2']:.3f})",
    )
    ax.plot(
        x_range,
        poly_model.predict(x_range_df),
        color="red",
        linewidth=2,
        linestyle="--",
        label=f"Polynomial deg-2 (R²={results['poly_r2']:.3f})",
    )

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Total training hours / year")
    ax.set_ylabel("Δ Motor score")
    ax.set_title("Overall Dose-Response: Training Hours vs Motor Score Change")
    ax.legend()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "overall_dose_response.png", dpi=150)


# -------------------------------------------------------
# Main
# -------------------------------------------------------

if __name__ == "__main__":
    from dataloader import load_data
    from connect_db import get_connection
    from preprocessing.preprocessing_md import (
        process_motorical_score_2_per_user_per_age,
        calculate_percentile_motor_score_3,
        calculate_expected_milestone_score_3,
        process_motorical_score_1,
    )
    from preprocessing.motor_scores import (
        motorscore_milestones_setvalue,
        motorscore_milestones,
    )
    from preprocessing.preprocessing_ht import process_training_per_type_per_year
    from preprocessing.preprocessing_it import (
        process_neurohab_hours_per_user_per_age,
        process_medical_treatments_per_user_per_age,
    )

    conn = get_connection()
    data = load_data(conn)

    motor_df = motorscore_milestones(data["motorical_development"])
    home_df = process_training_per_type_per_year(data["home_training"])
    neurohab_df = process_neurohab_hours_per_user_per_age(data["intensive_therapies"])
    medical_df = process_medical_treatments_per_user_per_age(
        data["intensive_therapies"]
    )

    # Optional: filter to completed surveys only
    # completed_ids = ["f9231c8d-2ade-4c0e-a878-a9524ccc3d65", "9a3aeeeb-b409-4052-af0e-27e4893fb48f", "65ab3206-7371-4471-845c-6d238050494f"]
    # motor_df = motor_df.filter(pl.col("introductory_id").is_in(completed_ids))
    # home_df = home_df.filter(pl.col("introductory_id").is_in(completed_ids))

    # --- Run all regressions ---
    results = run_analysis(motor_df, home_df, neurohab_df, medical_df)

    # --- Print one clean summary block ---
    print_summary(results)

    # --- Build all figures (no show yet) ---
    plot_training_components(results)
    plot_overall_dose_response(results)
    plot_treatment_effects(results)

    # --- Show everything at once ---
    plt.show()
