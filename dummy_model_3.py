import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

from preprocessing_it import process_neurohab_hours_per_user_per_age, process_medical_treatments_per_user_per_age

# -------------------------------------------------------
# Build dose-response dataset
# -------------------------------------------------------

def build_dose_response_dataset(
    motor_df: pl.DataFrame,
    home_df: pl.DataFrame,
) -> pd.DataFrame:
    """
    Joins delta motor score with total training hours per child per year.
    Returns a pandas DataFrame ready for regression.
    """

    motor_df = motor_df.sort(["introductory_id", "age"])
    motor_df = motor_df.with_columns(
        (
            pl.col("motorical_score_2")
            - pl.col("motorical_score_2").shift(1)
        ) 
        .over("introductory_id") 
        .alias("delta_motor_score") 
    ).filter(pl.col("delta_motor_score").is_not_null()) 

    total_hours = (
        home_df
        .group_by(["introductory_id", "age"])
        .agg(pl.sum("total_hours").alias("total_training_hours"))
    ) 

    category_hours = (
        home_df
        .group_by(["introductory_id", "age", "training_category"])
        .agg(pl.sum("total_hours").alias("hours")) 
        .pivot(values="hours", index=["introductory_id", "age"], on="training_category") 
        .fill_null(0)
    )

    df = (
        motor_df
        .join(total_hours, on=["introductory_id", "age"], how="left")
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
) -> pd.DataFrame:
    """
    Combines home training + sports/other training + intensive therapy center hours,
    explicitly excluding devices.
    """

    motor_df = motor_df.sort(["introductory_id", "age"])
    motor_df = motor_df.with_columns(
        (
            pl.col("motorical_score_2")
            - pl.col("motorical_score_2").shift(1)
        )
        .over("introductory_id")
        .alias("delta_motor_score")
    ).filter(pl.col("delta_motor_score").is_not_null()) 

    home_hours = (
        home_df
        .filter(pl.col("training_category") == "home")
        .group_by(["introductory_id", "age"])
        .agg(pl.sum("total_hours").alias("home_hours"))
    ) 

    sports_hours = (
        home_df
        .filter(pl.col("training_category") == "other")
        .group_by(["introductory_id", "age"])
        .agg(pl.sum("total_hours").alias("sports_hours"))
    ) 

    neurohab_hours = (
        neurohab_df
        .group_by(["introductory_id", "age"])
        .agg(pl.sum("total_hours").alias("neurohab_hours"))
    )

    df = (
        motor_df
        .join(home_hours, on=["introductory_id", "age"], how="left")
        .join(sports_hours, on=["introductory_id", "age"], how="left")
        .join(neurohab_hours, on=["introductory_id", "age"], how="left")
        .join(medical_df, on=["introductory_id", "age"], how="left")
        .fill_null(0)
    )

    df = df.with_columns(
        (
            pl.col("home_hours")
            + pl.col("sports_hours")
            + pl.col("neurohab_hours")
        ).alias("active_total")
    ) 

    df = df.with_columns([
        (pl.col("home_hours") / 60).alias("home_hours"),
        (pl.col("sports_hours") / 60).alias("sports_hours"),
        (pl.col("neurohab_hours") / 60).alias("neurohab_hours"),
        (pl.col("active_total") / 60).alias("active_total"),
    ])


    return df.to_pandas()

# -------------------------------------------------------
# Regression helpers
# -------------------------------------------------------

def _fit_linear(X: pd.DataFrame, y: pd.Series):
    model = LinearRegression()
    model.fit(X, y)
    r2 = r2_score(y, model.predict(X))
    return model, r2


def fit_linear_dose_response(df: pd.DataFrame, feature: str = "total_training_hours_hr"):
    subset = df[["delta_motor_score", feature]].dropna()
    X, y = subset[[feature]], subset["delta_motor_score"]
    return _fit_linear(X, y)


def fit_polynomial_dose_response(df: pd.DataFrame, feature: str = "total_training_hours_hr", degree: int = 2):
    subset = df[["delta_motor_score", feature]].dropna()
    X, y = subset[[feature]], subset["delta_motor_score"]
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression()
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
    Runs all regressions and collects results.
    Returns a dict consumed by print_summary() and the plot functions.
    """
    active_df = build_active_hours_dataset(motor_df, home_df, neurohab_df, medical_df)
    dose_df   = build_dose_response_dataset(motor_df, home_df)

    known_cols = {
        "introductory_id", "age", "motorical_score_2",
        "delta_motor_score", "home_hours", "sports_hours",
        "neurohab_hours", "active_total", "cum_unique_milestones",
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
        subset = active_df[["delta_motor_score", col]].dropna()
        subset = subset[subset[col] >= 0]
        if len(subset) < 5 or subset[col].sum() == 0:
            continue
        _, r2 = _fit_linear(subset[[col]], subset["delta_motor_score"])
        model, _ = _fit_linear(subset[[col]], subset["delta_motor_score"])
        component_results.append({
            "col":        col,
            "label":      label,
            "coeff":      model.coef_[0],
            "intercept":  model.intercept_,
            "r2":         r2,
            "n":          len(subset),
            "mean_hours": round(subset[col].mean(), 1),
        })

    treatment_results = []
    for col in treatment_cols:
        received     = active_df[active_df[col] == 1]["delta_motor_score"].dropna()
        not_received = active_df[active_df[col] == 0]["delta_motor_score"].dropna()
        if len(received) == 0:
            continue
        treatment_results.append({
            "col":          col,
            "n_received":   len(received),
            "n_not":        len(not_received),
            "mean_yes":     round(received.mean(), 3),
            "mean_no":      round(not_received.mean(), 3),
            "mean_diff":    round(received.mean() - not_received.mean(), 3),
        })

    linear_model, linear_r2   = fit_linear_dose_response(dose_df)
    poly_model,   poly_r2     = fit_polynomial_dose_response(dose_df)

    return {
        "active_df":          active_df,
        "dose_df":            dose_df,
        "component_results":  component_results,
        "treatment_results":  treatment_results,
        "treatment_cols":     treatment_cols,
        "linear_model":       linear_model,
        "linear_r2":          linear_r2,
        "poly_model":         poly_model,
        "poly_r2":            poly_r2,
    }
    
def print_summary(results: dict):
    sep  = "=" * 62
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
                f"  {t['col']:<28} "
                f"{t['n_received']:>8} "
                f"{t['n_not']:>8} "
                f"{diff_str:>8}"
            )

    print(f"\n{sep}\n")

# -------------------------------------------------------
# Figure 1 — 2×2 training component scatter plots
# -------------------------------------------------------

def plot_training_components(results: dict):
    """
    2×2 grid: home / sports / neurohab / combined total.
    Each panel has a scatter + regression line + R² annotation.
    """
    active_df = results["active_df"]
    panels = [r for r in results["component_results"]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, r in enumerate(panels):
        ax  = axes[i]
        col = r["col"]

        subset = active_df[["delta_motor_score", col]].dropna()
        subset = subset[subset[col] >= 0]

        ax.scatter(subset[col], subset["delta_motor_score"],
                   alpha=0.4, color="steelblue", s=30)

        if len(subset) >= 5 and subset[col].sum() > 0:
            model, _ = _fit_linear(subset[[col]], subset["delta_motor_score"])
            x_range    = np.linspace(subset[col].min(), subset[col].max(), 100)
            x_range_df = pd.DataFrame(x_range, columns=[col])
            ax.plot(x_range, model.predict(x_range_df),
                    color="orange", linewidth=2, label="Linear fit")

        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_xlabel("Training dose (hours / year)")
        ax.set_ylabel("Δ Motor score")
        ax.set_title(r["label"])

        ax.annotate(
            f"R² = {r['r2']:.3f}\nn = {r['n']}",
            xy=(0.97, 0.97), xycoords="axes fraction",
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )

    for j in range(len(panels), 4):
        axes[j].set_visible(False)

    plt.suptitle("Dose-Response by Training Component (devices excluded)", fontsize=13)
    plt.tight_layout()
    plt.savefig("training_components.png", dpi=150)


# -------------------------------------------------------
# Figure 2 — Treatment box plots
# -------------------------------------------------------

def plot_treatment_effects(results: dict):
    """
    One box plot per medical treatment, with sample sizes on the x-axis.
    """
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
        received     = active_df[active_df[col] == 1]["delta_motor_score"].dropna()
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
            xy=(0.5, 0.97), xycoords="axes fraction",
            ha="center", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )

    axes[0].set_ylabel("Δ Motor score")
    plt.suptitle("Motor Score Change by Medical Treatment", fontsize=13)
    plt.tight_layout()
    plt.savefig("treatment_effects.png", dpi=150)


# -------------------------------------------------------
# Figure 3 — Overall dose-response (linear + poly overlay)
# -------------------------------------------------------

def plot_overall_dose_response(results: dict):
    """
    Scatter + linear + polynomial fit for total home training hours.
    """
    dose_df      = results["dose_df"]
    feature      = "total_training_hours_hr"
    linear_model = results["linear_model"]
    poly_model   = results["poly_model"]

    subset = dose_df[["delta_motor_score", feature]].dropna()
    X, y   = subset[[feature]], subset["delta_motor_score"]

    x_range    = np.linspace(X[feature].min(), X[feature].max(), 200).reshape(-1, 1)
    x_range_df = pd.DataFrame(x_range, columns=[feature])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X[feature], y, alpha=0.5, color="steelblue", s=30, label="Observed")
    ax.plot(x_range, linear_model.predict(x_range_df),
            color="orange", linewidth=2, label=f"Linear (R²={results['linear_r2']:.3f})")
    ax.plot(x_range, poly_model.predict(x_range_df),
            color="red", linewidth=2, linestyle="--",
            label=f"Polynomial deg-2 (R²={results['poly_r2']:.3f})")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Total training hours / year")
    ax.set_ylabel("Δ Motor score")
    ax.set_title("Overall Dose-Response: Training Hours vs Motor Score Change")
    ax.legend()
    plt.tight_layout()
    plt.savefig("overall_dose_response.png", dpi=150)

# -------------------------------------------------------
# Main
# -------------------------------------------------------

if __name__ == "__main__":
    from dataloader import load_data
    from connect_db import get_connection
    from preprocessing_md import process_motorical_score_2_per_user_per_age, calculate_percentile_motor_score_3
    from preprocessing_ht import process_training_per_type_per_year
    from preprocessing_it import process_neurohab_hours_per_user_per_age, process_medical_treatments_per_user_per_age

    conn = get_connection()
    data = load_data(conn)

    motor_df = process_motorical_score_2_per_user_per_age(data["motorical_development"])
    home_df = process_training_per_type_per_year(data["home_training"])
    neurohab_df = process_neurohab_hours_per_user_per_age(data["intensive_therapies"])
    medical_df = process_medical_treatments_per_user_per_age(data["intensive_therapies"])


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