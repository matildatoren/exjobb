import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score


# -------------------------------------------------------
# Build dose-response dataset
# -------------------------------------------------------

def build_dose_response_dataset(
    motor_df: pl.DataFrame,
    home_df: pl.DataFrame,
    possible_milestones_by_age: dict[int, int],
) -> pd.DataFrame:
    """
    Joins delta motor score with total training hours per child per year.
    Returns a pandas DataFrame ready for regression.
    """

    # Add delta motor score
    motor_df = motor_df.sort(["introductory_id", "age"])
    motor_df = motor_df.with_columns(
        (
            pl.col("motorical_score_2")
            - pl.col("motorical_score_2").shift(1)
        )
        .over("introductory_id")
        .alias("delta_motor_score")
    ).filter(pl.col("delta_motor_score").is_not_null())

    # Total hours across all training types per child per year
    total_hours = (
        home_df
        .group_by(["introductory_id", "age"])
        .agg(pl.sum("total_hours").alias("total_training_hours"))
    )

    # Also keep hours per category for breakdown analysis
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

    return df.to_pandas()


# -------------------------------------------------------
# Linear dose-response
# -------------------------------------------------------

def fit_linear_dose_response(df: pd.DataFrame, feature: str = "total_training_hours"):
    """
    Fits a simple linear regression: delta_motor_score ~ training_hours.
    """
    subset = df[["delta_motor_score", feature]].dropna()
    X = subset[[feature]]
    y = subset["delta_motor_score"]

    model = LinearRegression()
    model.fit(X, y)

    preds = model.predict(X)
    r2 = r2_score(y, preds)

    print(f"\nLinear dose-response: delta_motor_score ~ {feature}")
    print(f"  Coefficient : {model.coef_[0]:.6f}")
    print(f"  Intercept   : {model.intercept_:.6f}")
    print(f"  R²          : {r2:.4f}")
    print(f"  n           : {len(y)}")

    return model


# -------------------------------------------------------
# Polynomial dose-response (detects plateau / threshold)
# -------------------------------------------------------

def fit_polynomial_dose_response(df: pd.DataFrame, feature: str = "total_training_hours", degree: int = 2):
    """
    Fits a polynomial regression to detect threshold or plateau effects.
    degree=2 detects a single turning point (e.g. diminishing returns).
    """
    subset = df[["delta_motor_score", feature]].dropna()
    X = subset[[feature]]
    y = subset["delta_motor_score"]

    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression()
    )
    model.fit(X, y)

    preds = model.predict(X)
    r2 = r2_score(y, preds)

    print(f"\nPolynomial (degree={degree}) dose-response: delta_motor_score ~ {feature}")
    print(f"  R²: {r2:.4f}")
    print(f"  n : {len(y)}")

    return model


# -------------------------------------------------------
# Per-therapy dose-response
# -------------------------------------------------------

def fit_per_category(df: pd.DataFrame):
    """
    Runs a linear regression for each training category separately.
    Useful for comparing which therapy type has the strongest dose-response.
    """
    categories = [c for c in ["home", "devices", "other"] if c in df.columns]

    results = []

    for cat in categories:
        subset = df[["delta_motor_score", cat]].dropna()
        if subset[cat].sum() == 0 or len(subset) < 5:
            continue

        X = subset[[cat]]
        y = subset["delta_motor_score"]

        model = LinearRegression()
        model.fit(X, y)
        r2 = r2_score(y, model.predict(X))

        results.append({
            "category": cat,
            "coefficient": model.coef_[0],
            "intercept": model.intercept_,
            "r2": r2,
            "n": len(y)
        })

    results_df = pd.DataFrame(results).sort_values("coefficient", ascending=False)

    print("\nPer-category dose-response:")
    print(results_df.to_string(index=False))

    return results_df


# -------------------------------------------------------
# Plot
# -------------------------------------------------------

def plot_dose_response(df: pd.DataFrame, linear_model, poly_model, feature: str = "total_training_hours"):
    """
    Scatter plot with linear and polynomial fit overlaid.
    """
    subset = df[["delta_motor_score", feature]].dropna()
    X = subset[[feature]]
    y = subset["delta_motor_score"]

    x_range = np.linspace(X[feature].min(), X[feature].max(), 200).reshape(-1, 1)
    x_range_df = pd.DataFrame(x_range, columns=[feature])

    linear_preds = linear_model.predict(x_range_df)
    poly_preds = poly_model.predict(x_range_df)

    plt.figure(figsize=(10, 6))
    plt.scatter(X[feature], y, alpha=0.6, label="Observed", color="steelblue")
    plt.plot(x_range, linear_preds, label="Linear fit", color="orange", linewidth=2)
    plt.plot(x_range, poly_preds, label="Polynomial fit (deg 2)", color="red", linewidth=2, linestyle="--")
    plt.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    plt.xlabel("Total training hours per year")
    plt.ylabel("Delta motor score")
    plt.title("Dose-Response: Training Hours vs Motor Score Change")
    plt.legend()
    plt.tight_layout()
    plt.savefig("dose_response_plot.png", dpi=150)
    plt.show()
    print("\nPlot saved as dose_response_plot.png")


# -------------------------------------------------------
# Main
# -------------------------------------------------------

if __name__ == "__main__":
    from dataloader import load_data
    from connect_db import get_connection
    from preprocessing_md import process_motorical_score_2_per_user_per_age
    from preprocessing_ht import process_training_per_type_per_year

    conn = get_connection()
    data = load_data(conn)

    possible_milestones_by_age = {
        1: 12,
        2: 19,
        3: 25,
        4: 31,
        5: 36,
        6: 39,
        7: 41
    }

    motor_df = process_motorical_score_2_per_user_per_age(
        data["motorical_development"], possible_milestones_by_age
    )
    home_df = process_training_per_type_per_year(data["home_training"])

    # Optional: filter to completed surveys only
    # completed_ids = [1, 4, 7, 12]
    # motor_df = motor_df.filter(pl.col("introductory_id").is_in(completed_ids))
    # home_df = home_df.filter(pl.col("introductory_id").is_in(completed_ids))

    df = build_dose_response_dataset(motor_df, home_df, possible_milestones_by_age)

    print(f"\nDataset shape: {df.shape}")
    print(df[["introductory_id", "age", "total_training_hours", "delta_motor_score"]].head(10))

    linear_model = fit_linear_dose_response(df)
    poly_model = fit_polynomial_dose_response(df, degree=2)

    fit_per_category(df)

    plot_dose_response(df, linear_model, poly_model)