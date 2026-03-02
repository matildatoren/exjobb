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

    # Add delta motor score
    motor_df = motor_df.sort(["introductory_id", "age"]) # Sorts motorscore by child and age
    motor_df = motor_df.with_columns(
        (
            pl.col("motorical_score_2")
            - pl.col("motorical_score_2").shift(1)
        ) # Calculates difference between two following years
        .over("introductory_id") # For each introductory id
        .alias("delta_motor_score") # Names the calculated variable
    ).filter(pl.col("delta_motor_score").is_not_null()) # Removes the first year for each child. 

    # Total hours across all training types per child per year
    total_hours = (
        home_df
        .group_by(["introductory_id", "age"])
        .agg(pl.sum("total_hours").alias("total_training_hours"))
    ) #Sums all minutes per age and year and names it

    # Also keep hours per category for breakdown analysis
    category_hours = (
        home_df
        .group_by(["introductory_id", "age", "training_category"])
        .agg(pl.sum("total_hours").alias("hours")) #Sums hours for each category
        .pivot(values="hours", index=["introductory_id", "age"], on="training_category") #Gives each category its own column
        .fill_null(0)
    )

    df = (
        motor_df
        .join(total_hours, on=["introductory_id", "age"], how="left")
        .join(category_hours, on=["introductory_id", "age"], how="left")
        .fill_null(0)
    ) #Joins total hours and category hours to motor df table and puts 0 if there is no training

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

    Input:
        motor_df     — output from process_motorical_score_2_per_user_per_age
        home_df      — output from process_training_per_type_per_year
        neurohab_df  — output from process_neurohab_hours_per_user_per_age
        

    Output:
        pandas DataFrame with one row per child per year, containing:
            - delta_motor_score
            - home_hours       (category == "home")
            - sports_hours     (category == "other")
            - neurohab_hours   (intensive therapy centers)
            - active_total     (sum of all three)
    """

    # Delta motor score
    motor_df = motor_df.sort(["introductory_id", "age"])
    motor_df = motor_df.with_columns(
        (
            pl.col("motorical_score_2")
            - pl.col("motorical_score_2").shift(1)
        )
        .over("introductory_id")
        .alias("delta_motor_score")
    ).filter(pl.col("delta_motor_score").is_not_null()) # See comment in dose response dataset

    # Home training hours only (exclude devices)
    home_hours = (
        home_df
        .filter(pl.col("training_category") == "home")
        .group_by(["introductory_id", "age"])
        .agg(pl.sum("total_hours").alias("home_hours"))
    ) # Filters home hours and sums and name them.

    # Sports / other training hours
    sports_hours = (
        home_df
        .filter(pl.col("training_category") == "other")
        .group_by(["introductory_id", "age"])
        .agg(pl.sum("total_hours").alias("sports_hours"))
    ) # Filters other hours and sums and name them.

    # Neurohabilitation center hours (sum across all centers)
    neurohab_hours = (
        neurohab_df
        .group_by(["introductory_id", "age"])
        .agg(pl.sum("total_hours").alias("neurohab_hours"))
    )# Filters neurohab hours and sums and name them.

    # Join everything onto motor scores
    df = (
        motor_df
        .join(home_hours, on=["introductory_id", "age"], how="left")
        .join(sports_hours, on=["introductory_id", "age"], how="left")
        .join(neurohab_hours, on=["introductory_id", "age"], how="left")
        .join(medical_df, on=["introductory_id", "age"], how="left")
        .fill_null(0)
    ) # Joins all tables and put 0 if there are no training in that category

    # Combined active total
    df = df.with_columns(
        (
            pl.col("home_hours")
            + pl.col("sports_hours")
            + pl.col("neurohab_hours")
        ).alias("active_total")
    ) # sums all hours for each year and child to make total active hours

    df = df.with_columns([
        (pl.col("home_hours") / 60).alias("home_hr"),
        (pl.col("sports_hours") / 60).alias("sports_hr"),
        (pl.col("neurohab_hours") / 60).alias("neurohab_hr"),
        (pl.col("active_total") / 60).alias("active_total_hr"),
    ])


    return df.to_pandas()


# -------------------------------------------------------
# Analyze active hours dose-response
# -------------------------------------------------------

def analyze_active_hours(df: pd.DataFrame):
    """
    Runs linear regression for each active hour component and the combined total.
    Prints a comparison table and plots all four against delta_motor_score.

    Input:  DataFrame from build_active_hours_dataset
    Output: DataFrame with coefficients and R² per component
    """

    base_features = {
        "home_hr": "Home training (per hour)",
        "sports_hr": "Sports / other (per hour)",
        "neurohab_hr": "Intensive therapy (per hour)",
        "active_total_hr": "Combined active total (per hour)",
    }
    #Makes readable namnes for the 4 different features

    # Detect treatment columns automatically — anything not a known column
    known_cols = [
        "introductory_id", "age", "motorical_score_2",
        "delta_motor_score",
        "home_hr", "sports_hr",
        "neurohab_hr", "active_total_hr"
    ]

    treatment_features = {
        col: f"Medical: {col}"
        for col in df.columns
        if col not in known_cols
    }

    # Use all features for the regression table
    all_features = {**base_features, **treatment_features}

    results = []

    for col, label in all_features.items(): #Iterates over each feature
        subset = df[["delta_motor_score", col]].dropna() #Selects only the current iterations column and the motorscore
        subset = subset[subset[col] >= 0] # Removes rows with negative training hours (error in data)

        if len(subset) < 5 or subset[col].sum() == 0:
            print(f"  Skipping {label} — insufficient data")
            continue #Safety check, skipps regresion if there are fewer than 5 data points

        X = subset[[col]] # Creates the feature matrix, double brackets to keep as a df and not Series
        y = subset["delta_motor_score"] # creates the target variable motor score

        model = LinearRegression()
        model.fit(X, y)
        r2 = r2_score(y, model.predict(X)) #Predicts and compares predictions to actual values and gives score from 0-1

        results.append({
            "component": label,
            "coefficient": model.coef_[0],
            "intercept": model.intercept_,
            "r2": r2,
            "n": len(y),
            "mean_hours": subset[col].mean().round(1), #added to see how many hours participants averaged
        }) # Adds a dict of results to the results list

    results_df = pd.DataFrame(results).sort_values("coefficient", ascending=False) # Converts dict list into df and sorts by coefficient

    #Printing results for all four components
    print("\nActive hours dose-response (devices excluded):")
    print(results_df.to_string(index=False))

    # Plot all four components as scatterplits
    fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # Creates a figure with 2x2 grid of subplots
    axes = axes.flatten()

    for i, (col, label) in enumerate(base_features.items()): # Loops over features again
        if col not in df.columns:
            continue #Safety checks, skips plotting if the column does not exist in df

        subset = df[["delta_motor_score", col]].dropna() # only 2 relevant columns and no null
        ax = axes[i] # Selects the correct subplot for this feature iteration

        ax.scatter(subset[col], subset["delta_motor_score"], alpha=0.4, color="steelblue") # Draws actual datapoints

        if len(subset) >= 5 and subset[col].sum() > 0: # Only draws the regression line if there is enough data
            model = LinearRegression()
            model.fit(subset[[col]], subset["delta_motor_score"])
            x_range = np.linspace(subset[col].min(), subset[col].max(), 100) # Creates 100 evenly spread values to fit the coeff and such fitted by the model
            x_range_df = pd.DataFrame(x_range, columns=[col]) # Puts them in dataframe as that is what is expected from the plot
            ax.plot(x_range, model.predict(x_range_df), color="orange", linewidth=2) #Plots a smooth regression line

        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_xlabel("Training dose ((hours per year))")
        ax.set_ylabel("Delta motor score")
        ax.set_title(label)

    #Plots the results
    plt.suptitle("Dose-Response by Training Component (devices excluded)", fontsize=13)
    plt.tight_layout()
    plt.savefig("active_hours_dose_response.png", dpi=150)
    plt.show()
    print("\nPlot saved as active_hours_dose_response.png")

    return results_df

    
def plot_treatment_effects(df: pd.DataFrame):
    known_cols = [
        "introductory_id", "age", "motorical_score_2",
        "delta_motor_score", "home_hours", "sports_hours",
        "neurohab_hours", "active_total", "home_hr", "sports_hr",
        "neurohab_hr", "active_total_hr"
    ]

    # known_cols = [
    #     "introductory_id", "age", "motorical_score_2",
    #     "delta_motor_score",
    #     "home_hr", "sports_hr",
    #     "neurohab_hr", "active_total_hr"
    # ]

    treatment_cols = [c for c in df.columns if c not in known_cols]

    if not treatment_cols:
        print("No treatment columns found")
        return

    fig, axes = plt.subplots(1, len(treatment_cols), figsize=(5 * len(treatment_cols), 5))
    if len(treatment_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, treatment_cols):
        received = df[df[col] == 1]["delta_motor_score"].dropna()
        not_received = df[df[col] == 0]["delta_motor_score"].dropna()

        ax.boxplot([not_received, received], tick_labels=["No", "Yes"])
        ax.set_title(col)
        ax.set_xlabel("Received treatment")
        ax.set_ylabel("Delta motor score")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")

    plt.suptitle("Motor Score Change by Medical Treatment", fontsize=13)
    plt.tight_layout()
    plt.savefig("treatment_effects.png", dpi=150)
    plt.show()
    print("\nPlot saved as treatment_effects.png")


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
    plt.xlabel("Total training per year")
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

    # --- Plotting dose response from hometraining, including devices ---

    df = build_dose_response_dataset(motor_df, home_df)

    linear_model = fit_linear_dose_response(df, feature="total_training_hours_hr")
    poly_model = fit_polynomial_dose_response(df, feature="total_training_hours_hr", degree=2)

    fit_per_category(df)

    plot_dose_response(
        df,
        linear_model,
        poly_model,
        feature="total_training_hours_hr"
    )

    # --- Plotting (active hours = home training + other sports + therapies at centers), devices excluded ---
    active_df = build_active_hours_dataset(motor_df, home_df, neurohab_df, medical_df)

    analyze_active_hours(active_df)

    # --- boxplots for medical treatments ----

    plot_treatment_effects(active_df)