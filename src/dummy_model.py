import time

import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split

# Only check ids from database with complete survey
INCLUDE_IDS = [
    "65ab3206-7371-4471-845c-6d238050494f",
    "9a3aeeeb-b409-4052-af0e-27e4893fb48f",
    "a1c29f34-c3a0-4140-8398-e3d8eb980292",
    "f9231c8d-2ade-4c0e-a878-a9524ccc3d65",
]

POSSIBLE_MILESTONES_BY_AGE = {
    1: 12,
    2: 19,
    3: 25,
    4: 31,
    5: 36,
    6: 39,
    7: 41,
}

BASE_DIR = Path(__file__).resolve().parent.parent  
IMAGES_DIR = BASE_DIR / "images"


def add_delta_motor_score(motor_df: pl.DataFrame) -> pl.DataFrame:
    """
    Add change in motor score compared with the previous age interval for each child.

    Args:
        motor_df (pl.DataFrame): DataFrame containing motor scores per child and age.

    Returns:
        pl.DataFrame: Polars DataFrame with an additional column `delta_motor_score`
        representing the change in motorical_score_2 from the previous age interval.
    """
    return (
        motor_df
        .sort(["introductory_id", "age"])
        .with_columns(
            (
                pl.col("motorical_score_2")
                - pl.col("motorical_score_2").shift(1)
            )
            .over("introductory_id")
            .alias("delta_motor_score")
        )
    )


def prepare_home_features(home_training_df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert home training data from long to wide format.
     Each training type becomes its own column with the total number of hours
    for that therapy at a given age.

    Args:
        home_training_df (pl.DataFrame): Long-format DataFrame with one row per
        child, age, and training type.

    Returns:
        pl.DataFrame: Wide-format Polars DataFrame with one row per
        (introductory_id, age) and one column per training type.
    """
    return (
        home_training_df
        .with_columns(
            (pl.col("training_category") + "_" + pl.col("training_name"))
            .alias("training_name")
        )
        .pivot(
            values="total_hours",
            index=["introductory_id", "age"],
            on="training_name",
        )
        .fill_null(0)
    )


def build_ml_dataset(motor_df: pl.DataFrame, home_df: pl.DataFrame) -> pl.DataFrame:
    """
    Combine motor development data with home training data into a single
    machine learning dataset.

    The function calculates delta_motor_score and joins the motor data with
    home training features.

    Args:
        motor_df (pl.DataFrame): Motor development data with motor scores per child and age.
        home_df (pl.DataFrame): Home training data with therapy hours per child and age.

    Returns:
        pl.DataFrame: Polars DataFrame containing motor scores, delta_motor_score,
        and home training features for each child and age.
    """    
    motor_with_delta = add_delta_motor_score(motor_df)
    home_wide = prepare_home_features(home_df)

    return (
        motor_with_delta
        .join(home_wide, on=["introductory_id", "age"], how="left")
        .fill_null(0)
        .filter(pl.col("delta_motor_score").is_not_null())
    )


def train_model(polars_df: pl.DataFrame) -> RandomForestRegressor:
    """
    Train a Random Forest model to predict change in motor score.

    The model uses therapy hours as features and delta_motor_score as the target.
    Cross-validation is performed to estimate model performance.

    Args:
        polars_df (pl.DataFrame): Machine learning dataset containing motor scores
        and training features.

    Returns:
        RandomForestRegressor: Trained Random Forest model.
    """
    df = polars_df.to_pandas()
    df = df[df["introductory_id"].isin(INCLUDE_IDS)]

    print(f"Using {len(df)} rows from {len(INCLUDE_IDS)} participants")

    y = df["delta_motor_score"]
    X = df.drop(
        columns=[
            "introductory_id",
            "age",
            "motorical_score_2",
            "delta_motor_score",
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=3,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    print("\nDataset evaluation")
    print("CV R² scores:", scores)
    print("Mean R²:", scores.mean())

    print("\nTraining model...")
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    print(f"Training took {end_time - start_time:.2f} seconds")
   
    importance = pd.Series(
        model.feature_importances_,
        index=X.columns,
    ).sort_values(ascending=False)

    print("\nTop 10 most important features:")
    print(importance.head(10))

    # ---------------------------
    # SHAP analysis
    # ---------------------------
    print("\nCalculating SHAP values...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    # Summary plot:  negativt värde → minskar Δ motor score. positivt värde → ökar Δ motor score
    # Hur modellen beter sig över hela datasetet. Visar vilka features som är viktiga i modellen generellt
    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "shap_summary_plot.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Waterfall plot for first sample in training set
    # Hur modellen beter sig för en specifik prediktion. Varför modellen gjorde just den prediktionen för detta barn & år
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "shap_waterfall_plot.png", dpi=300, bbox_inches="tight")
    plt.show()

    return model


def main() -> None:
    """
    Load data, preprocess it, build the machine learning dataset,
    and train the model.
    """
    from connect_db import get_connection
    from dataloader import load_data
    from preprocessing.preprocessing_ht import process_training_per_type_per_year
    from preprocessing.preprocessing_md import process_motorical_score_2_per_user_per_age

    conn = get_connection()
    data = load_data(conn)

    motor_df = process_motorical_score_2_per_user_per_age(
        data["motorical_development"]
    )
    home_df = process_training_per_type_per_year(data["home_training"])

    ml_df = build_ml_dataset(motor_df, home_df)
    train_model(ml_df)


if __name__ == "__main__":
    main()