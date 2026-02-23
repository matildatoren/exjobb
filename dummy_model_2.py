import polars as pl
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


def train_simple_model(polars_df: pl.DataFrame):

    df = polars_df.to_pandas()

    include_ids = ["a1c29f34-c3a0-4140-8398-e3d8eb980292", "f9231c8d-2ade-4c0e-a878-a9524ccc3d65", "9a3aeeeb-b409-4052-af0e-27e4893fb48f", "65ab3206-7371-4471-845c-6d238050494f"]  # your completed survey ids
    df = df[df["introductory_id"].isin(include_ids)]
    print(f"Using {len(df)} rows from {len(include_ids)} participants")

    features_of_interest = [
        'devices_AFOs', 
        'devices_Handsplint', 
        'devices_Kinesiotaping', 
        'devices_Standing frame', 
        'devices_inga'
    ]

    # Only keep features that actually exist in the dataframe
    features_of_interest = [f for f in features_of_interest if f in df.columns]
    print("Matched features:", features_of_interest)
    print("\nAll available columns:\n", df.columns.tolist())

    X = df[features_of_interest]
    y = df["delta_motor_score"]

    model = LinearRegression()

    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    print("\nCV R² scores:", scores)
    print("Mean R²:", scores.mean())

    # Fit on full data to inspect coefficients
    model.fit(X, y)

    results = pd.DataFrame({
        "feature": features_of_interest,
        "coefficient": model.coef_
    }).sort_values("coefficient", ascending=False)

    print("\nCoefficients (positive = associated with improvement):")
    print(results)
    print(f"\nIntercept: {model.intercept_:.4f}")

    return model


if __name__ == "__main__":
    from dataloader import load_data
    from connect_db import get_connection
    from preprocessing_md import process_motorical_score_2_per_user_per_age
    from preprocessing_ht import process_training_per_type_per_year
    from dummy_model import build_ml_dataset

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

    ml_df = build_ml_dataset(motor_df, home_df)

    train_simple_model(ml_df)