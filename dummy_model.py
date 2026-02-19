import polars as pl
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error


def add_delta_motor_score(motor_df: pl.DataFrame) -> pl.DataFrame:

    motor_df = motor_df.sort(["introductory_id", "age"])

    motor_df = motor_df.with_columns(
        (
            pl.col("motorical_score_2")
            - pl.col("motorical_score_2").shift(1)
        )
        .over("introductory_id")
        .alias("delta_motor_score")
    )

    return motor_df


def prepare_home_features(home_training_df: pl.DataFrame) -> pl.DataFrame:

    home_wide = (
        home_training_df
        .pivot(
            values="total_hours",
            index=["introductory_id", "age"],
            columns="training_name"
        )
        .fill_null(0)
    )

    return home_wide


def build_ml_dataset(motor_df, home_df):

    motor_df = add_delta_motor_score(motor_df)
    home_wide = prepare_home_features(home_df)

    df = motor_df.join(
        home_wide,
        on=["introductory_id", "age"],
        how="left"
    ).fill_null(0)

    df = df.filter(pl.col("delta_motor_score").is_not_null())

    return df

def train_model(polars_df):

    import time

    df = polars_df.to_pandas()

    y = df["delta_motor_score"]

    X = df.drop(
        columns=[
            "introductory_id",
            "age",
            "motorical_score_2",
            "delta_motor_score"
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        verbose=1,
        n_jobs=-1
    )

    print("\nTraining model...")
    start_time = time.time()

    model.fit(X_train, y_train)

    end_time = time.time()
    print(f"\nTraining took {end_time - start_time:.2f} seconds")

    preds = model.predict(X_test)

    print("\nModel evaluation:")
    print("R²:", r2_score(y_test, preds))
    print("MAE:", mean_absolute_error(y_test, preds))
    print("MSE:", mean_squared_error(y_test, preds))

    importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    print("\nTop 10 most important features:\n")
    print(importance.head(10))

    return model



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

    motor_df = process_motorical_score_2_per_user_per_age(data["motorical_development"], possible_milestones_by_age)
    home_df = process_training_per_type_per_year(data["home_training"])

    ml_df = build_ml_dataset(motor_df, home_df)

    model = train_model(ml_df)
