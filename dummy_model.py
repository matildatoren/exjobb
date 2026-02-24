import polars as pl
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


def add_delta_motor_score(motor_df: pl.DataFrame) -> pl.DataFrame: # Funktion som tar en Polars-dataframe med motor-data och lägger till en ny kolumn.

    motor_df = motor_df.sort(["introductory_id", "age"]) # sorterar barnens rader efter ålder (annars funkar ej shift(1))

    motor_df = motor_df.with_columns(
        (
            pl.col("motorical_score_2") # Nuvarande score
            - pl.col("motorical_score_2").shift(1) # Score året innan (raden innan)
        )
        .over("introductory_id") # Ser till så att raden innan är samma barn 
        .alias("delta_motor_score") # Döper nya kolumnen till delta_motor_score
    )

    return motor_df # Returnerar med en extra kolumn


def prepare_home_features(home_training_df: pl.DataFrame) -> pl.DataFrame:
# Tar hemträning i “long format” (en rad per therapy-typ) och gör om till “wide format” (en kolumn per therapy-typ).
    home_wide = (
        home_training_df
        .with_columns( # Added to include all therapies even if there are duplicates in therapy at home
            (pl.col("training_category") + "_" + pl.col("training_name"))
            .alias("training_name")
        ) # skriver över kolumnen training_name med ett nytt kombinerat namn.
        .pivot(
            values="total_hours", # Cellvärdet
            index=["introductory_id", "age"], # Detta blir raderna
            on="training_name" # Detta blir kolumnerna
        )
        .fill_null(0) # Om det ej finns nåt, lägg en nolla
    )

    return home_wide


def build_ml_dataset(motor_df, home_df): # Tar motor-data och home-training-data och gör en ML-tabell.

    motor_df = add_delta_motor_score(motor_df)
    home_wide = prepare_home_features(home_df)

    df = motor_df.join(
        home_wide,
        on=["introductory_id", "age"],
        how="left" # Behåll alla motor-rader även om home saknas.
    ).fill_null(0) # Om home-träning saknas → anta 0 timmar.

    df = df.filter(pl.col("delta_motor_score").is_not_null())
    # Tar bort första året per barn (där delta blir null eftersom det inte finns något “föregående år” att jämföra med).
    return df

def train_model(polars_df): # Tränar modellen på den färdiga ML tabellen

    import time # För att mäta träningstid

    df = polars_df.to_pandas() # Sklearn jobbar lättast med pandas/numpy.

    #Ändra här för att välja vilka surveys som används när man tränar
    include_ids = ["65ab3206-7371-4471-845c-6d238050494f", 
                    "9a3aeeeb-b409-4052-af0e-27e4893fb48f", 
                    "a1c29f34-c3a0-4140-8398-e3d8eb980292", 
                    "f9231c8d-2ade-4c0e-a878-a9524ccc3d65"]  
    df = df[df["introductory_id"].isin(include_ids)]
    print(f"Using {len(df)} rows from {len(include_ids)} participants")


    y = df["delta_motor_score"] # Target = förändring i motor score. Modellen försöker lära sig prediktera den kolumnen.

    X = df.drop( # Features = allt utom det i lastan nedan
        columns=[
            "introductory_id",
            "age",
            "motorical_score_2",
            "delta_motor_score"
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split( # 80% train
        X, y, test_size=0.2, random_state=42 
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=3,         
        min_samples_leaf=5,   
        random_state=42,
        n_jobs=-1
    )


    scores = cross_val_score(model, X, y, cv=5, scoring="r2") # 5 fold cross validation på hela X,y. Tränar 5 gngr och ger 5 R2 värden
    print("\nDataset evaluation:")
    print("CV R² scores:", scores)
    print("Mean R²:", scores.mean())

    print("\nTraining model...") # Här lär sig modellen sambandet mellan therapy-timmar och delta-motor
    start_time = time.time()

    model.fit(X_train, y_train) # Modellen försöker hitta: f(therapy hours)≈ Δ motorscore

    end_time = time.time()
    print(f"\nTraining took {end_time - start_time:.2f} seconds")


    importance = pd.Series(
        model.feature_importances_, # Hur viktiga features var (baserat på hur mycket de minskade impurity). Vad som hjälpte mest i prediktionen

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

# Hämtar data från databasen
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
# Skapar 2 tabeller. 
    motor_df = process_motorical_score_2_per_user_per_age(data["motorical_development"], possible_milestones_by_age) #en rad per barn+ålder med motorical_score_2
    home_df = process_training_per_type_per_year(data["home_training"]) # En rad per barn+ålder+training med total_hours

    ml_df = build_ml_dataset(motor_df, home_df) # Joinar ihop o skapar delta

    model = train_model(ml_df) # Tränar modellen
