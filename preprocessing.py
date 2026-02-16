"""
Minimal preprocessing script: calculate total home training hours per child.
Handles multiple yearly rows per child and struct columns.
"""
import polars as pl
from dataloader import load_data
import os
import dotenv
from dotenv import load_dotenv



def extract_training_hours_from_struct(struct_col: pl.Series) -> pl.Series:
    return struct_col.struct.field("hours").fill_null(0)

def process_home_training_hours(df: pl.DataFrame) -> pl.DataFrame:
    # Extract hours from struct column
    df = df.with_columns([
        extract_training_hours_from_struct(pl.col("training_methods_therapies")).alias("training_hours")
    ])
    
    # Aggregate per introductory_id (sum all yearly rows)
    df_agg = df.groupby("introductory_id").agg([
        pl.sum("training_hours").alias("total_home_training_hours")
    ])
    
    return df_agg

# --- Main for testing ---
if __name__ == "__main__":
    load_dotenv()

    connection_string = os.getenv("DB_URL")
    data = load_data(connection_string)
    
    home_training = data["home_training"]
    print("Original home_training shape:", home_training.shape)
    print(home_training.head())

    processed = process_home_training_hours(home_training)
    print("\nProcessed home_training_hours shape:", processed.shape)
    print(processed.head())
