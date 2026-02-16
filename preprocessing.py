import polars as pl
from connect_db import get_connection

def load_home_training() -> pl.DataFrame:
    query = """
    SELECT 
        id,
        introductory_id,
        training_methods_therapies
    FROM home_training
    """
    with get_connection() as conn:
        return pl.read_database(query, conn)

def extract_total_hours(training):
    if training is None:
        return 0.0
    training_dict = dict(training) if isinstance(training, dict) else {}
    details = training_dict.get("details", {})
    total_hours = 0.0
    for therapy_info in details.values():
        hours = therapy_info.get("hours", 0) if therapy_info else 0
        try:
            total_hours += float(hours)
        except (TypeError, ValueError):
            pass
    return total_hours

def process_home_training_hours(df: pl.DataFrame) -> pl.DataFrame:
    # Convert struct column to list of dicts
    structs = df["training_methods_therapies"].to_list()
    total_hours_list = [extract_total_hours(s) for s in structs]

    # Add as new column
    df = df.with_columns(pl.Series("training_hours", total_hours_list))

    # Aggregate per child
    df_agg = df.group_by("introductory_id").agg([
        pl.sum("training_hours").alias("total_home_training_hours")
    ])

    return df_agg

if __name__ == "__main__":
    home_training = load_home_training()
    print("Original home_training:")
    print(home_training.head())

    processed = process_home_training_hours(home_training)
    print("\nTotal home training hours per child:")
    print(processed)
