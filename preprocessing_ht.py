import polars as pl

# ----- Home training functions -----

def extract_total_hours(training):
    if training is None:
        return 0.0

    training_dict = dict(training) if isinstance(training, dict) else {}
    details = training_dict.get("details", {})
    total_hours = 0.0

    for training_info in details.values():
        if not training_info:
            continue

        hours = training_info.get("hours", 0)
        days = training_info.get("days", 1)
        weeks = training_info.get("weeks", 1)

        try:
            hours = float(hours or 0)
            days = float(days or 0)
            weeks = float(weeks or 0)

            total_hours += hours * days * weeks

        except (TypeError, ValueError):
            pass

    return total_hours


def process_home_training_hours_per_user_per_year(df: pl.DataFrame) -> pl.DataFrame:
    # Calculate hours * days * weeks
    structs = df["training_methods_therapies"].to_list()
    total_hours_list = [extract_total_hours(s) for s in structs]

    df = df.with_columns(
        pl.Series("training_hours", total_hours_list)
    )

    df_grouped = (
        df.group_by(["introductory_id", "age"])
        .agg(
            pl.sum("training_hours").alias("total_home_training_hours")
        )
        .sort(["introductory_id", "age"])
    )

    return df_grouped



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
    # Import your dataloader
    from dataloader import load_data
    from connect_db import get_connection

    conn = get_connection()
    data = load_data(conn)

    # Use the home_training table from your loader
    home_training = data["home_training"]


    processed_home = process_home_training_hours(home_training)
    print("\nTotal home training hours per child:")
    print(processed_home)

    processed_home = process_home_training_hours_per_user_per_year(home_training)

    print("\nTotal home training hours per user per year:")
    print(processed_home)
