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


def count_milestones(struct):
    if struct is None:
        return 0

    data = dict(struct) if isinstance(struct, dict) else {}
    milestones = data.get("milestones", [])

    return len(milestones) if milestones else 0


def sum_impairments(struct):
    if struct is None:
        return 0.0

    data = dict(struct) if isinstance(struct, dict) else {}
    details = data.get("details", {})

    total = 0.0
    for value in details.values():
        try:
            total += float(value)
        except (TypeError, ValueError):
            pass

    return total

def process_motorical_score(df: pl.DataFrame) -> pl.DataFrame:
    gross_list = df["gross_motor_development"].to_list()
    fine_list = df["fine_motor_development"].to_list()
    lower_list = df["motorical_impairments_lower"].to_list()
    upper_list = df["motorical_impairments_upper"].to_list()

    scores = []

    for g, f, l, u in zip(gross_list, fine_list, lower_list, upper_list):

        milestones = count_milestones(g) + count_milestones(f)
        impairments = sum_impairments(l) + sum_impairments(u)

        motor_score = milestones - (0.2 * impairments)
        scores.append(motor_score)

    df = df.with_columns(
        pl.Series("motorical_score", scores)
    )

    return df.select(["introductory_id", "motorical_score"])


def process_motorical_score_per_user_per_age(df: pl.DataFrame) -> pl.DataFrame:

    gross_list = df["gross_motor_development"].to_list()
    fine_list = df["fine_motor_development"].to_list()
    lower_list = df["motorical_impairments_lower"].to_list()
    upper_list = df["motorical_impairments_upper"].to_list()

    scores = []

    for g, f, l, u in zip(gross_list, fine_list, lower_list, upper_list):

        milestones = count_milestones(g) + count_milestones(f)
        impairments = sum_impairments(l) + sum_impairments(u)

        motor_score = milestones - (0.2 * impairments)
        scores.append(motor_score)

    df = df.with_columns(
        pl.Series("motorical_score", scores)
    )

    # Group per introductory_id AND age
    result = (
        df.group_by(["introductory_id", "age"])
        .agg(pl.mean("motorical_score").alias("motorical_score"))
        .sort(["introductory_id", "age"])
    )

    return result


if __name__ == "__main__":
    # Import your dataloader
    from dataloader import load_data
    from connect_db import get_connection

    conn = get_connection()
    data = load_data(conn)

    # Use the home_training table from your loader
    home_training = data["home_training"]
    motorical_dev = data["motorical_development"]

   # Home training
    print("Original home_training:")
    print(home_training.head())
    processed_home = process_home_training_hours(home_training)
    print("\nTotal home training hours per child:")
    print(processed_home)

    processed_home = process_home_training_hours_per_user_per_year(home_training)

    print("\nTotal home training hours per user per year:")
    print(processed_home)

    motor_score_table = process_motorical_score_per_user_per_age(motorical_dev)

    print("\nMotorical score per introductory_id per age:\n")
    print(motor_score_table)
