import polars as pl

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


def process_motorical_score_1(df: pl.DataFrame) -> pl.DataFrame:
    
    gross_list = df["gross_motor_development"].to_list()
    fine_list = df["fine_motor_development"].to_list()
    lower_list = df["motorical_impairments_lower"].to_list()
    upper_list = df["motorical_impairments_upper"].to_list()

    scores = []

    for g, f, l, u in zip(gross_list, fine_list, lower_list, upper_list):

        milestones = count_milestones(g) + count_milestones(f)
        impairments = sum_impairments(l) + sum_impairments(u)

        motor_score = milestones - (0.1 * impairments)
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
    motorical_dev = data["motorical_development"]

    motor_score_table = process_motorical_score_1(motorical_dev)

    print("\nMotorical score per introductory_id per age:\n")
    print(motor_score_table)
