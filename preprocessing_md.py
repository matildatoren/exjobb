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

def extract_milestone_keys(struct) -> set[str]:
    """
    Gör milestones-listan till ett set av stabila nycklar (id/label/value eller str()).
    Krävs för att kunna räkna unika milestones över år.
    """
    if struct is None:
        return set()
    data = dict(struct) if isinstance(struct, dict) else {}
    milestones = data.get("milestones", [])
    if not milestones:
        return set()

    keys = set()
    for m in milestones:
        if isinstance(m, dict):
            if m.get("id") is not None:
                keys.add(str(m["id"]))
            elif m.get("value") is not None:
                keys.add(str(m["value"]))
            elif m.get("label") is not None:
                keys.add(str(m["label"]))
            else:
                keys.add(str(m))
        else:
            keys.add(str(m))
    return keys


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


def process_motorical_score_2_per_user_per_age(
    df: pl.DataFrame,
    possible_milestones_by_age: dict[int, int],
) -> pl.DataFrame:
    
    #MotorScore 2 = (kumulativt antal unika milestones hittills) / (möjliga milestones vid den åldern)
    #0..1

    # per rad -> vilka milestones uppnåddes detta år (gross+fine)
    gross_list = df["gross_motor_development"].to_list()
    fine_list = df["fine_motor_development"].to_list()

    per_row_keys = []
    for g, f in zip(gross_list, fine_list):
        per_row_keys.append(sorted(extract_milestone_keys(g) | extract_milestone_keys(f)))

    df2 = df.with_columns(pl.Series("milestone_keys", per_row_keys))

    # slå ihop om det finns flera rader för samma (id, age): union inom året
    per_age = (
        df2.group_by(["introductory_id", "age"])
        .agg(pl.col("milestone_keys").flatten().unique().alias("milestone_keys_age"))
        .sort(["introductory_id", "age"])
    )

    # kumulativ union per individ
    def _cumulate(group: pl.DataFrame) -> pl.DataFrame:
        group = group.sort("age")
        seen = set()
        cum_counts = []

        for keys in group["milestone_keys_age"].to_list():
            keys = keys or []
            seen |= set(keys)
            cum_counts.append(len(seen))

        return group.with_columns(pl.Series("cum_unique_milestones", cum_counts))

    cum = per_age.group_by("introductory_id").map_groups(_cumulate)

    # normalisering med möjliga milestones vid åldern
    ages = cum["age"].to_list()
    possible = [possible_milestones_by_age.get(int(a), None) for a in ages]

    return (
        cum.with_columns(pl.Series("possible_milestones", possible))
        .with_columns(
            (pl.col("cum_unique_milestones") / pl.col("possible_milestones"))
            .cast(pl.Float64)
            .alias("motorical_score_2")
        )
        .select(["introductory_id", "age", "motorical_score_2"])
        .sort(["introductory_id", "age"])
 )

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

    possible_milestones_by_age = {
        1: 12,
        2: 19,
        3: 25,
        4: 31,
        5: 36,
        6: 39,
        7: 41
    }

    score2 = process_motorical_score_2_per_user_per_age(
        motorical_dev, possible_milestones_by_age
    )

    print(motor_score_table)
    print(score2)

