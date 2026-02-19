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
     Filtrerar bort "tomma" placeholders (None, "", "None", dict utan id/value/label).
    """
    if struct is None:
        return set()
    
    data = struct if isinstance(struct, dict) else {}
    milestones = data.get("milestones", [])
    if not milestones:
        return set()
    
    keys: set[str] = set()

    for m in milestones:
    # 1) Ignorera explicita None
        if m is None:
            continue

        # 2) Dict-milestones: ta id/value/label om finns, annars ignorera "tom" dict
        if isinstance(m, dict):
            mid = m.get("id")
            val = m.get("value")
            lab = m.get("label")

            # Tom placeholder-dict -> ignorera
            if mid is None and val is None and lab is None:
                continue

            if mid is not None:
                keys.add(str(mid))
            elif val is not None:
                keys.add(str(val))
            elif lab is not None:
                keys.add(str(lab))
            else:
                # fallback (borde sällan trigga efter filtren ovan)
                s = str(m).strip()
                if s and s.lower() != "none":
                    keys.add(s)

        # 3) Icke-dict milestones (t.ex. str)
        else:
            s = str(m).strip()
            if not s or s.lower() == "none":
                continue
            keys.add(s)

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
        keys = extract_milestone_keys(g) | extract_milestone_keys(f)
        per_row_keys.append(sorted(keys))

    df2 = df.with_columns(pl.Series("milestone_keys", per_row_keys))

    # slå ihop om det finns flera rader för samma (id, age): union inom året
   # slå ihop om det finns flera rader för samma (id, age): union inom året
    per_age = (
        df2.group_by(["introductory_id", "age"])
        .agg(
            pl.col("milestone_keys")
            .list.explode(keep_nulls=False)   # tar bort nulls och ersätter flatten()-deprecated
            .unique()
            .alias("milestone_keys_age")
        )
        .sort(["introductory_id", "age"])
    )

    # kumulativ union per individ
    def _cumulate(group: pl.DataFrame) -> pl.DataFrame:
        group = group.sort("age")
        seen: set[str] = set()
        cum_counts: list[int] = []

        for keys in group["milestone_keys_age"].to_list():
            keys = keys or []
            # failsafe: filtrera bort None/"none"/tomma strängar om något ändå slinker igenom
            cleaned = [
                k for k in keys
                if k is not None and str(k).strip() != "" and str(k).lower() != "none"
            ]
            seen |= set(cleaned)
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

def process_motorical_score_3_within_age_gmfcs(
    score2_df: pl.DataFrame,
    introductory_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    MotorScore 3 = percentil-rank av motorical_score_2 inom (age, GMFCS).
    Returnerar 0..1 per (introductory_id, age).
    """

    # Koppla GMFCS till varje (introductory_id, age)
    df = score2_df.join(
        introductory_df.select([pl.col("id").alias("introductory_id"), "gmfcs_lvl"]),
        on="introductory_id",
        how="left"
    )

    # percentil inom (age, gmfcs): (rank-1)/(n-1). Om n=1 -> 1.0
    df = df.with_columns(
        pl.len().over(["age", "gmfcs_lvl"]).alias("_n"),
        pl.col("motorical_score_2")
          .rank(method="average")
          .over(["age", "gmfcs_lvl"])
          .alias("_r"),
    ).with_columns(
        pl.when(pl.col("_n") <= 1)
          .then(pl.lit(1.0))
          .otherwise((pl.col("_r") - 1) / (pl.col("_n") - 1))
          .alias("motorical_score_3")
    ).drop(["_n", "_r"])

    return (
        df.select(["introductory_id", "age", "gmfcs_lvl", "motorical_score_2", "motorical_score_3"])
          .sort(["introductory_id", "age"])
    )


if __name__ == "__main__":
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
    score1 = process_motorical_score_per_user_per_age(motorical_dev)

    score2 = process_motorical_score_2_per_user_per_age(
        motorical_dev, possible_milestones_by_age
    )


    score3 = process_motorical_score_3_within_age_gmfcs(
        score2_df=score2,
        introductory_df=intro
    )

    # -------- Slå ihop allt --------
    final_table = (
        score1
        .join(score2, on=["introductory_id", "age"], how="left")
        .join(
            score3.select(["introductory_id", "age", "motorical_score_3"]),
            on=["introductory_id", "age"],
            how="left"
        )
        .sort(["introductory_id", "age"])
    )

    print("\nFinal motor score table:\n")
    print(final_table)

