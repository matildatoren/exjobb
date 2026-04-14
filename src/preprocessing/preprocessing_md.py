import polars as pl

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

# --------------------------------------------
# Helper methods
# --------------------------------------------

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

def count_impairments(struct) -> int:
    """Count how many impairments have a non-zero rating in details."""
    if struct is None:
        return 0
    data = dict(struct) if isinstance(struct, dict) else {}
    details = data.get("details", {})
    return sum(1 for v in details.values() if v not in (None, 0, 0.0, "", "0"))

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

# --------------------------------------------
# Calculation of 4 different motorical scores
# --------------------------------------------


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

    result = (
        df.group_by(["introductory_id", "age"])
        .agg(pl.mean("motorical_score").alias("motorical_score"))
        .sort(["introductory_id", "age"])
    )

    return result


def process_motorical_score_2_per_user_per_age(
    df: pl.DataFrame,
) -> pl.DataFrame:
    
    #MotorScore 2 = (kumulativt antal unika milestones hittills) / (möjliga milestones vid den åldern)
    #0..1

    possible_milestones_by_age = {
        1: 12,
        2: 19,
        3: 25,
        4: 31,
        5: 36,
        6: 39,
        7: 41
    }

    gross_list = df["gross_motor_development"].to_list()
    fine_list = df["fine_motor_development"].to_list()

    per_row_keys = []
    for g, f in zip(gross_list, fine_list):
        keys = extract_milestone_keys(g) | extract_milestone_keys(f)
        per_row_keys.append(sorted(keys))

    df2 = df.with_columns(pl.Series("milestone_keys", per_row_keys))

    per_age = (
        df2.group_by(["introductory_id", "age"])
        .agg(
            pl.col("milestone_keys")
            .explode()       
            .drop_nulls()  
            .unique()
            .alias("milestone_keys_age")
        )
        .sort(["introductory_id", "age"])
    )

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
            .select(["introductory_id", "age", "cum_unique_milestones", "motorical_score_2"])
            .sort(["introductory_id", "age"])
 )

def calculate_percentile_motor_score_3(
    score2_df: pl.DataFrame,
    introductory_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    MotorScore 3 = percentil-rank av motorical_score_2 inom (age, GMFCS).
    Returnerar 0..1 per (introductory_id, age).
    """

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


def calculate_expected_milestone_score_3(
    score2_df: pl.DataFrame,
    introductory_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    MotorScore 3 = cum_unique_milestones / expected_milestones_for_age_and_gmfcs
    Returns 0..1 per (introductory_id, age).

    The expected milestones matrix is hardcoded based on clinical knowledge.
    Rows = age, Columns = GMFCS level (1-5).
    """

    gmfcs_mapping = {
        "Level I – Walks without limitations": 1,
        "Level II – Walks with limitations": 2,
        "Level III – Walks using a hand-held mobility device": 3,
        "Level IV – Self-mobility with limitations; may use powered mobility": 4,
        "Level V – Transported in a manual wheelchair": 5,
    }

    expected_milestones = {
        # age: {gmfcs_level: expected_milestones}
        # GMFCS I ≈ 95%, II ≈ 80%, III ≈ 60%, IV ≈ 40%, V ≈ 20% of max possible
        1: {1: 11, 2: 10, 3: 7,  4: 5,  5: 2},   # max = 12
        2: {1: 18, 2: 15, 3: 11, 4: 8,  5: 4},   # max = 19
        3: {1: 24, 2: 20, 3: 15, 4: 10, 5: 5},   # max = 25
        4: {1: 29, 2: 25, 3: 19, 4: 12, 5: 6},   # max = 31
        5: {1: 34, 2: 29, 3: 22, 4: 14, 5: 7},   # max = 36
        6: {1: 37, 2: 31, 3: 23, 4: 16, 5: 8},   # max = 39
        7: {1: 39, 2: 33, 3: 25, 4: 16, 5: 8},   # max = 41
    }


    df = score2_df.join(
        introductory_df.select([pl.col("id").alias("introductory_id"), "gmfcs_lvl"]),
        on="introductory_id",
        how="left"
    )

    # Look up expected milestones for each (age, gmfcs) row
    ages = df["age"].to_list()
    gmfcs_levels = df["gmfcs_lvl"].to_list()
    cum_milestones = df["cum_unique_milestones"].to_list()

    scores = []
    for age, gmfcs, cum in zip(ages, gmfcs_levels, cum_milestones):
        try:
            gmfcs_int = gmfcs_mapping.get(gmfcs)    
            if gmfcs_int is None:
                scores.append(None)
                continue
            expected = expected_milestones[int(age)][gmfcs_int]
        except (KeyError, TypeError):
            scores.append(None)
            continue

        if expected is None or expected == 0:
            scores.append(None)
        else:
            score = min(cum / expected, 1.0) 
            scores.append(score)

    return (
        df.with_columns(pl.Series("motorical_score_3", scores))
            .with_columns(pl.col("motorical_score_3").fill_null(0))
            .select(["introductory_id", "age", "gmfcs_lvl", "motorical_score_2", "motorical_score_3"])
            .sort(["introductory_id", "age"])
    )

if __name__ == "__main__":
    from dataloader import load_data
    from connect_db import get_connection

    conn = get_connection()
    data = load_data(conn)

    motorical_dev = data["motorical_development"]

    score1 = process_motorical_score_1(motorical_dev)

    score2 = process_motorical_score_2_per_user_per_age(
        motorical_dev
    )


    score3 = calculate_percentile_motor_score_3(
        score2_df=score2,
        introductory_df=data["introductory"]
    )

    score3_2 = calculate_expected_milestone_score_3(      
        score2_df=score2,
        introductory_df=data["introductory"]
    )


    # -------- Slå ihop allt --------
    final_table = (
        score1
        .join(score2, on=["introductory_id", "age"], how="left")
        .join(
            score3.select(["introductory_id", "age", 
                pl.col("motorical_score_3").alias("motorical_score_3_percentile")]),
            on=["introductory_id", "age"],
            how="left"
        )
        .join(
            score3_2.select(["introductory_id", "age", 
                pl.col("motorical_score_3").alias("motorical_score_3_expected")]),
            on=["introductory_id", "age"],
            how="left"
        )
        .sort(["introductory_id", "age"])
    )
    print("\nFinal motor score table:\n")
    print(final_table)



