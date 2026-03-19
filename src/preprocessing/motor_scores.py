from preprocessing_md import extract_milestone_keys, count_milestones, sum_impairments

import polars as pl

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


# normalierat över en satt vektor med "unlocked" milestones/impairments
def motorscore_milestones_setvalue(df: pl.DataFrame) -> pl.DataFrame:
    """
    MotorScore = (kumulativt antal unika milestones hittills) / (möjliga milestones vid den åldern)
    Returnerar 0..1 per (introductory_id, age).
    """
    possible_milestones_by_age = {1: 12, 2: 19, 3: 25, 4: 31, 5: 36, 6: 39, 7: 41}

    # --- Steg 1: Extrahera milestone-nycklar per rad ---
    gross_list = df["gross_motor_development"].to_list()
    fine_list = df["fine_motor_development"].to_list()

    per_row_keys = [
        sorted(extract_milestone_keys(g) | extract_milestone_keys(f))
        for g, f in zip(gross_list, fine_list)
    ]

    # --- Steg 2: Aggregera unika milestones per (user, age) ---
    per_age = (
        df.group_by(["introductory_id", "age"])
        .agg(
            pl.col("milestone_keys")
            .explode()
            .drop_nulls()
            .unique()
            .alias("milestone_keys_age")
        )
        .sort(["introductory_id", "age"])
    )

    # --- Steg 3: Kumulera unika milestones per user över tid ---
    def _cumulate(group: pl.DataFrame) -> pl.DataFrame:
        group = group.sort("age")
        seen: set[str] = set()
        cum_counts: list[int] = []

        for keys in group["milestone_keys_age"].to_list():
            cleaned = [
                k for k in (keys or [])
                if k is not None and str(k).strip() not in ("", "none")
            ]
            seen |= set(cleaned)
            cum_counts.append(len(seen))

        return group.with_columns(pl.Series("cum_unique_milestones", cum_counts))

    cum = per_age.group_by("introductory_id").map_groups(_cumulate)

    # --- Steg 4: Normalisera mot möjliga milestones vid åldern ---
    possible = [
        possible_milestones_by_age.get(int(a)) for a in cum["age"].to_list()
    ]

    return (
        cum.with_columns(pl.Series("possible_milestones", possible))
        .with_columns(
            (pl.col("cum_unique_milestones") / pl.col("possible_milestones"))
            .cast(pl.Float64)
            .alias("milestone_score")
        )
        .select(["introductory_id", "age", "cum_unique_milestones", "milestone_score"])
        .sort(["introductory_id", "age"])
    )

def motorscore_impairments_setvalue() -> float:
    print("hej")

# normaliserat över alla i den åldersklassen
def motorscore_milestones() -> float:
    print("hej")

def motorscore_impairments() -> float:
    print("hej")

# normaliserat över alla i den åldersklassen och gmfcs nivån
def motorscore_milestones_future() -> float:
    print("hej")

def motorscore_impairments_future() -> float:
    print("hej")

# kombinerat score 
def motorscore_combined(imScore: float, moScore: float) -> float:
    print("hej")
