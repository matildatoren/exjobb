from preprocessing_md import extract_milestone_keys, count_milestones, sum_impairments

import polars as pl

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


# normalierat över en satt vektor med "unlocked" milestones/impairments
def motorscore_milestones_setvalue() -> float:
    possible_milestones_by_age = {1: 12, 2: 19, 3: 25, 4: 31, 5: 36, 6: 39, 7: 41}
    print("hej")


def motorscore_impairments_setvalue(
    df: pl.DataFrame,
    upper_col: str = "motorical_impairments_upper",
    lower_col: str = "motorical_impairments_lower",
) -> pl.DataFrame:
    """
    Computes the Motor Milestone Score (MMS) per participant per age.

    Formula:
        MMS = (N_named(age) + N_other) * 5 - sum(r_i)

    N_named is the number of age-appropriate named impairments:
        age < 1  →  9   (upper tone + early fine motor)
        age 1–2  → 16   (+ 5 gait + 2 fine motor)
        age 2–3  → 17   (+ crouch gait)
        age ≥ 3  → 18   (+ jump gait, all named)

    N_other is the count of "Other" milestone keys selected
    (detected by the substring "other" in the key, case-insensitive).

    Parameters
    ----------
    df         : polars DataFrame, one row per submission/timepoint
    upper_col  : column name for upper body impairment struct
    lower_col  : column name for lower body impairment struct

    Returns
    -------
    DataFrame with columns:
        introductory_id, age, sum_ratings, n_other,
        n_named, n_total, max_score, mms
    """

    N_NAMED_BY_AGE: dict[int, int] = {
        1: 9,   # age 0–<1: upper tone + early fine motor only
        2: 16,  # age 1–<2: + 5 gait + 2 fine motor
        3: 17,  # age 2–<3: + crouch gait
        4: 18,  # age ≥3:   + jump gait (all named impairments)
    }

    def _n_named(age: float) -> int:
        if age < 1:
            return N_NAMED_BY_AGE[1]
        elif age < 2:
            return N_NAMED_BY_AGE[2]
        elif age < 3:
            return N_NAMED_BY_AGE[3]
        else:
            return N_NAMED_BY_AGE[4]

    # ------------------------------------------------------------------ #
    # 1. Extract per-row: sum of ratings + "other" keys from both regions #
    # ------------------------------------------------------------------ #
    upper_list = df[upper_col].to_list()
    lower_list = df[lower_col].to_list()

    row_sum_ratings: list[float] = []
    row_other_keys: list[list[str]] = []

    for u, lo in zip(upper_list, lower_list):
        row_sum_ratings.append(
            sum_impairments(u) + sum_impairments(lo)
        )
        all_keys = extract_milestone_keys(u) | extract_milestone_keys(lo)
        others = sorted(k for k in all_keys if "other" in k.lower())
        row_other_keys.append(others)

    df2 = df.with_columns([
        pl.Series("_sum_ratings", row_sum_ratings),
        pl.Series("_other_keys", row_other_keys),
    ])

    # ------------------------------------------------------------------ #
    # 2. Aggregate per participant per age                                #
    #    — sum all ratings, collect unique "other" keys                  #
    # ------------------------------------------------------------------ #
    per_age = (
        df2.group_by(["introductory_id", "age"])
        .agg([
            pl.col("_sum_ratings").sum().alias("sum_ratings"),
            pl.col("_other_keys")
              .explode()
              .drop_nulls()
              .unique()
              .alias("other_keys_age"),
        ])
        .sort(["introductory_id", "age"])
    )

    # ------------------------------------------------------------------ #
    # 3. Derive N_other, N_named, max_score, MMS                         #
    # ------------------------------------------------------------------ #
    ages         = per_age["age"].to_list()
    other_keys   = per_age["other_keys_age"].to_list()

    n_named_list : list[int]   = []
    n_other_list : list[int]   = []
    n_total_list : list[int]   = []
    max_score_list: list[int]  = []
    mms_list     : list[float] = []

    for age, o_keys, s_ratings in zip(
        ages, other_keys, per_age["sum_ratings"].to_list()
    ):
        n_named  = _n_named(float(age))
        n_other  = len(o_keys) if o_keys else 0
        n_total  = n_named + n_other
        max_sc   = n_total * 5
        mms      = max_sc - s_ratings

        n_named_list.append(n_named)
        n_other_list.append(n_other)
        n_total_list.append(n_total)
        max_score_list.append(max_sc)
        mms_list.append(float(mms))

    return (
        per_age
        .with_columns([
            pl.Series("n_named",    n_named_list),
            pl.Series("n_other",    n_other_list),
            pl.Series("n_total",    n_total_list),
            pl.Series("max_score",  max_score_list),
            pl.Series("mms",        mms_list),
        ])
        .select([
            "introductory_id", "age",
            "sum_ratings",
            "n_other", "n_named", "n_total",
            "max_score", "mms",
        ])
        .sort(["introductory_id", "age"])
    )


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

if __name__ == "__main__":
    from dataloader import load_data
    from connect_db import get_connection

    conn = get_connection()
    data = load_data(conn)

    motorical_dev = data["motorical_development"]

    impairmentsSetValue = motorscore_impairments_setvalue(motorical_dev)

    print(impairmentsSetValue)