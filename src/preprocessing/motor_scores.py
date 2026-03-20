from preprocessing_md import extract_milestone_keys, count_milestones, sum_impairments

import polars as pl

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

# ------------------------------------------------------------------
# normalierat över en satt vektor med "unlocked" milestones/impairments
# ------------------------------------------------------------------

def motorscore_milestones_setvalue(df: pl.DataFrame) -> pl.DataFrame:
    """
    MotorScore = (kumulativt antal unika milestones hittills) / (möjliga milestones vid den åldern)
    Returnerar 0..1 per (introductory_id, age).
    """
    possible_milestones_by_age = {1: 12, 2: 19, 3: 25, 4: 31, 5: 36, 6: 39, 7: 41, 8:41, 9:41}

    # --- Steg 1: Extrahera milestone-nycklar per rad ---
    gross_list = df["gross_motor_development"].to_list()
    fine_list = df["fine_motor_development"].to_list()

    per_row_keys = [
        sorted(extract_milestone_keys(g) | extract_milestone_keys(f))
        for g, f in zip(gross_list, fine_list)
    ]

    df = df.with_columns(pl.Series("milestone_keys", per_row_keys))

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


def motorscore_impairments_setvalue(
    df: pl.DataFrame,
    upper_col: str = "motorical_impairments_upper",
    lower_col: str = "motorical_impairments_lower",
) -> pl.DataFrame:

    def _n_named(age: float) -> int:
        if age < 1:   return 9
        elif age < 2: return 16
        elif age < 3: return 17
        else:         return 18

    upper_list = df[upper_col].to_list()
    lower_list = df[lower_col].to_list()

    row_sum_ratings: list[float] = []
    for u, lo in zip(upper_list, lower_list):
        row_sum_ratings.append(
            sum_impairments(u) + sum_impairments(lo)
        )

    df2 = df.with_columns(
        pl.Series("_sum_ratings", row_sum_ratings)
    )

    per_age = (
        df2.group_by(["introductory_id", "age"])
        .agg(
            pl.col("_sum_ratings").sum().alias("sum_ratings")
        )
        .sort(["introductory_id", "age"])
    )

    ages         = per_age["age"].to_list()
    sum_ratings  = per_age["sum_ratings"].to_list()

    n_named_list  : list[int]   = []
    max_score_list: list[int]   = []
    mms_list      : list[float] = []
    mms_norm_list : list[float] = []

    for age, s in zip(ages, sum_ratings):
        n_named   = _n_named(float(age))
        max_score = n_named * 5
        mms       = max_score - s
        mms_norm  = mms / max_score if max_score > 0 else None

        n_named_list.append(n_named)
        max_score_list.append(max_score)
        mms_list.append(float(mms))
        mms_norm_list.append(float(mms_norm))

    return (
        per_age
        .with_columns([
            pl.Series("n_named",        n_named_list),
            pl.Series("max_score",      max_score_list),
            pl.Series("mms",            mms_list),
            pl.Series("mms_normalized", mms_norm_list),
        ])
        .select([
            "introductory_id", "age",
            "sum_ratings", "n_named",
            "max_score", "mms", "mms_normalized",
        ])
        .sort(["introductory_id", "age"])
    )

# ------------------------------------------------------------------
# normaliserat över alla i den åldersklassen
# ------------------------------------------------------------------

def motorscore_milestones(df: pl.DataFrame) -> pl.DataFrame:
    """
    MotorScore = cum_unique_milestones / max(cum_unique_milestones) inom åldersgruppen
    Returnerar 0..1 per (introductory_id, age).
    """

    # --- Steg 1–3: identiska med motorscore_milestones_setvalue ---
    per_row_keys = [
        sorted(extract_milestone_keys(g) | extract_milestone_keys(f))
        for g, f in zip(df["gross_motor_development"].to_list(), df["fine_motor_development"].to_list())
    ]

    df = df.with_columns(pl.Series("milestone_keys", per_row_keys))

    per_age = (
        df.group_by(["introductory_id", "age"])
        .agg(
            pl.col("milestone_keys").explode().drop_nulls().unique().alias("milestone_keys_age")
        )
        .sort(["introductory_id", "age"])
    )

    def _cumulate(group: pl.DataFrame) -> pl.DataFrame:
        group = group.sort("age")
        seen: set[str] = set()
        cum_counts: list[int] = []
        for keys in group["milestone_keys_age"].to_list():
            cleaned = [k for k in (keys or []) if k is not None and str(k).strip() not in ("", "none")]
            seen |= set(cleaned)
            cum_counts.append(len(seen))
        return group.with_columns(pl.Series("cum_unique_milestones", cum_counts))

    cum = per_age.group_by("introductory_id").map_groups(_cumulate)

    # --- Steg 4: Normalisera mot max inom åldersgruppen ---
    return (
        cum.with_columns(
            pl.col("cum_unique_milestones").mean().over("age").alias("mean_in_age")
        )
        .with_columns(
            (pl.col("cum_unique_milestones") / pl.col("mean_in_age"))
            .cast(pl.Float64)
            .alias("milestone_score")
        )
        .select(["introductory_id", "age", "cum_unique_milestones", "milestone_score"])
        .sort(["introductory_id", "age"])
    )

def motorscore_impairments(
    df: pl.DataFrame,
    upper_col: str = "motorical_impairments_upper",
    lower_col: str = "motorical_impairments_lower",
) -> pl.DataFrame:

    N_NAMED_BY_AGE: dict[int, int] = {
        1: 9,
        2: 16,
        3: 17,
        4: 18,
    }

    def _n_named(age: float) -> int:
        if age < 1:   return N_NAMED_BY_AGE[1]
        elif age < 2: return N_NAMED_BY_AGE[2]
        elif age < 3: return N_NAMED_BY_AGE[3]
        else:         return N_NAMED_BY_AGE[4]

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
    ages       = per_age["age"].to_list()
    other_keys = per_age["other_keys_age"].to_list()

    n_named_list  : list[int]   = []
    n_other_list  : list[int]   = []
    n_total_list  : list[int]   = []
    max_score_list: list[int]   = []
    mms_list      : list[float] = []

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

    per_age = (
        per_age
        .with_columns([
            pl.Series("n_named",   n_named_list),
            pl.Series("n_other",   n_other_list),
            pl.Series("n_total",   n_total_list),
            pl.Series("max_score", max_score_list),
            pl.Series("mms",       mms_list),
        ])
    )

    # ------------------------------------------------------------------ #
    # 4. Normalize by dividing by the mean MMS at each age               #
    # ------------------------------------------------------------------ #
    age_means = (
        per_age
        .group_by("age")
        .agg(pl.col("mms").mean().alias("mms_age_mean"))
    )

    return (
        per_age
        .join(age_means, on="age", how="left")
        .with_columns(
            pl.when(pl.col("mms_age_mean") > 0)
            .then(pl.col("mms") / pl.col("mms_age_mean"))
            .otherwise(None)
            .cast(pl.Float64)
            .alias("mms_normalized")
        )
        .select([
            "introductory_id", "age",
            "sum_ratings", "n_other", "n_named", "n_total",
            "max_score", "mms", "mms_age_mean", "mms_normalized",
        ])
        .sort(["introductory_id", "age"])
    )


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

    milestonevalue1 = motorscore_milestones_setvalue(motorical_dev)
    milestonesvalue2 = motorscore_milestones(motorical_dev)
    print('First milestone score')
    print(milestonevalue1)
    print('Second milestone score')
    print(milestonesvalue2)

    impairmentvalue1 = motorscore_impairments_setvalue(motorical_dev)
    impairmentvalue2 = motorscore_impairments(motorical_dev)
    print('First impairment score')
    print(impairmentvalue1)
    print('Second impairment score')
    print(impairmentvalue2)
  