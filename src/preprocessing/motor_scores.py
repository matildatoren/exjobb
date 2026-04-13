from src.preprocessing.preprocessing_md import extract_milestone_keys, count_milestones, sum_impairments

import polars as pl

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

# ------------------------------------------------------------------
# Lookup tables
# ------------------------------------------------------------------

POSSIBLE_MILESTONES_BY_AGE_GMFCS: dict[int, dict[int, int]] = {
    # ┌─────────────────────────────────────────────────────────────────────────┐
    # │ Gross motor (21 total)                                                  │
    # │  Pre-walking   1–6 : rolls, sits, crawls, stands w support, cruises     │
    # │  Basic walking 7–11: first steps, stairs on hands/knees, squats, runs   │
    # │  Advanced     12–21: balance, tricycle, hops, skips, bike, rope, sports │
    # │                                                                         │
    # │ Fine motor (20 total)                                                   │
    # │  Bucket 1  1–5 : grasps, transfers, pincer, self-feeds                 │
    # │  Bucket 2  6–8 : turns pages, stacks blocks, scribbles                 │
    # │  Bucket 3  9–14: copies lines/shapes, scissors, draws, pencil grasp    │
    # │  Bucket 4 15–20: laces, writes words, threading needle, craftwork      │
    # │                                                                         │
    # │ GMFCS IV/V: gross motor 7–21 BLOCKED (cannot walk)                     │
    # │ GMFCS III : gross motor 11–21 BLOCKED (no running/jumping/hopping)     │
    # │ GMFCS II  : gross motor 17–21 reduced (hops, skips, 2-wheel bike hard) │
    # └─────────────────────────────────────────────────────────────────────────┘
    #        I    II   III   IV    V
    1: {1: 12, 2: 11, 3:  9, 4:  7, 5:  4},  # pre-walking: small gap between levels
    2: {1: 19, 2: 17, 3: 13, 4:  9, 5:  5},  # IV/V gain only fine motor; III walks but can't run
    3: {1: 27, 2: 23, 3: 16, 4: 11, 5:  6},  # III plateaus (no running/jumping); IV/V fine motor only
    4: {1: 35, 2: 29, 3: 18, 4: 12, 5:  7},  # I/II gain advanced milestones; IV/V max ~3 gross + 9 fine
}

N_NAMED_BY_AGE_GMFCS: dict[int, dict[int, int]] = {
    # ┌─────────────────────────────────────────────────────────────────────────┐
    # │ Lower-body gait impairments (7 named):                                  │
    # │  In-toeing, out-toeing, crouch, scissoring, toe-walking,               │
    # │  jump gait, dropping foot                                               │
    # │  → GMFCS I/II : all 7 applicable                                        │
    # │  → GMFCS III  : ~5 applicable (gait visible when walking with device)   │
    # │  → GMFCS IV/V : 0 applicable — non-walkers, gait patterns absent        │
    # │                                                                         │
    # │ Upper-body impairments (11 named):                                      │
    # │  Grip, releasing, thumb-in-palm, stiff fingers, finger control,        │
    # │  pinching, wrist flexion, pronation, bilateral coord, elbow ext,       │
    # │  difficulty using both hands                                            │
    # │  → All GMFCS applicable; GMFCS V reduced (severe involvement)          │
    # │                                                                         │
    # │ Year 1: child not walking yet regardless → fewer gait signs visible     │
    # └─────────────────────────────────────────────────────────────────────────┘
    #        I    II   III   IV    V
    1: {1:  9, 2:  9, 3:  8, 4:  7, 5:  5},  # no walking yet; 3 gait signs + 6 upper for I/II
    2: {1: 16, 2: 16, 3: 14, 4: 11, 5:  7},  # I/II: 6 gait + 10 upper; IV/V: 0 gait + 11/7 upper
    3: {1: 17, 2: 17, 3: 15, 4: 11, 5:  8},  # I/II: all 7 gait + 10 upper; III: 5 gait + 10 upper
    4: {1: 18, 2: 18, 3: 16, 4: 11, 5:  8},  # I/II: 7 gait + 11 upper = 18; IV/V: 0 gait + 11/8 upper
}
 
_GMFCS_STR_TO_INT: dict[str, int] = {
    "Level I – Walks without limitations": 1,
    "Level II – Walks with some limitations": 2,
    "Level III – Walks with assistive devices": 3,
    "Level IV – Limited mobility, primarily uses a wheelchair": 4,
    "Level V – Severe limitations, needs full assistance for mobility": 5,
}
 
# ════════════════════════════════════════════════════════════════════════════
# Shared internal helpers
# ════════════════════════════════════════════════════════════════════════════
 
def _gmfcs_lookup(introductory_df: pl.DataFrame) -> dict[str, int]:
    """Return {introductory_id: gmfcs_int} from the introductory table."""
    return {
        uid: _GMFCS_STR_TO_INT.get(lvl, 0)
        for uid, lvl in zip(
            introductory_df["id"].to_list(),
            introductory_df["gmfcs_lvl"].to_list(),
        )
    }
 
 
def _extract_milestone_keys_per_age(df: pl.DataFrame) -> pl.DataFrame:
    """Extract and aggregate unique milestone keys per (introductory_id, age)."""
    per_row_keys = [
        sorted(extract_milestone_keys(g) | extract_milestone_keys(f))
        for g, f in zip(
            df["gross_motor_development"].to_list(),
            df["fine_motor_development"].to_list(),
        )
    ]
    df2 = df.with_columns(pl.Series("milestone_keys", per_row_keys))
    return (
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
 
 
def _cumulate_milestones(per_age: pl.DataFrame) -> pl.DataFrame:
    """Cumulate unique milestones per user over age."""
    def _group(group: pl.DataFrame) -> pl.DataFrame:
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
 
    return per_age.group_by("introductory_id").map_groups(_group)
 
 
# ════════════════════════════════════════════════════════════════════════════
# Normalised over a fixed "unlocked" ceiling (set value)
# ════════════════════════════════════════════════════════════════════════════
 
def motorscore_milestones_setvalue(
    df: pl.DataFrame,
    introductory_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    milestone_score = cum_unique_milestones / possible_milestones
 
    Ceiling looked up from POSSIBLE_MILESTONES_BY_AGE_GMFCS using the
    age bucket (1–4, as stored in DB) and the child's GMFCS level.
    Returns 0..1 per (introductory_id, age).
    """
    per_age = _extract_milestone_keys_per_age(df)
    cum     = _cumulate_milestones(per_age)
    gmfcs   = _gmfcs_lookup(introductory_df)
 
    possible = [
        POSSIBLE_MILESTONES_BY_AGE_GMFCS[int(age)].get(gmfcs.get(uid, 0))
        for uid, age in zip(cum["introductory_id"].to_list(), cum["age"].to_list())
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
    introductory_df: pl.DataFrame,
    upper_col: str = "motorical_impairments_upper",
    lower_col: str = "motorical_impairments_lower",
) -> pl.DataFrame:
    """
    mms_normalized = (max_score - sum_ratings) / max_score
 
    n_named looked up from N_NAMED_BY_AGE_GMFCS using the age bucket
    (1–4, as stored in DB) and the child's GMFCS level.
    Returns 0..1 per (introductory_id, age).
    """
    row_sum_ratings = [
        sum_impairments(u) + sum_impairments(lo)
        for u, lo in zip(df[upper_col].to_list(), df[lower_col].to_list())
    ]
 
    per_age = (
        df.with_columns(pl.Series("_sum_ratings", row_sum_ratings))
        .group_by(["introductory_id", "age"])
        .agg(pl.col("_sum_ratings").sum().alias("sum_ratings"))
        .sort(["introductory_id", "age"])
    )
 
    gmfcs = _gmfcs_lookup(introductory_df)
 
    n_named_list, max_score_list, mms_list, mms_norm_list = [], [], [], []
 
    for uid, age, s in zip(
        per_age["introductory_id"].to_list(),
        per_age["age"].to_list(),
        per_age["sum_ratings"].to_list(),
    ):
        gmfcs_int = gmfcs.get(uid, 0)
        n_named   = N_NAMED_BY_AGE_GMFCS[int(age)].get(gmfcs_int, 18)
        max_score = n_named * 5
        mms       = max_score - s
        mms_norm  = (mms / max_score) if max_score > 0 else None
 
        n_named_list.append(n_named)
        max_score_list.append(max_score)
        mms_list.append(float(mms))
        mms_norm_list.append(float(mms_norm) if mms_norm is not None else None)
 
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
            "sum_ratings", "n_named", "max_score", "mms", "mms_normalized",
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
def motorscore_milestones_future(
    df: pl.DataFrame,
    introductory_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    MotorScore = cum_unique_milestones / mean(cum_unique_milestones) inom (age, gmfcs_lvl)
    Värden > 1 indikerar fler milstolpar än genomsnittet för åldersgrupp + GMFCS-nivå.
    """

    # --- Steg 1–3: identiska med motorscore_milestones ---
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

    # --- Steg 4: Joina in GMFCS och normalisera mot medelvärde inom (age, gmfcs_lvl) ---
    return (
        cum.join(
            introductory_df.select([pl.col("id").alias("introductory_id"), "gmfcs_lvl"]),
            on="introductory_id",
            how="left"
        )
        .with_columns(
            pl.col("cum_unique_milestones").mean().over(["age", "gmfcs_lvl"]).alias("mean_in_age_gmfcs")
        )
        .with_columns(
            (pl.col("cum_unique_milestones") / pl.col("mean_in_age_gmfcs"))
            .cast(pl.Float64)
            .alias("milestone_score")
        )
        .select(["introductory_id", "age", "gmfcs_lvl", "cum_unique_milestones", "milestone_score"])
        .sort(["introductory_id", "age"])
    )

# impairment score normaliserat över ålder och gmfcs
def motorscore_impairments_future(
    df: pl.DataFrame,
    introductory_df: pl.DataFrame,
    upper_col: str = "motorical_impairments_upper",
    lower_col: str = "motorical_impairments_lower",
) -> pl.DataFrame:
    """
    ImpairmentScore = mms / mean(mms) inom (age, gmfcs_lvl)
    Värden > 1 indikerar lägre grad av nedsättning än genomsnittet för åldersgrupp + GMFCS-nivå.
    """

    def _n_named(age: float) -> int:
        if age < 1:   return 9
        elif age < 2: return 16
        elif age < 3: return 17
        else:         return 18

    # --- Steg 1–2: identiska med motorscore_impairments ---
    upper_list = df[upper_col].to_list()
    lower_list = df[lower_col].to_list()

    row_sum_ratings: list[float] = []
    row_other_keys: list[list[str]] = []

    for u, lo in zip(upper_list, lower_list):
        row_sum_ratings.append(sum_impairments(u) + sum_impairments(lo))
        all_keys = extract_milestone_keys(u) | extract_milestone_keys(lo)
        others = sorted(k for k in all_keys if "other" in k.lower())
        row_other_keys.append(others)

    df2 = df.with_columns([
        pl.Series("_sum_ratings", row_sum_ratings),
        pl.Series("_other_keys", row_other_keys),
    ])

    per_age = (
        df2.group_by(["introductory_id", "age"])
        .agg([
            pl.col("_sum_ratings").sum().alias("sum_ratings"),
            pl.col("_other_keys").explode().drop_nulls().unique().alias("other_keys_age"),
        ])
        .sort(["introductory_id", "age"])
    )

    ages       = per_age["age"].to_list()
    other_keys = per_age["other_keys_age"].to_list()

    n_named_list  : list[int]   = []
    n_other_list  : list[int]   = []
    n_total_list  : list[int]   = []
    max_score_list: list[int]   = []
    mms_list      : list[float] = []

    for age, o_keys, s_ratings in zip(ages, other_keys, per_age["sum_ratings"].to_list()):
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

    per_age = per_age.with_columns([
        pl.Series("n_named",   n_named_list),
        pl.Series("n_other",   n_other_list),
        pl.Series("n_total",   n_total_list),
        pl.Series("max_score", max_score_list),
        pl.Series("mms",       mms_list),
    ])

    # --- Steg 3: Joina in GMFCS och normalisera mot medelvärde inom (age, gmfcs_lvl) ---
    return (
        per_age.join(
            introductory_df.select([pl.col("id").alias("introductory_id"), "gmfcs_lvl"]),
            on="introductory_id",
            how="left"
        )
        .with_columns(
            pl.col("mms").mean().over(["age", "gmfcs_lvl"]).alias("mean_mms_age_gmfcs")
        )
        .with_columns(
            pl.when(pl.col("mean_mms_age_gmfcs") > 0)
            .then(pl.col("mms") / pl.col("mean_mms_age_gmfcs"))
            .otherwise(None)
            .cast(pl.Float64)
            .alias("mms_normalized")
        )
        .select([
            "introductory_id", "age", "gmfcs_lvl",
            "sum_ratings", "n_other", "n_named", "n_total",
            "max_score", "mms", "mean_mms_age_gmfcs", "mms_normalized",
        ])
        .sort(["introductory_id", "age"])
    )

# kombinerat score (medelvärdet av milestone och impairment-settvalue) 
def motorscore_combined(milestone_df: pl.DataFrame, impairment_df: pl.DataFrame,) -> pl.DataFrame:
    return (
        milestone_df.select(["introductory_id", "age", "milestone_score"])
        .join(
            impairment_df.select(["introductory_id", "age", "mms_normalized"]),
            on=["introductory_id", "age"],
            how="inner",
        )
        .with_columns(
            ((pl.col("milestone_score") + pl.col("mms_normalized")) / 2)
            .cast(pl.Float64)
            .alias("combined_score")
        )
        .select(["introductory_id", "age", "milestone_score", "mms_normalized", "combined_score"])
        .sort(["introductory_id", "age"])
    )

if __name__ == "__main__":
    from dataloader import load_data
    from connect_db import get_connection

    conn = get_connection()
    data = load_data(conn)

    motorical_dev = data["motorical_development"]
    introductory = data["introductory"]

    milestonevalue1 = motorscore_milestones_setvalue(motorical_dev, introductory)
    milestonesvalue2 = motorscore_milestones(motorical_dev)
    print('First milestone score')
    print(milestonevalue1)
    print('Second milestone score')
    print(milestonesvalue2)

    impairmentvalue1 = motorscore_impairments_setvalue(motorical_dev, introductory)
    impairmentvalue2 = motorscore_impairments(motorical_dev)
    print('First impairment score')
    print(impairmentvalue1)
    print('Second impairment score')
    print(impairmentvalue2)
  