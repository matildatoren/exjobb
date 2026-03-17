import polars as pl

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

def process_motorical_score_2_normalized(
    df: pl.DataFrame,
    variant: str = "multiplicative",
    lam: float = 0.5,
    w: float = 0.5,
) -> pl.DataFrame:
    """
    Score 2+ — cumulative milestone score with normalized impairment penalty.

    Normalization (Option A):
        I_hat(i, a) = [I_lower(i,a) + I_upper(i,a)] / global_max_I

    Variants (pick via `variant` argument):
        "additive"       : S2+ = max(S2 - λ * I_hat, 0)          [λ = lam]
        "multiplicative" : S2+ = S2 * (1 - I_hat)                 [recommended]
        "composite"      : S2+ = w * S2 + (1-w) * (1 - I_hat)    [w = w]
        "geometric"      : S2+ = sqrt(S2 * (1 - I_hat))

    Parameters
    ----------
    df       : motorical_development DataFrame (same shape as in other functions)
    variant  : one of "additive", "multiplicative", "composite", "geometric"
    lam      : penalty weight for "additive" variant (λ ∈ [0, 1])
    w        : milestone weight for "composite" variant (w ∈ [0, 1])

    Returns
    -------
    DataFrame with columns:
        introductory_id, age, cum_unique_milestones, s2, i_hat, motorical_score_2_norm
    """

    # ------------------------------------------------------------------
    # Step 1: compute S2 (reuse existing logic inline)
    # ------------------------------------------------------------------
    possible_milestones_by_age = {1: 12, 2: 19, 3: 25, 4: 31, 5: 36, 6: 39, 7: 41}

    gross_list = df["gross_motor_development"].to_list()
    fine_list  = df["fine_motor_development"].to_list()

    per_row_keys = []
    for g, f in zip(gross_list, fine_list):
        keys = extract_milestone_keys(g) | extract_milestone_keys(f)
        per_row_keys.append(sorted(keys))

    df2 = df.with_columns(pl.Series("milestone_keys", per_row_keys))

    per_age_milestones = (
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
            cleaned = [
                k for k in (keys or [])
                if k is not None and str(k).strip() != "" and str(k).lower() != "none"
            ]
            seen |= set(cleaned)
            cum_counts.append(len(seen))
        return group.with_columns(pl.Series("cum_unique_milestones", cum_counts))

    cum = per_age_milestones.group_by("introductory_id").map_groups(_cumulate)

    ages     = cum["age"].to_list()
    possible = [possible_milestones_by_age.get(int(a)) for a in ages]
    s2_vals  = [
        (c / p) if p and p > 0 else None
        for c, p in zip(cum["cum_unique_milestones"].to_list(), possible)
    ]

    s2_df = (
        cum
        .with_columns(pl.Series("s2", s2_vals, dtype=pl.Float64))
        .with_columns(pl.col("s2").fill_null(0.0))
    )

    # ------------------------------------------------------------------
    # Step 2: compute raw impairment sum per (introductory_id, age)
    # ------------------------------------------------------------------
    lower_list = df["motorical_impairments_lower"].to_list()
    upper_list = df["motorical_impairments_upper"].to_list()

    raw_impairments = [
        sum_impairments(l) + sum_impairments(u)
        for l, u in zip(lower_list, upper_list)
    ]

    imp_df = df.with_columns(pl.Series("raw_impairment", raw_impairments))

    imp_per_age = (
        imp_df
        .group_by(["introductory_id", "age"])
        .agg(pl.mean("raw_impairment").alias("raw_impairment"))
    )

    # ------------------------------------------------------------------
    # Step 3: Option A normalization — divide by global maximum
    # ------------------------------------------------------------------
    global_max = imp_per_age["raw_impairment"].max()

    if global_max is None or global_max == 0:
        imp_per_age = imp_per_age.with_columns(
            pl.lit(0.0).alias("i_hat")
        )
    else:
        imp_per_age = imp_per_age.with_columns(
            (pl.col("raw_impairment") / global_max)
            .clip(0.0, 1.0)
            .alias("i_hat")
        )

    # ------------------------------------------------------------------
    # Step 4: merge S2 and I_hat, then apply chosen variant
    # ------------------------------------------------------------------
    merged = (
        s2_df
        .select(["introductory_id", "age", "cum_unique_milestones", "s2"])
        .join(
            imp_per_age.select(["introductory_id", "age", "i_hat"]),
            on=["introductory_id", "age"],
            how="left",
        )
        .with_columns(pl.col("i_hat").fill_null(0.0))
    )

    s2       = merged["s2"].to_list()
    i_hat    = merged["i_hat"].to_list()
    scores: list[float] = []

    for s, i in zip(s2, i_hat):
        if variant == "additive":
            scores.append(max(s - lam * i, 0.0))
        elif variant == "multiplicative":
            scores.append(s * (1.0 - i))
        elif variant == "composite":
            scores.append(w * s + (1.0 - w) * (1.0 - i))
        elif variant == "geometric":
            scores.append((s * (1.0 - i)) ** 0.5)
        else:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                "Choose from: additive, multiplicative, composite, geometric."
            )

    return (
        merged
        .with_columns(pl.Series("motorical_score_2_norm", scores, dtype=pl.Float64))
        .select(["introductory_id", "age", "cum_unique_milestones", "s2", "i_hat", "motorical_score_2_norm"])
        .sort(["introductory_id", "age"])
    )


# ------------------------------------------------------------------
# Usage example (mirrors the __main__ block in motorical_score.py)
# ------------------------------------------------------------------

if __name__ == "__main__":
    from dataloader import load_data
    from connect_db import get_connection

    conn = get_connection()
    data = load_data(conn)
    md   = data["motorical_development"]

    # Recommended: Option A + multiplicative (no free parameters)
    score_2_norm = process_motorical_score_2_normalized(md, variant="multiplicative")
    print("\nScore 2 normalized (multiplicative):\n")
    print(score_2_norm)

    # Compare all four variants side by side
    variants = ["additive", "multiplicative", "composite", "geometric"]
    comparison = score_2_norm.select(["introductory_id", "age", "s2", "i_hat"])

    for v in variants:
        result = process_motorical_score_2_normalized(md, variant=v)
        comparison = comparison.join(
            result.select(["introductory_id", "age",
                           pl.col("motorical_score_2_norm").alias(f"s2_norm_{v}")]),
            on=["introductory_id", "age"],
            how="left",
        )

    print("\nVariant comparison:\n")
    print(comparison)