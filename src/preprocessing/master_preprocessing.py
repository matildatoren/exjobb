from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

# ── resolve imports regardless of where the script is called from ──────────
ROOT = Path(__file__).resolve().parents[1]   # src/
sys.path.append(str(ROOT))

from preprocessing.home_training import (
    process_home_training_hours_per_user_per_year,
    process_other_training_hours_per_user_per_year,
)
from preprocessing.intensive_therapies import (
    process_neurohab_hours_per_user_per_age,
    process_medical_treatments_per_user_per_age,
)
from preprocessing.motor_development import (
    process_motorical_score_1,
)
from preprocessing.motor_scores import (
    motorscore_milestones_setvalue,
    motorscore_impairments_setvalue,
    motorscore_milestones,
    motorscore_impairments,
    motorscore_combined,
    motorscore_impairments_presence_severity,

)
from preprocessing.training_categories import (
    build_category_hours_table,
    CATEGORY_COLS,
)


# ════════════════════════════════════════════════════════════════════════════
# GMFCS helpers
# ════════════════════════════════════════════════════════════════════════════

_GMFCS_MAP: dict[str, int] = {
    "Level I – Walks without limitations": 1,
    "Level II – Walks with some limitations": 2,
    "Level III – Walks with assistive devices": 3,
    "Level IV – Limited mobility, primarily uses a wheelchair": 4,
    "Level V – Severe limitations, needs full assistance for mobility": 5,
    "Not sure / Don't know": 3,  # default to middle of scale
}


def _encode_gmfcs(introductory_df: pl.DataFrame) -> pl.DataFrame:
    mapping_series = pl.Series(
        name="gmfcs_int",
        values=[_GMFCS_MAP.get(v) for v in introductory_df["gmfcs_lvl"].to_list()],
        dtype=pl.Int32,
    )
    return (
        introductory_df
        .select([pl.col("id").alias("introductory_id")])
        .with_columns(mapping_series)
    )


# ════════════════════════════════════════════════════════════════════════════
# Device binary table
# ════════════════════════════════════════════════════════════════════════════

def _build_device_binary(home_training_df: pl.DataFrame) -> pl.DataFrame:
    """
    Pivot device usage into binary columns (prefix "device_").

    Returns a wide table: one row per (introductory_id, age),
    one column per device name  →  1 if used that year, else 0.
    Also adds 'has_any_device'.
    """
    rows: list[dict] = []

    for row in home_training_df.iter_rows(named=True):
        intro_id = row["introductory_id"]
        age = row["age"]
        devices_raw = row.get("devices") or {}

        details = {}
        if isinstance(devices_raw, dict):

            device_names = devices_raw.get("devices", []) or []
            for device_name in device_names:
                if device_name and device_name != "None":
                    rows.append({
                        "introductory_id": intro_id,
                        "age": age,
                        "device_name": str(device_name),
                    })

    if not rows:
        return pl.DataFrame(
            schema={
                "introductory_id": pl.Utf8,
                "age": pl.Int64,
                "has_any_device": pl.Int32,
            }
        )

    long = pl.DataFrame(rows)

    wide = (
        long.with_columns(pl.lit(1).alias("used"))
        .pivot(
            values="used",
            index=["introductory_id", "age"],
            on="device_name",
            aggregate_function="sum",
        )
        .fill_null(0)
    )

    # Rename columns to prefix "device_"
    device_cols = [c for c in wide.columns if c not in ("introductory_id", "age")]
    wide = wide.rename({c: f"device_{c}" for c in device_cols})

    # Aggregate: has any device that year?
    device_cols_renamed = [
        c for c in wide.columns
        if c.startswith("device_") and c.lower() != "device_none"
    ]
    wide = wide.with_columns(
        pl.when(
            pl.fold(
                acc=pl.lit(0),
                function=lambda acc, x: acc + x,
                exprs=[pl.col(c) for c in device_cols_renamed],
            )
            > 0
        )
        .then(1)
        .otherwise(0)
        .cast(pl.Int32)
        .alias("has_any_device")
    )

    return wide.sort(["introductory_id", "age"])


# ════════════════════════════════════════════════════════════════════════════
# Medical treatment helpers
# ════════════════════════════════════════════════════════════════════════════

def _prefix_medical_cols(medical_df: pl.DataFrame) -> pl.DataFrame:
    """
    Rename treatment columns to prefix "med_" and add has_any_medical_treatment.
    """
    treatment_cols = [
        c for c in medical_df.columns if c not in ("introductory_id", "age")
    ]

    df = medical_df.rename({c: f"med_{c}" for c in treatment_cols})

    med_cols = [
        c for c in df.columns
        if c.startswith("med_") and c.lower() not in ("med_no", "med_none", "med_nej")
    ]
    df = df.with_columns(
        pl.when(
            pl.fold(
                acc=pl.lit(0),
                function=lambda acc, x: acc + x,
                exprs=[pl.col(c) for c in med_cols],
            )
            > 0
        )
        .then(1)
        .otherwise(0)
        .cast(pl.Int32)
        .alias("has_any_medical_treatment")
    )

    return df


# ════════════════════════════════════════════════════════════════════════════
# Motor score assembly
# ════════════════════════════════════════════════════════════════════════════
 
def _build_motor_table(motorical_dev: pl.DataFrame, introductory: pl.DataFrame) -> pl.DataFrame:

    # ── score 1 ─────────────────────────────────────────────────────────────
    score1 = process_motorical_score_1(motorical_dev)

    # ── milestone scores ────────────────────────────────────────────────────
    ms_set = (
        motorscore_milestones_setvalue(motorical_dev, introductory)
        .select([
            "introductory_id", "age",
            pl.col("milestone_score").alias("milestone_score_setvalue"),
        ])
    )
 
    ms_norm = (
        motorscore_milestones(motorical_dev)
        .select([
            "introductory_id", "age",
            pl.col("milestone_score").alias("milestone_score"),
        ])
    )
 
    # ── impairment scores ───────────────────────────────────────────────────
    imp_set = (
        motorscore_impairments_setvalue(motorical_dev, introductory)
        .select([
            "introductory_id", "age",
            pl.col("mms_normalized").alias("impairment_score_setvalue"),
        ])
    )
 
    imp_norm = (
        motorscore_impairments(motorical_dev)
        .select([
            "introductory_id", "age",
            pl.col("mms_normalized").alias("impairment_score"),
        ])
    )

    ps = (
    motorscore_impairments_presence_severity(motorical_dev, introductory)
    .select([
        "introductory_id", "age",
        "presence_score",
        "severity_score",
    ])
    )
 
    # ── join all scores ─────────────────────────────────────────────────────
    motor = (
        ms_set
        .join(ms_norm,  on=["introductory_id", "age"], how="left")
        .join(imp_set,  on=["introductory_id", "age"], how="left")
        .join(imp_norm, on=["introductory_id", "age"], how="left")
        .join(score1,    on=["introductory_id", "age"], how="left")
        .join(ps, on=["introductory_id", "age"], how="left")
        .sort(["introductory_id", "age"])
    )

    # ── combined score via motorscore_combined ───────────────────────────────
    combined_set = motorscore_combined(ms_set.rename({"milestone_score_setvalue": "milestone_score"}),
                                    imp_set.rename({"impairment_score_setvalue": "mms_normalized"}))
    combined_norm = motorscore_combined(
        ms_norm,
        imp_norm.rename({"impairment_score": "mms_normalized"}),
    )

    motor = (
        motor
        .join(combined_set.select(["introductory_id", "age",
                                pl.col("combined_score").alias("combined_score_setvalue")]),
            on=["introductory_id", "age"], how="left")
        .join(combined_norm.select(["introductory_id", "age", "combined_score"]),
            on=["introductory_id", "age"], how="left")
    )
 
    # ── delta for every motor score ─────────────────────────────────────────
    score_cols = [
        "milestone_score_setvalue",
        "milestone_score",
        "impairment_score_setvalue",
        "impairment_score",
        "combined_score_setvalue",
        "combined_score",
        "motorical_score", 
        "presence_score",
        "severity_score", 
    ]

    motor = motor.with_columns([
        (
            pl.col(c) - pl.col(c).shift(1).over("introductory_id")
        ).alias(f"delta_{c}")
        for c in score_cols
    ])
    
    return motor
 
# ════════════════════════════════════════════════════════════════════════════
# Master builder
# ════════════════════════════════════════════════════════════════════════════

def build_master_feature_table(data: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """
    Join all feature tables into a single wide master table.

    Returns one row per (introductory_id, age) with motor scores, training hours,
    device usage, medical treatments, therapy category hours, and log-transformed
    hour features.
    """
    home_training      = data["home_training"]
    intensive_therapies = data["intensive_therapies"]
    motorical_dev      = data["motorical_development"]
    introductory       = data["introductory"]

    # ── 1. GMFCS static features ────────────────────────────────────────────
    gmfcs_df = _encode_gmfcs(introductory)

    # ── 2. Home-training hours (aggregated) ─────────────────────────────────
    home_hours_df = process_home_training_hours_per_user_per_year(home_training)
    other_hours_df = process_other_training_hours_per_user_per_year(home_training)

    # ── 3. Device binary table ──────────────────────────────────────────────
    device_binary_df = _build_device_binary(home_training)

    # ── 4. Intensive therapies ──────────────────────────────────────────────
    neurohab_df = (
        process_neurohab_hours_per_user_per_age(intensive_therapies)
        .group_by(["introductory_id", "age"])
        .agg(pl.sum("total_hours").alias("total_hours"))
        .sort(["introductory_id", "age"]))
    medical_df = process_medical_treatments_per_user_per_age(intensive_therapies)
    medical_df = _prefix_medical_cols(medical_df)


    # ── 5. Therapy category hours ───────────────────────────────────────────
    category_hours_df = build_category_hours_table(
        home_training,
        intensive_therapies,
    )

    # ── 6. active_total_hours (home + other + neurohab) ─────────────────────
    active_df = (
        home_hours_df.rename({"total_home_training_hours": "_home_h"})
        .join(
            other_hours_df.rename({"total_other_training_hours": "_other_h"}),
            on=["introductory_id", "age"], how="full", coalesce=True,
        )
        .join(
            neurohab_df.rename({"total_hours": "_neuro_h"}),
            on=["introductory_id", "age"], how="full", coalesce=True,
        )
        .fill_null(0)
        .with_columns(
            (pl.col("_home_h") + pl.col("_other_h") + pl.col("_neuro_h"))
            .alias("active_total_hours")
        )
        .select(["introductory_id", "age", "active_total_hours"])
    )
 

    # ── 7. All motor scores + delta / lag ────────────────────────────────────
    motor_df = _build_motor_table(motorical_dev, introductory)

    # ── 8. Join everything ───────────────────────────────────────────────────
    result = (
        motor_df
        .join(gmfcs_df,       on="introductory_id",           how="left")
        .join(home_hours_df,  on=["introductory_id", "age"],  how="left")
        .join(other_hours_df, on=["introductory_id", "age"],  how="left")
        .join(active_df,      on=["introductory_id", "age"],  how="left")
        .join(
            neurohab_df.rename({"total_hours": "neurohab_hours"}),
            on=["introductory_id", "age"], how="left",
        )
        .join(device_binary_df, on=["introductory_id", "age"], how="left")
        .join(medical_df,       on=["introductory_id", "age"], how="left")
        .join(category_hours_df, on=["introductory_id", "age"], how="left")
        # ── log1p-transformed hours ───────────────────────────────────────────
        .with_columns([
            (pl.col("total_home_training_hours") + 1).log(base=2.718281828).alias("log_total_home_training_hours"),
            (pl.col("total_other_training_hours") + 1).log(base=2.718281828).alias("log_total_other_training_hours"),
            (pl.col("active_total_hours") + 1).log(base=2.718281828).alias("log_active_total_hours"),
            (pl.col("neurohab_hours") + 1).log(base=2.718281828).alias("log_neurohab_hours"),
            # ── log1p for therapy categories (unclassified excluded) ──────────
            *[
                (pl.col(c) + 1).log(base=2.718281828).alias(f"log_{c}")
                for c in CATEGORY_COLS
                if c != "cat_unclassified"
            ],
        ])
        .sort(["introductory_id", "age"])
    )

    # ── Fill only training/binary cols with 0, leave scores as None ─────────
    _SCORE_PREFIXES = (
        "milestone_score", "impairment_score", "combined_score",
        "presence_score", "severity_score",
        "mms", "delta_", "motorical_score", "gmfcs_int",
    )
    fill_cols = [
        c for c in result.columns
        if not any(c.startswith(p) for p in _SCORE_PREFIXES)
        and result[c].dtype in (pl.Float64, pl.Int64, pl.Int32, pl.Float32)
    ]
    result = result.with_columns([
        pl.col(c).fill_null(0) for c in fill_cols
    ])
    return result


# ════════════════════════════════════════════════════════════════════════════
# Convenience accessor
# ════════════════════════════════════════════════════════════════════════════

def get_feature_groups(master_df: pl.DataFrame) -> dict[str, list[str]]:
    """
    Return a dictionary mapping feature-group names to their column names.
    Useful for selecting subsets in modelling code.
 
    Example
    -------
    >>> groups = get_feature_groups(master_df)
    >>> X = master_df.select(groups["training_hours"] + groups["medical"])
    """
    cols = set(master_df.columns)
 
    return {
        "id": ["introductory_id", "age"],
        "gmfcs": [c for c in ["gmfcs_int"] if c in cols],
        "training_hours": [c for c in [
            "total_home_training_hours",
            "total_other_training_hours",
            "neurohab_hours",
            "active_total_hours",
            "log_total_home_training_hours",
            "log_total_other_training_hours",
            "log_active_total_hours",
            "log_neurohab_hours",
        ] if c in cols],
        "devices": [c for c in master_df.columns
                    if c.startswith("device_") or c == "has_any_device"],
        "medical": [c for c in master_df.columns
                    if c.startswith("med_") or c == "has_any_medical_treatment"],
        "motor_milestones": [c for c in [
            "milestone_score_setvalue",
            "milestone_score",
        ] if c in cols],
        "motor_impairments": [c for c in [
            "impairment_score_setvalue",
            "impairment_score",
        ] if c in cols],
        "targets": [c for c in [
            "delta_milestone_score_setvalue",
            "delta_milestone_score",
            "delta_impairment_score_setvalue",
            "delta_impairment_score",
            "combined_score_setvalue",
            "combined_score",
        ] if c in cols],
        "therapy_categories": [c for c in CATEGORY_COLS if c in cols],
        "motor_score_1": [c for c in ["motorical_score", "delta_motorical_score"] if c in cols],
        "log_therapy_categories": [f"log_{c}" for c in CATEGORY_COLS
                            if f"log_{c}" in cols],
    }
 

# ════════════════════════════════════════════════════════════════════════════
# Quick sanity-check
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from dataloader import load_data
    from connect_db import get_connection

    conn = get_connection()
    data = load_data(conn)

    master = build_master_feature_table(data)

    groups = get_feature_groups(master)

    print("=" * 60)
    print(f"Master feature table  →  {master.shape[0]} rows × {master.shape[1]} columns")
    print("=" * 60)

    for group_name, group_cols in groups.items():
        existing = [c for c in group_cols if c in master.columns]
        print(f"\n[{group_name}]  ({len(existing)} columns)")
        for c in existing:
            print(f"    {c}")

    print("\nSample (first 5 rows):")
    print(master.head(5))

    print("\n" + "=" * 60)
    print("ALL COLUMNS")
    print("=" * 60)
    for i, col in enumerate(master.columns, 1):
        print(f"  {i:>3}. {col}")