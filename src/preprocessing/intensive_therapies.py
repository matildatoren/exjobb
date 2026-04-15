import polars as pl

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════
def extract_neurohab_center_hours(struct):
    if struct is None:
        return []
    rows = []
    for center_name, info in (struct if isinstance(struct, dict) else {}).items():
        try:
            hours = float((info or {}).get("hours") or 0)
            weeks = float((info or {}).get("weeks") or 0)
            total_hours = hours * weeks   # hours/week × weeks/year
        except (TypeError, ValueError):
            total_hours = 0.0
        rows.append({"center_name": center_name, "total_hours": total_hours})
    return rows

# ════════════════════════════════════════════════════════════════════════════
# Hours and treatments per user per year
# ════════════════════════════════════════════════════════════════════════════
def process_neurohab_hours_per_user_per_age(df: pl.DataFrame) -> pl.DataFrame:

    all_rows = []

    for row in df.iter_rows(named=True):

        intro_id = row["introductory_id"]
        age = row["age"]

        centers_rows = extract_neurohab_center_hours(
            row.get("neurohabilitation_centers")
        )

        for r in centers_rows:
            r["introductory_id"] = intro_id
            r["age"] = age
            all_rows.append(r)

    if not all_rows:
        return pl.DataFrame()

    result = pl.DataFrame(all_rows)

    result = (
        result
        .group_by(["introductory_id", "age", "center_name"])
        .agg(pl.sum("total_hours").alias("total_hours"))
        .sort(["introductory_id", "age", "center_name"])
    )

    return result

def process_medical_treatments_per_user_per_age(df: pl.DataFrame) -> pl.DataFrame:

    all_rows = []

    for row in df.iter_rows(named=True):
        intro_id = row["introductory_id"]
        age = row["age"]
        treatments = row.get("medical_treatments")

        if treatments is None:
            continue

        if isinstance(treatments, list):
            treatment_list = treatments
        elif isinstance(treatments, dict):
            treatment_list = treatments.get("details", [])
        else:
            continue

        for treatment_name in treatment_list:
            if not treatment_name:
                continue
            all_rows.append({
                "introductory_id": intro_id,
                "age": age,
                "treatment_name": str(treatment_name),
            })

    if not all_rows:
        return pl.DataFrame(schema={"introductory_id": pl.Utf8, "age": pl.Int64})

    result = pl.DataFrame(all_rows)

    result = (
        result
        .with_columns(pl.lit(1).alias("received"))
        .pivot(
            values="received",
            index=["introductory_id", "age"],
            on="treatment_name",
            aggregate_function="sum"
        )
        .fill_null(0)
        .sort(["introductory_id", "age"])
    )

    return result



if __name__ == "__main__":
    from dataloader import load_data
    from connect_db import get_connection

    conn = get_connection()
    data = load_data(conn)

    intensive_therapies = data["intensive_therapies"]

    neurohab_hours = process_neurohab_hours_per_user_per_age(intensive_therapies)
    medical_df = process_medical_treatments_per_user_per_age(intensive_therapies)

    print("\nTotal hours per neurohabilitation center, per user, per age:")
    print(neurohab_hours)

    print("\nMedical treatments detected:")
    print(medical_df.columns)

    print(f"\nShape: {medical_df.shape}")
    print("\nMedical treatments table:")
    print(medical_df)