import polars as pl

def extract_neurohab_center_hours(struct):
    if struct is None:
        return []

    struct_dict = dict(struct) if isinstance(struct, dict) else {}

    rows = []

    for center_name, info in struct_dict.items():

        if not info:
            # if empty {} for this center, still register 0
            rows.append({
                "center_name": center_name,
                "total_hours": 0.0
            })
            continue

        hours = info.get("hours", 0)
        days = info.get("days", 1)
        weeks = info.get("weeks", 1)

        try:
            total_hours = (
                float(hours or 0)
                * float(days or 0)
                * float(weeks or 0)
            )
        except (TypeError, ValueError):
            total_hours = 0.0

        rows.append({
            "center_name": center_name,
            "total_hours": total_hours
        })

    return rows



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


if __name__ == "__main__":
    from dataloader import load_data
    from connect_db import get_connection

    conn = get_connection()
    data = load_data(conn)

    intensive_therapies = data["intensive_therapies"]

    neurohab_hours = process_neurohab_hours_per_user_per_age(intensive_therapies)

    print("\nTotal hours per neurohabilitation center, per user, per age:")
    print(neurohab_hours)
