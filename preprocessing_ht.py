import polars as pl

# -------------------------------------------- 
# therapies, devices and other training as separate tables
# --------------------------------------------

# ------ helper methods for extracting data ------

def extract_hometraining_hours(training):
    if training is None:
        return 0.0

    training_dict = dict(training) if isinstance(training, dict) else {}
    details = training_dict.get("details", {})
    total_hours = 0.0

    for training_info in details.values():
        if not training_info:
            continue

        hours = training_info.get("hours", 0)
        days = training_info.get("days", 1)
        weeks = training_info.get("weeks", 1)

        try:
            hours = float(hours or 0)
            days = float(days or 0)
            weeks = float(weeks or 0)

            total_hours += hours * days * weeks

        except (TypeError, ValueError):
            pass

    return total_hours

def extract_device_hours(devices):
    if devices is None:
        return 0.0

    devices_dict = dict(devices) if isinstance(devices, dict) else {}
    details = devices_dict.get("details", {})
    total_hours = 0.0

    for device_info in details.values():
        if not device_info:
            continue

        hours = device_info.get("hours", 0)
        days = device_info.get("days", 1)
        weeks = device_info.get("weeks", 1)

        try:
            hours = float(hours or 0)
            days = float(days or 0)
            weeks = float(weeks or 0)

            total_hours += hours * days * weeks

        except (TypeError, ValueError):
            pass

    return total_hours

def extract_other_training_hours(other_training):
    if other_training is None:
        return 0.0

    other_dict = dict(other_training) if isinstance(other_training, dict) else {}
    details = other_dict.get("details", {})
    total_hours = 0.0

    for training_info in details.values():
        if not training_info:
            continue

        hours = training_info.get("hours", 0)
        days = training_info.get("days", 1)
        weeks = training_info.get("weeks", 1)

        try:
            hours = float(hours or 0)
            days = float(days or 0)
            weeks = float(weeks or 0)

            total_hours += hours * days * weeks

        except (TypeError, ValueError):
            pass

    return total_hours

# ----- functions for extracting the data into tables per year and per survey ---------

def process_home_training_hours_per_user_per_year(df: pl.DataFrame) -> pl.DataFrame:
    structs = df["training_methods_therapies"].to_list()
    total_hours_list = [extract_hometraining_hours(s) for s in structs]

    df = df.with_columns(
        pl.Series("training_hours", total_hours_list)
    )

    df_grouped = (
        df.group_by(["introductory_id", "age"])
        .agg(
            pl.sum("training_hours").alias("total_home_training_hours")
        )
        .sort(["introductory_id", "age"])
    )

    return df_grouped



def process_device_hours_per_user_per_year(df: pl.DataFrame) -> pl.DataFrame:
    structs = df["devices"].to_list()
    total_hours_list = [extract_device_hours(s) for s in structs]

    df = df.with_columns(
        pl.Series("device_hours", total_hours_list)
    )

    df_grouped = (
        df.group_by(["introductory_id", "age"])
        .agg(
            pl.sum("device_hours").alias("total_device_hours")
        )
        .sort(["introductory_id", "age"])
    )

    return df_grouped

def process_other_training_hours_per_user_per_year(df: pl.DataFrame) -> pl.DataFrame:
    structs = df["other_training_methods_therapies"].to_list()
    total_hours_list = [extract_other_training_hours(s) for s in structs]

    df = df.with_columns(
        pl.Series("other_training_hours", total_hours_list)
    )

    df_grouped = (
        df.group_by(["introductory_id", "age"])
        .agg(
            pl.sum("other_training_hours").alias("total_other_training_hours")
        )
        .sort(["introductory_id", "age"])
    )

    return df_grouped

# -------------------------------------------- 
# all tables together, therapies, devices and other training 
# --------------------------------------------

#---- helper method ----
def extract_training_details(training_struct, category_name):

    if training_struct is None:
        return []

    struct_dict = dict(training_struct) if isinstance(training_struct, dict) else {}
    details = struct_dict.get("details", {})

    rows = []

    for training_name, info in details.items():
        if not info:
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

            rows.append({
                "training_category": category_name,
                "training_name": training_name,
                "total_hours": total_hours
            })

        except (TypeError, ValueError):
            continue

    return rows

#---- extracting everything by type ----
def process_training_per_type_per_year(df: pl.DataFrame) -> pl.DataFrame:
    all_rows = []

    for row in df.iter_rows(named=True):

        intro_id = row["introductory_id"]
        age = row["age"]

        home_rows = extract_training_details(
            row.get("training_methods_therapies"),
            "home"
        )

        device_rows = extract_training_details(
            row.get("devices"),
            "devices"
        )

        other_rows = extract_training_details(
            row.get("other_training_methods_therapies"),
            "other"
        )

        for r in home_rows + device_rows + other_rows:
            r["introductory_id"] = intro_id
            r["age"] = age
            all_rows.append(r)

    if not all_rows:
        return pl.DataFrame()

    result = pl.DataFrame(all_rows)

    result = (
        result
        .group_by([
            "introductory_id",
            "age",
            "training_category",
            "training_name"
        ])
        .agg(
            pl.sum("total_hours").alias("total_hours")
        )
    )


    all_training_types = result.select(
        ["training_category", "training_name"]
    ).unique()

    child_years = df.select(
        ["introductory_id", "age"]
    ).unique()

    full_panel = child_years.join(
        all_training_types,
        how="cross"
    )

    result = (
        full_panel
        .join(
            result,
            on=[
                "introductory_id",
                "age",
                "training_category",
                "training_name"
            ],
            how="left"
        )
        .with_columns(
            pl.col("total_hours").fill_null(0)
        )
        .sort([
            "introductory_id",
            "age",
            "training_category",
            "training_name"
        ])
    )

    return result



if __name__ == "__main__":
    from dataloader import load_data
    from connect_db import get_connection

    conn = get_connection()
    data = load_data(conn)

    home_training = data["home_training"]

    # ---- Home training hours ----
    processed_home = process_home_training_hours_per_user_per_year(home_training)
    print("\nTotal home training hours per user per year:")
    print(processed_home)

    # ---- Device hours ----
    processed_devices = process_device_hours_per_user_per_year(home_training)
    print("\nTotal device hours per user per age:")
    print(processed_devices)

    # ---- Other training hours ----
    processed_other = process_other_training_hours_per_user_per_year(home_training)
    print("\nTotal OTHER training hours per user per age:")
    print(processed_other)

    # ----- Extracting everything by type ---
    detailed_training = process_training_per_type_per_year(home_training)
    print("\nTraining per child, per year, per exact type:")
    print(detailed_training)