import os
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from dotenv import load_dotenv

load_dotenv()

DB_DSN = os.getenv("DB_URL")

def get_connection():
    return psycopg2.connect(DB_DSN)

# ─── Excluded introductory_ids ────────────────────────────────────────────────
# These were created after the format change and are already in new format

EXCLUDED_INTRODUCTORY_IDS = {
    "83de457e-7e32-4fb9-a8ed-d52a5f4fd80c",
    "19d2295a-bf58-499e-a91c-d88a2cf52e93",
    "f9231c8d-2ade-4c0e-a878-a9524ccc3d65",
    "0a584ba1-cdf4-4251-9168-5f8ccc0240e3"
}

# ─── Safe value helpers ───────────────────────────────────────────────────────

def safe_float(val) -> float:
    try:
        return float(val) if val != "" and val is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def is_empty_value(val) -> bool:
    if val is None:
        return True
    if val == "" or val == [] or val == {}:
        return True
    return False


# ─── Table config ─────────────────────────────────────────────────────────────
# Only the JSONB fields that contain days/hours/weeks structs

TABLE_CONFIG = {
    "home_training": {
        "jsonb_fields_with_details": ["training_methods_therapies", "other_training_methods_therapies"],
        "jsonb_fields_flat": [],
    },
    "intensive_therapies": {
        "jsonb_fields_with_details": [],
        "jsonb_fields_flat": ["neurohabilitation_centers", "methods_applied_during_intense_training"],
    }
}

# ─── Format conversion ────────────────────────────────────────────────────────

def is_already_new_format(jdata: dict, is_flat: bool) -> bool:
    """Check if a JSONB field is already in new format (days = "")."""
    details = jdata if is_flat else jdata.get("details", {})
    if is_empty_value(details):
        return True  # empty — nothing to convert
    for vals in details.values():
        if isinstance(vals, dict):
            if vals.get("days") == "":
                return True  # new format
            if isinstance(vals.get("days"), (int, float)):
                return False  # old format
    return True  # no recognizable values — leave as-is


def convert_details(details: dict) -> dict:
    """Convert a details dict from old format to new format."""
    new_details = {}
    for method, vals in details.items():
        if not isinstance(vals, dict):
            new_details[method] = vals
            continue

        days_raw  = vals.get("days")
        hours_num = safe_float(vals.get("hours"))
        weeks_num = safe_float(vals.get("weeks"))

        if days_raw == "" or days_raw is None:
            # Already new format — keep as-is
            new_details[method] = vals
        else:
            # Old format: days/week × minutes/day ÷ 60 = hours/week
            days_num = safe_float(days_raw)
            hpw = round((days_num * hours_num) / 60, 2)
            new_details[method] = {"days": "", "hours": hpw, "weeks": weeks_num}

    return new_details


def convert_row(row: dict, config: dict) -> dict:
    """
    Returns a dict of {field: new_jsonb_value} for all relevant fields.
    Each method is handled individually — already-new-format methods kept as-is,
    old-format methods converted.
    """
    result = {}

    # ── Fields with details-wrapper ──
    for field in config["jsonb_fields_with_details"]:
        jdata = row.get(field)
        if is_empty_value(jdata) or "details" not in jdata:
            continue
        if is_empty_value(jdata.get("details")):
            continue
        new_details = convert_details(jdata["details"])
        result[field] = {
            "other": jdata.get("other", ""),
            "details": new_details,
            "methods": list(new_details.keys())
        }

    # ── Flat fields ──
    for field in config["jsonb_fields_flat"]:
        jdata = row.get(field)
        if is_empty_value(jdata) or not isinstance(jdata, dict):
            continue
        result[field] = convert_details(jdata)

    return result


# ─── Safety check ─────────────────────────────────────────────────────────────

def check_old_tables_exist(conn):
    with conn.cursor() as cur:
        for table in ["home_training_old", "intensive_therapies_old"]:
            cur.execute(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)",
                (table,)
            )
            if not cur.fetchone()[0]:
                raise Exception(
                    f"\n❌ Säkerhetstabell '{table}' saknas!\n"
                    f"   Kör SQL-stegen för att spara age 1+2 till _old-tabeller innan du kör detta skript:\n"
                    f"   INSERT INTO {table} SELECT * FROM {table.replace('_old', '')} WHERE age IN (1, 2);"
                )


# ─── Core migration ───────────────────────────────────────────────────────────

def convert_table(conn, table_name: str, dry_run: bool = True):
    config = TABLE_CONFIG[table_name]

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            f"SELECT * FROM {table_name} WHERE age IN (1, 2) ORDER BY age"
        )
        rows = cur.fetchall()

        print(f"\n{'='*60}")
        print(f"Table : {table_name} (age 1+2 format conversion)")
        print(f"Rows found: {len(rows)}")
        print(f"Mode  : {'DRY RUN – no changes written' if dry_run else '⚡ LIVE – writing to DB'}")
        print("="*60)

        stats = {"converted": 0, "excluded": 0}

        for row in rows:
            intro_id = str(row["introductory_id"])
            row_id   = str(row["id"])

            # Skip excluded ids
            if intro_id in EXCLUDED_INTRODUCTORY_IDS:
                print(f"\n  id={row_id} age={row['age']} intro={intro_id} — [SKIP] exkluderad")
                stats["excluded"] += 1
                continue

            converted = convert_row(row, config)

            if not converted:
                print(f"\n  id={row_id} age={row['age']} intro={intro_id} — [SKIP] inga JSONB-fält att konvertera")
                continue

            if dry_run:
                print(f"\n  id={row_id} age={row['age']} intro={intro_id}")
                for field, new_val in converted.items():
                    details = new_val.get("details", new_val)
                    print(f"  [DRY] {field}:")
                    for method, vals in details.items():
                        if isinstance(vals, dict):
                            print(f"    · {method}: {vals.get('hours')} hrs/week, {vals.get('weeks')} weeks/year")
            else:
                set_clauses = []
                values      = []
                for field, val in converted.items():
                    set_clauses.append(f'"{field}" = %s')
                    values.append(Json(val))
                values.append(row_id)
                cur.execute(
                    f'UPDATE {table_name} SET {", ".join(set_clauses)} WHERE id = %s',
                    values
                )
                stats["converted"] += 1

        print(f"\n  Stats: {stats}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    conn = get_connection()
    try:
        check_old_tables_exist(conn)

        for table in ["home_training", "intensive_therapies"]:
            convert_table(conn, table, dry_run=True)

        print("\n" + "="*60)
        confirm = input("Apply all changes to the database? (yes/no): ").strip().lower()

        if confirm == "yes":
            for table in ["home_training", "intensive_therapies"]:
                convert_table(conn, table, dry_run=False)
            conn.commit()
            print("\n✅ All changes committed successfully.")
        else:
            print("\n❌ Aborted — no changes written.")

    except Exception as e:
        conn.rollback()
        print(f"\n❌ Error — rolled back. Details: {e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    main()