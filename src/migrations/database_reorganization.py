import os
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from dotenv import load_dotenv

load_dotenv()

DB_DSN = os.getenv("DB_URL")

# ─── Excluded introductory_ids ────────────────────────────────────────────────
# These will be completely skipped by the migration script

EXCLUDED_INTRODUCTORY_IDS = {
    "83de457e-7e32-4fb9-a8ed-d52a5f4fd80c",
}

def get_connection():
    return psycopg2.connect(DB_DSN)

# ─── Safe value helpers ───────────────────────────────────────────────────────

def safe_float(val) -> float:
    """Converts any value to float safely — returns 0.0 for empty strings, None, or invalid."""
    try:
        return float(val) if val != "" and val is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def is_empty_value(val) -> bool:
    """Returns True if a value should be considered empty/absent."""
    if val is None:
        return True
    if val == "" or val == [] or val == {}:
        return True
    return False


def has_meaningful_data(details: dict) -> bool:
    """
    Returns True if a details dict has at least one method with
    non-zero hours or days — i.e. is worth including in averaging.
    """
    for vals in details.values():
        if safe_float(vals.get("days")) > 0 or safe_float(vals.get("hours")) > 0:
            return True
    return False


# ─── JSONB helpers ────────────────────────────────────────────────────────────

def normalize_details(details: dict) -> dict:
    """
    Converts old format (days/week, hours=minutes/day, weeks/year)
    → normalized {method: {hours_per_week, weeks}}

    Format detection:
    - New format: days = "" → hours is already hours/week
    - Old format: days = any number (including 0) → calculate days × hours / 60
    """
    result = {}
    for method, vals in details.items():
        days_raw  = vals.get("days")
        hours_num = safe_float(vals.get("hours"))
        weeks_num = safe_float(vals.get("weeks"))

        if days_raw == "" or days_raw is None:
            # New format: hours is already hours/week
            hpw = hours_num
        else:
            # Old format: days/week × minutes/day ÷ 60 = hours/week
            days_num = safe_float(days_raw)
            hpw = round((days_num * hours_num) / 60, 2)

        result[method] = {
            "hours_per_week": hpw,
            "weeks": weeks_num
        }
    return result


def average_normalized_groups(norm_list: list) -> dict:
    """
    For each method: average over all rows that contain it.
    Methods unique to one row are kept as-is.
    """
    all_methods = set()
    for n in norm_list:
        all_methods.update(n.keys())

    averaged = {}
    for method in all_methods:
        rows_with = [n[method] for n in norm_list if method in n]
        avg_hpw   = round(sum(r["hours_per_week"] for r in rows_with) / len(rows_with), 2)
        avg_weeks = round(sum(r["weeks"]          for r in rows_with) / len(rows_with), 1)
        averaged[method] = {"hours_per_week": avg_hpw, "weeks": avg_weeks}

    return averaged


def build_new_jsonb_with_details(averaged: dict, original_jsons: list) -> dict:
    """
    Builds new JSONB for fields using details-wrapper format.
    { other, details: { method: {days, hours, weeks} }, methods: [...] }
    """
    other = next((j.get("other", "") for j in original_jsons if j.get("other")), "")
    details = {
        method: {"days": "", "hours": vals["hours_per_week"], "weeks": vals["weeks"]}
        for method, vals in averaged.items()
    }
    return {
        "other": other,
        "details": details,
        "methods": list(averaged.keys())
    }


def build_new_jsonb_flat(averaged: dict) -> dict:
    """
    Builds new JSONB for flat format (no details-wrapper).
    { method_name: {days, hours, weeks} }
    """
    return {
        method: {"days": "", "hours": vals["hours_per_week"], "weeks": vals["weeks"]}
        for method, vals in averaged.items()
    }


def merge_devices(rows: list) -> dict:
    """
    Collects all unique device names and 'other' text across rows.
    days/hours/weeks are intentionally not saved.
    For duplicate device names, earliest row (lowest age) wins.
    'other' texts from all rows are concatenated with a separator.
    """
    merged_devices = []  # ordered list of unique device names
    seen_devices   = set()
    merged_others  = []

    for row in rows:
        jdata = row.get("devices")
        if is_empty_value(jdata):
            continue

        # Collect 'other' text
        other_text = jdata.get("other", "")
        if other_text and other_text not in merged_others:
            merged_others.append(other_text)

        # Collect device names — from details keys if available, else from devices array
        details = jdata.get("details", {})
        if not is_empty_value(details):
            names = list(details.keys())
        else:
            names = jdata.get("devices", [])

        for device in names:
            if device and device != "None" and device not in seen_devices:
                seen_devices.add(device)
                merged_devices.append(device)

    if not merged_devices and not merged_others:
        return None

    return {
        "other": " | ".join(merged_others) if merged_others else "",
        "details": {},
        "devices": merged_devices
    }


def merge_yes_no(rows: list, field: str) -> str:
    """
    Returns "yes" if ANY row has "yes" (case-insensitive).
    Returns "no" only if ALL non-empty rows have "no".
    Returns None if all rows are empty/missing.
    """
    values = [
        row.get(field, "").strip().lower()
        for row in rows
        if not is_empty_value(row.get(field))
    ]
    if not values:
        return None
    return "Yes" if any(v == "yes" for v in values) else "No"


def merge_array_field(rows: list, field: str) -> list:
    """
    Combines all unique strings from a TEXT[] field across rows.
    Preserves order of first appearance.
    FIX: field name is now passed explicitly — no hardcoding.
    """
    seen   = set()
    result = []
    for row in rows:
        items = row.get(field)
        if is_empty_value(items):
            continue
        if isinstance(items, list):
            for t in items:
                if t and t not in seen:
                    seen.add(t)
                    result.append(t)
    return result if result else None


def concat_text_fields(rows: list, field: str) -> str:
    """
    Concatenates non-empty text values from all rows in age order.
    Rows are assumed to already be sorted by age ascending.
    """
    parts = []
    for row in rows:
        val = row.get(field)
        if not is_empty_value(val):
            parts.append(val.strip())
    return "\n\n".join(parts) if parts else None


def merge_milestones_jsonb(rows: list, field: str) -> dict:
    """
    Collects all unique milestones from a { milestones: [...] } JSONB field.
    Preserves order of first appearance.
    """
    seen   = set()
    result = []
    for row in rows:
        jdata = row.get(field)
        if is_empty_value(jdata) or "milestones" not in jdata:
            continue
        for milestone in jdata["milestones"]:
            if milestone and milestone not in seen:
                seen.add(milestone)
                result.append(milestone)
    return {"milestones": result} if result else None


def latest_row_value(rows: list, field: str):
    """
    Returns the value of field from the row with the highest age.
    Used for impairment fields where we want the most recent assessment.
    """
    sorted_rows = sorted(rows, key=lambda r: r["age"], reverse=True)
    for row in sorted_rows:
        val = row.get(field)
        if not is_empty_value(val):
            return val
    return None


# ─── Table config ─────────────────────────────────────────────────────────────
# jsonb_fields_with_details → { details: { method: {...} }, methods: [], other: "" }
# jsonb_fields_flat         → { method_name: {days, hours, weeks} }
# jsonb_fields_devices      → special union logic, original values kept
# array_fields              → TEXT[], union of unique strings
# yes_no_fields             → "yes" if ANY is yes, "no" only if ALL are no
# concat_text_fields        → all non-empty values joined with double newline, in age order
# extra_text_fields         → first non-empty value wins

TABLE_CONFIG = {
    "home_training": {
        "jsonb_fields_with_details": ["training_methods_therapies", "other_training_methods_therapies"],
        "jsonb_fields_flat": [],
        "jsonb_fields_devices": True,
        "array_fields": [],
        "yes_no_fields": [],
        "jsonb_milestones_fields": [],
        "jsonb_latest_fields": [],
        "concat_text_fields": ["story"],
        "extra_text_fields": ["reflection"]
    },
    "intensive_therapies": {
        "jsonb_fields_with_details": [],
        "jsonb_fields_flat": ["neurohabilitation_centers", "methods_applied_during_intense_training"],
        "jsonb_fields_devices": False,
        "array_fields": ["medical_treatments"],
        "yes_no_fields": ["participate_therapies_neurohabilitation"],
        "jsonb_milestones_fields": [],
        "jsonb_latest_fields": [],
        "concat_text_fields": ["story"],
        "extra_text_fields": []
    },
    "motorical_development": {
        "jsonb_fields_with_details": [],
        "jsonb_fields_flat": [],
        "jsonb_fields_devices": False,
        "array_fields": [],
        "yes_no_fields": [],
        "jsonb_milestones_fields": ["gross_motor_development", "fine_motor_development"],
        "jsonb_latest_fields": ["motorical_impairments_lower", "motorical_impairments_upper"],
        "concat_text_fields": ["story"],
        "extra_text_fields": []
    }
}


# ─── Row merging ──────────────────────────────────────────────────────────────

def merge_group(rows: list, config: dict) -> dict:
    """
    Merges a list of same-bucket rows into one dict.
    - JSONB with details-wrapper: normalize + average
    - JSONB flat: normalize + average
    - devices: union of all unique devices, original values kept
    - TEXT[]: union of all unique strings
    - yes/no: yes if ANY row is yes
    - Extra text fields: first non-empty value wins
    """
    base = dict(rows[0])

    # ── Fields with details-wrapper ──
    for field in config["jsonb_fields_with_details"]:
        norm_list      = []
        original_jsons = []

        for row in rows:
            jdata = row.get(field)
            if is_empty_value(jdata) or "details" not in jdata:
                continue
            details = jdata["details"]
            # Include row if it has any method names, even if all values are empty
            if is_empty_value(details):
                continue
            original_jsons.append(jdata)
            norm_list.append(normalize_details(details))

        if norm_list:
            averaged = average_normalized_groups(norm_list)
            base[field] = build_new_jsonb_with_details(averaged, original_jsons)

    # ── Flat fields (no details-wrapper) ──
    for field in config["jsonb_fields_flat"]:
        norm_list = []

        for row in rows:
            jdata = row.get(field)
            if is_empty_value(jdata) or not isinstance(jdata, dict):
                continue
            # Include row if it has any method names, even if all values are empty
            if len(jdata) == 0:
                continue
            norm_list.append(normalize_details(jdata))

        if norm_list:
            averaged = average_normalized_groups(norm_list)
            base[field] = build_new_jsonb_flat(averaged)

    # ── Devices: union, original values kept ──
    if config["jsonb_fields_devices"]:
        merged_devs = merge_devices(rows)
        if merged_devs is not None:
            base["devices"] = merged_devs

    # ── TEXT[] fields: union of unique strings ──
    for field in config["array_fields"]:
        merged_arr = merge_array_field(rows, field)  # FIX: pass field explicitly
        if merged_arr is not None:
            base[field] = merged_arr

    # ── Milestones JSONB fields: union of unique milestones ──
    for field in config["jsonb_milestones_fields"]:
        merged_milestones = merge_milestones_jsonb(rows, field)
        if merged_milestones is not None:
            base[field] = merged_milestones

    # ── Latest-row JSONB fields: take value from highest age row ──
    for field in config["jsonb_latest_fields"]:
        latest_val = latest_row_value(rows, field)
        if latest_val is not None:
            base[field] = latest_val

    # ── yes/no fields: yes if ANY row is yes, no only if ALL are no ──
    for field in config["yes_no_fields"]:
        result = merge_yes_no(rows, field)
        if result is not None:
            base[field] = result

    # ── Concatenated text fields: all non-empty values joined in age order ──
    for field in config["concat_text_fields"]:
        result = concat_text_fields(rows, field)
        if result is not None:
            base[field] = result

    # ── Extra text fields: first non-empty value wins ──
    for field in config["extra_text_fields"]:
        for row in rows:
            val = row.get(field)
            if not is_empty_value(val):
                base[field] = val
                break

    return base


# ─── DB update ────────────────────────────────────────────────────────────────

def update_row(cur, table_name: str, row_id: str, merged: dict, config: dict):
    """
    Updates the existing row (by id) with all merged fields.
    Leaves id, introductory_id, user_id, age, created_at untouched.
    Skips fields that are None or empty.
    """
    all_jsonb_fields = (
        config["jsonb_fields_with_details"] +
        config["jsonb_fields_flat"] +
        (["devices"] if config["jsonb_fields_devices"] else [])
    )
    fields_to_update = (
        all_jsonb_fields +
        config["jsonb_milestones_fields"] +
        config["jsonb_latest_fields"] +
        config["array_fields"] +
        config["yes_no_fields"] +
        config["concat_text_fields"] +
        config["extra_text_fields"]
    )

    set_clauses = []
    values      = []

    for field in fields_to_update:
        val = merged.get(field)
        if is_empty_value(val):
            continue
        set_clauses.append(f'"{field}" = %s')
        if isinstance(val, dict):
            values.append(Json(val))
        elif isinstance(val, list):
            values.append(val)
        else:
            values.append(val)

    if not set_clauses:
        return

    values.append(row_id)
    cur.execute(
        f'UPDATE {table_name} SET {", ".join(set_clauses)} WHERE id = %s::uuid',
        values
    )


# ─── Dry run printer ──────────────────────────────────────────────────────────

def print_merged_dry(label: str, row_id, merged: dict, source_rows: list, config: dict):
    print(f"  [DRY] {label} → id={row_id}")

    all_jsonb_fields = (
        config["jsonb_fields_with_details"] +
        config["jsonb_fields_flat"] +
        (["devices"] if config["jsonb_fields_devices"] else [])
    )
    for field in all_jsonb_fields:
        jdata = merged.get(field)
        if is_empty_value(jdata):
            print(f"    {field}: (tom — hoppas över)")
            continue

        if field == "devices":
            # devices: special format — show names from devices array only
            device_list = jdata.get("devices", [])
            device_list = [d for d in device_list if d and d != "None"]
            if not device_list and not jdata.get("other"):
                print(f"    {field}: (tom — hoppas över)")
                continue
            print(f"    {field}:")
            for m in device_list:
                print(f"      · {m}  (union)")
            if jdata.get("other"):
                print(f"      other: {jdata['other']}")
            else:
                print(f"      other: \"\"")
            continue

        # FIX: simplified — "details" check covers both details-wrapper and devices
        if "details" in jdata:
            items = jdata["details"].items()
        else:
            items = {k: v for k, v in jdata.items() if isinstance(v, dict)}.items()

        print(f"    {field}:")
        for m, v in items:
            count = 0
            for r in source_rows:
                rdata   = r.get(field) or {}
                rdetails = rdata.get("details", rdata)
                if m in rdetails:
                    count += 1
            if field == "devices":
                print(f"      · {m}  (union)")
            else:
                tag = f"(avg of {count} rows)" if count > 1 else "(single row)"
                print(f"      · {m}: {v.get('hours')} hrs/week, {v.get('weeks')} weeks/year  {tag}")

    for field in config["array_fields"]:
        val = merged.get(field)
        if not is_empty_value(val):
            print(f"    {field}: {val}")

    for field in config["jsonb_milestones_fields"]:
        val = merged.get(field)
        if not is_empty_value(val):
            milestones = val.get("milestones", [])
            print(f"    {field}: {len(milestones)} unika milestones (union)")
            for m in milestones:
                print(f"      · {m}")

    for field in config["jsonb_latest_fields"]:
        val = merged.get(field)
        if not is_empty_value(val):
            selected = val.get("selected", list(val.get("details", {}).keys()))
            print(f"    {field}: hämtat från högsta åldern, {len(selected)} impairments")

    for field in config["yes_no_fields"]:
        val = merged.get(field)
        if not is_empty_value(val):
            print(f"    {field}: {val}")

    for field in config["concat_text_fields"]:
        val = merged.get(field)
        if not is_empty_value(val):
            preview = val[:80] + "..." if len(val) > 80 else val
            print(f"    {field} (concat): {preview}")

    for field in config["extra_text_fields"]:
        val = merged.get(field)
        if not is_empty_value(val):
            print(f"    {field}: (finns)")


# ─── Safety check ─────────────────────────────────────────────────────────────

def check_old_tables_exist(conn):
    """Ensures _old backup tables exist before any live changes are made."""
    with conn.cursor() as cur:
        for table in ["home_training_old", "intensive_therapies_old", "motorical_development_old"]:
            cur.execute(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)",
                (table,)
            )
            if not cur.fetchone()[0]:
                raise Exception(
                    f"\n❌ Säkerhetstabell '{table}' saknas!\n"
                    f"   Kör SQL-stegen för att skapa _old-tabeller innan du kör detta skript."
                )


# ─── Core migration ───────────────────────────────────────────────────────────

def migrate_table(conn, table_name: str, dry_run: bool = True):
    config = TABLE_CONFIG[table_name]

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            f"SELECT DISTINCT introductory_id FROM {table_name} WHERE age >= 3"
        )
        intro_ids = [r["introductory_id"] for r in cur.fetchall()]

        print(f"\n{'='*60}")
        print(f"Table : {table_name}")
        print(f"Affected introductory_ids: {len(intro_ids)}")
        print(f"Mode  : {'DRY RUN – no changes written' if dry_run else '⚡ LIVE – writing to DB'}")
        print("="*60)

        stats = {"age3_rows_updated": 0, "age4_rows_updated": 0, "rows_deleted": 0, "skipped": 0}

        for intro_id in intro_ids:
            # Fetch all rows at once — groups split in memory, never re-queried
            cur.execute(
                f"SELECT * FROM {table_name} "
                f"WHERE introductory_id = %s AND age >= 3 ORDER BY age",
                (str(intro_id),)
            )
            rows = cur.fetchall()

            group_3  = [r for r in rows if r["age"] in (3, 4)]
            group_4  = [r for r in rows if r["age"] >= 5]
            age3_row = next((r for r in group_3 if r["age"] == 3), None)
            age4_row = next((r for r in group_3 if r["age"] == 4), None)

            print(f"\n  intro_id : {intro_id}")
            print(f"  Ages in DB : {[r['age'] for r in rows]}")

            # ── Skip excluded introductory_ids ──
            if str(intro_id) in EXCLUDED_INTRODUCTORY_IDS:
                print(f"  [SKIP] Denna introductory_id är exkluderad — hoppas över.")
                continue

            # ── FIX: impossible case — no age-3 row but both age-4 AND age-5+ exist ──
            # Cannot save both buckets with only one row — skip entirely and warn
            if not age3_row and age4_row and group_4:
                msg = (
                    f"  ⚠️  VARNING: Saknar age-3 rad men har både age-4 och age-5+.\n"
                    f"      Kan inte spara båda buckets — hoppas över hela detta introductory_id!\n"
                    f"      Hantera detta manuellt."
                )
                print(msg)
                stats["skipped"] += 1
                continue

            # ── group_3: merge age 3+4 → write into age-3 row ──
            # If age-3 row is missing, use age-4 row as target (safe since group_4 is empty here)
            if group_3:
                merged3    = merge_group(group_3, config)
                target_row = age3_row if age3_row else age4_row

                if dry_run:
                    if not age3_row and age4_row:
                        print(f"  [DRY] ⚠️  Ingen age-3 rad — använder age-4 rad som target (group_4 är tom)")
                    print_merged_dry(
                        f"UPDATE age-3 bucket (merge of ages {[r['age'] for r in group_3]})",
                        target_row["id"] if target_row else "N/A",
                        merged3, group_3, config
                    )
                    if age4_row and age4_row != target_row and not group_4:
                        print(f"  [DRY] DELETE age-4 row id={age4_row['id']} (no age-5+ exists)")
                    elif age4_row and age4_row != target_row and group_4:
                        print(f"  [DRY] age-4 row id={age4_row['id']} kept — will receive age-5+ merge")
                else:
                    if target_row:
                        update_row(cur, table_name, str(target_row["id"]), merged3, config)
                        stats["age3_rows_updated"] += 1
                    # Only delete age-4 row if it wasn't used as target and group_4 won't use it
                    if age4_row and age4_row != target_row and not group_4:
                        cur.execute(f"DELETE FROM {table_name} WHERE id = %s::uuid", (str(age4_row["id"]),))
                        stats["rows_deleted"] += 1

            # ── group_4: merge age 5+ → write into age-4 row, delete age 5+ ──
            if group_4:
                merged4 = merge_group(group_4, config)

                if dry_run:
                    if age4_row:
                        print_merged_dry(
                            f"UPDATE age-4 bucket (merge of ages {[r['age'] for r in group_4]})",
                            age4_row["id"],
                            merged4, group_4, config
                        )
                        print(f"  [DRY] DELETE age-5+ rows: {[str(r['id']) for r in group_4]}")
                    else:
                        print(f"  [DRY] ⚠️  Ingen age-4 rad finns — age-5+ merge kan inte sparas och inget raderas. Hantera manuellt!")
                else:
                    if age4_row:
                        # FIX: deletion now inside the if-block — only delete if data was saved
                        update_row(cur, table_name, str(age4_row["id"]), merged4, config)
                        stats["age4_rows_updated"] += 1
                        ids_to_delete = [str(r["id"]) for r in group_4]
                        cur.execute(
                            f"DELETE FROM {table_name} WHERE id = ANY(%s::uuid[])",
                            (ids_to_delete,)
                        )
                        stats["rows_deleted"] += cur.rowcount
                    else:
                        print(f"  ⚠️  VARNING: Ingen age-4 rad för introductory_id={intro_id} — hoppade över group_4, inget raderat")

        if not dry_run:
            print(f"\n  Stats: {stats}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    conn = get_connection()
    try:
        check_old_tables_exist(conn)

        for table in ["home_training", "intensive_therapies", "motorical_development"]:
            migrate_table(conn, table, dry_run=True)

        print("\n" + "="*60)
        confirm = input("Apply all changes to the database? (yes/no): ").strip().lower()

        if confirm == "yes":
            for table in ["home_training", "intensive_therapies", "motorical_development"]:
                migrate_table(conn, table, dry_run=False)
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