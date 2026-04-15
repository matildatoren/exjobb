import sys
from pathlib import Path

import polars as pl

sys.path.append(str(Path(__file__).resolve().parents[1]))


# ════════════════════════════════════════════════════════════════════════════
# Helper functions
# ════════════════════════════════════════════════════════════════════════════

def count_milestones(struct) -> int:
    if struct is None:
        return 0

    data = dict(struct) if isinstance(struct, dict) else {}
    milestones = data.get("milestones", [])

    return len(milestones) if milestones else 0


def sum_impairments(struct) -> float:
    if struct is None:
        return 0.0

    data = dict(struct) if isinstance(struct, dict) else {}
    details = data.get("details", {})

    total = 0.0
    for value in details.values():
        try:
            total += float(value)
        except (TypeError, ValueError):
            pass

    return total


def count_impairments(struct) -> int:
    """Count how many impairments have a non-zero rating in details."""
    if struct is None:
        return 0
    data = dict(struct) if isinstance(struct, dict) else {}
    details = data.get("details", {})
    return sum(1 for v in details.values() if v not in (None, 0, 0.0, "", "0"))


def extract_milestone_keys(struct) -> set[str]:
    """
    Convert the milestones list into a set of stable keys (id/label/value or str()).
    Required to count unique milestones across years.
    Filters out empty placeholders (None, "", "None", dicts without id/value/label).
    """
    if struct is None:
        return set()

    data = struct if isinstance(struct, dict) else {}
    milestones = data.get("milestones", [])
    if not milestones:
        return set()

    keys: set[str] = set()

    for m in milestones:
        # 1) Skip explicit None
        if m is None:
            continue

        # 2) Dict milestones: use id/value/label if present, ignore empty dicts
        if isinstance(m, dict):
            mid = m.get("id")
            val = m.get("value")
            lab = m.get("label")

            # Empty placeholder dict — skip
            if mid is None and val is None and lab is None:
                continue

            if mid is not None:
                keys.add(str(mid))
            elif val is not None:
                keys.add(str(val))
            elif lab is not None:
                keys.add(str(lab))
            else:
                # Fallback (rarely triggered after the filters above)
                s = str(m).strip()
                if s and s.lower() != "none":
                    keys.add(s)

        # 3) Non-dict milestones (e.g. plain strings)
        else:
            s = str(m).strip()
            if not s or s.lower() == "none":
                continue
            keys.add(s)

    return keys


# ════════════════════════════════════════════════════════════════════════════
# Simplified motor score (score 1)
# ════════════════════════════════════════════════════════════════════════════

def process_motorical_score_1(df: pl.DataFrame) -> pl.DataFrame:
    """
    motorical_score = milestone_count - 0.1 * impairment_sum

    A simple combined score without normalization.
    Returns mean per (introductory_id, age).
    """
    gross_list = df["gross_motor_development"].to_list()
    fine_list = df["fine_motor_development"].to_list()
    lower_list = df["motorical_impairments_lower"].to_list()
    upper_list = df["motorical_impairments_upper"].to_list()

    scores = []
    for g, f, l, u in zip(gross_list, fine_list, lower_list, upper_list):
        milestones = count_milestones(g) + count_milestones(f)
        impairments = sum_impairments(l) + sum_impairments(u)
        scores.append(milestones - (0.1 * impairments))

    df = df.with_columns(pl.Series("motorical_score", scores))

    return (
        df.group_by(["introductory_id", "age"])
        .agg(pl.mean("motorical_score").alias("motorical_score"))
        .sort(["introductory_id", "age"])
    )


if __name__ == "__main__":
    from dataloader import load_data
    from connect_db import get_connection

    conn = get_connection()
    data = load_data(conn)

    motorical_dev = data["motorical_development"]

    score1 = process_motorical_score_1(motorical_dev)
    print("\nFinal motor score table:\n")
    print(score1)
