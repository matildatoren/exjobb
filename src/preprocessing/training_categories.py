"""
training_categories.py
=======================
Maps training names from ``training_methods_therapies`` (home_training)
and ``methods_applied_during_intense_training`` (intensive_therapies) onto
six therapy categories, then aggregates hours per category per child per year.

Categories
----------
1. neurodevelopmental_reflex   – NDT, DNS, Vojta
2. motor_learning_task         – Bimanual, CIMT, DMI
3. technology_assisted         – Treadmill, Robotic, SpiderCage, E-stim, Vibration
4. suit_based                  – Mollii, TheraSuit / NeuroSuit
5. physical_conditioning       – Strength, Hippotherapy, Hydrotherapy, Climbing
6. complementary               – Massage & stretching, Acupuncture

Classification strategy
-----------------------
Standard answers are matched first (exact, case-insensitive).
Free-text answers fall through to a broader regex layer so that
user-written variants are still captured.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


# ════════════════════════════════════════════════════════════════════════════
# 1. Exact standard-answer map
#    Keys are the known fixed-choice strings from the survey (stripped,
#    lowercased at match time). Values are category names.
# ════════════════════════════════════════════════════════════════════════════

_EXACT: dict[str, str] = {
    # ── 1. Neurodevelopmental & Reflex-based ─────────────────────────────────
    "neurodevelopmental physiotherapy":           "neurodevelopmental_reflex",
    "dns (dynamic neuromuscular stabilization)":  "neurodevelopmental_reflex",
    "vojta":                                      "neurodevelopmental_reflex",

    # ── 2. Motor Learning & Task-oriented ────────────────────────────────────
    "bimanual training":                                      "motor_learning_task",
    "cimt (constraint-induced movement therapy)":             "motor_learning_task",
    "dmi (dynamic movement intervention)":                    "motor_learning_task",

    # ── 3. Technology-assisted & Specialized ─────────────────────────────────
    "treadmill training":                                     "technology_assisted",
    "robotic assisted therapy":                               "technology_assisted",
    "robotic-assisted therapy":                               "technology_assisted",
    "spidercage physical therapy":                            "technology_assisted",
    "electrical stimulation therapy (eg. e-stim)":            "technology_assisted",
    "vibration therapy (eg. galieo)":                         "technology_assisted",
    "vibration therapy (eg. galileo)":                        "technology_assisted",

    # ── 4. Suit-based ─────────────────────────────────────────────────────────
    "suit therapy - mollii":                                  "suit_based",
    "suit therapy - therasuit / neurosuit":                   "suit_based",
    "suit therapy - therasuit/neurosuit":                     "suit_based",

    # ── 5. Physical Conditioning & Activity-based ─────────────────────────────
    "strength training":   "physical_conditioning",
    "hippotherapy":        "physical_conditioning",
    "hydrotherapy":        "physical_conditioning",
    "climbing therapy":    "physical_conditioning",

    # ── 6. Complementary ──────────────────────────────────────────────────────
    "massage and stretching": "complementary",
    "acupuncture":            "complementary",
}


# ════════════════════════════════════════════════════════════════════════════
# 2. Regex fallback for free-text / user-written answers
#    Order matters: first match wins.
# ════════════════════════════════════════════════════════════════════════════

_RAW_FALLBACK: list[tuple[str, str]] = [

    # ── 1. Neurodevelopmental & Reflex-based ─────────────────────────────────
    (r"ndt|bobath|neurodevelopmental.physio|neurodev", "neurodevelopmental_reflex"),
    (r"\bdns\b|dynamic.neuromusc.stab",                "neurodevelopmental_reflex"),
    (r"vojta",                                         "neurodevelopmental_reflex"),

    # ── 2. Motor Learning & Task-oriented ────────────────────────────────────
    (r"bimanual",                        "motor_learning_task"),
    (r"cimt|constraint.induced",         "motor_learning_task"),
    (r"\bdmi\b|dynamic.movement.interv", "motor_learning_task"),

    # ── 3. Technology-assisted & Specialized ─────────────────────────────────
    (r"treadmill",                                                      "technology_assisted"),
    (r"robotic|lokomat|exoskeleton",                                    "technology_assisted"),
    (r"spider.?cage",                                                   "technology_assisted"),
    (r"e.?stim|electrical.stim|\bfes\b|\bnmes\b|neuromuscular.electr", "technology_assisted"),
    (r"vibration|galileo|galleo",                                       "technology_assisted"),

    # ── 4. Suit-based ─────────────────────────────────────────────────────────
    (r"mollii",                           "suit_based"),
    (r"therasuit|thera.suit",             "suit_based"),
    (r"neurosuit|napa.suit",              "suit_based"),
    (r"\bdso\b|dynamic.suit|suit.therap", "suit_based"),

    # ── 5. Physical Conditioning & Activity-based ─────────────────────────────
    (r"strength.train|styrketr|resistance.train", "physical_conditioning"),
    (r"hippother|horse.rid|häst",                 "physical_conditioning"),
    (r"hydro.?therap|water.train|vattentr|aqua",  "physical_conditioning"),
    (r"climbing|klättr|klimbning",                "physical_conditioning"),
    (r"\bsport\b|\bidrott\b|\bgymnastics\b",      "physical_conditioning"),

    # ── 6. Complementary ──────────────────────────────────────────────────────
    (r"massage|stretch",    "complementary"),
    (r"acupunct|akupunkt",  "complementary"),
]

_FALLBACK_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(raw, re.IGNORECASE), cat)
    for raw, cat in _RAW_FALLBACK
]


# ════════════════════════════════════════════════════════════════════════════
# Canonical column list
# ════════════════════════════════════════════════════════════════════════════

CATEGORY_COLS: list[str] = [
    "cat_neurodevelopmental_reflex",
    "cat_motor_learning_task",
    "cat_technology_assisted",
    "cat_suit_based",
    "cat_physical_conditioning",
    "cat_complementary",
    "cat_unclassified",   # catch-all so no hours are silently lost
]

_ALL_CATEGORIES: list[str] = [c.removeprefix("cat_") for c in CATEGORY_COLS]


# ════════════════════════════════════════════════════════════════════════════
# Classification
# ════════════════════════════════════════════════════════════════════════════

def classify_training_name(name: str) -> str:
    """
    Classify a raw training name.
    Exact standard answers are tried first; regex fallback handles free text.
    """
    key = name.strip().lower()

    # 1 — exact match
    if key in _EXACT:
        return _EXACT[key]

    # 2 — regex fallback (free-text / user-written variants)
    for pattern, category in _FALLBACK_PATTERNS:
        if pattern.search(name):
            return category

    return "unclassified"


# ════════════════════════════════════════════════════════════════════════════
# Extraction helpers
# ════════════════════════════════════════════════════════════════════════════

def _hours_from_info(info: Optional[dict]) -> float:
    """hours/week × weeks  →  total hours for the survey year."""
    if not info:
        return 0.0
    try:
        return float(info.get("hours") or 0) * float(info.get("weeks") or 0)
    except (TypeError, ValueError):
        return 0.0


def _extract_from_struct(struct) -> list[dict]:
    """
    Extract (name, hours) pairs from a struct that has a ``details`` dict.
    Used for both training_methods_therapies and
    methods_applied_during_intense_training.
    """
    if not struct or not isinstance(struct, dict):
        return []
    out = []
    for name, info in struct.get("details", {}).items():
        if name:
            out.append({"name": str(name), "hours": _hours_from_info(info)})
    return out


# ════════════════════════════════════════════════════════════════════════════
# Main builder
# ════════════════════════════════════════════════════════════════════════════

def build_category_hours_table(
    home_training_df: pl.DataFrame,
    intensive_therapies_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Return a wide table: one row per (introductory_id, age), one column per
    therapy category (prefixed ``cat_``).

    Sources
    -------
    - home_training_df        → col ``training_methods_therapies``
    - intensive_therapies_df  → col ``methods_applied_during_intense_training``
    """
    all_rows: list[dict] = []

    # ── home training ─────────────────────────────────────────────────────────
    for row in home_training_df.iter_rows(named=True):
        intro_id = row["introductory_id"]
        age      = row["age"]
        for item in _extract_from_struct(row.get("training_methods_therapies")):
            all_rows.append({
                "introductory_id": intro_id,
                "age":             age,
                "category":        classify_training_name(item["name"]),
                "hours":           item["hours"],
            })

    # ── intensive therapies ───────────────────────────────────────────────────
    for row in intensive_therapies_df.iter_rows(named=True):
        intro_id = row["introductory_id"]
        age      = row["age"]
        for item in _extract_from_struct(
            row.get("methods_applied_during_intense_training")
        ):
            all_rows.append({
                "introductory_id": intro_id,
                "age":             age,
                "category":        classify_training_name(item["name"]),
                "hours":           item["hours"],
            })

    # ── empty guard ───────────────────────────────────────────────────────────
    if not all_rows:
        return pl.DataFrame(
            schema={"introductory_id": pl.Utf8, "age": pl.Int64,
                    **{c: pl.Float64 for c in CATEGORY_COLS}},
        )

    long = pl.DataFrame(all_rows)

    # Sum hours per child × year × category
    summed = (
        long
        .group_by(["introductory_id", "age", "category"])
        .agg(pl.sum("hours").alias("total_hours"))
    )

    # Pivot to wide
    wide = (
        summed
        .pivot(
            values="total_hours",
            index=["introductory_id", "age"],
            on="category",
            aggregate_function="sum",
        )
        .fill_null(0)
    )

    # Rename pivot columns to cat_ prefix
    raw_cols = [c for c in wide.columns
                if c not in ("introductory_id", "age") and not c.startswith("cat_")]
    wide = wide.rename({c: f"cat_{c}" for c in raw_cols})

    # Guarantee every canonical column exists
    for col in CATEGORY_COLS:
        if col not in wide.columns:
            wide = wide.with_columns(pl.lit(0.0).alias(col))

    final_cols = ["introductory_id", "age"] + [c for c in CATEGORY_COLS if c in wide.columns]
    return wide.select(final_cols).sort(["introductory_id", "age"])


# ════════════════════════════════════════════════════════════════════════════
# Audit helper — run this once to verify classifications look correct
# ════════════════════════════════════════════════════════════════════════════

def audit_classifications(
    home_training_df: pl.DataFrame,
    intensive_therapies_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Returns every unique raw training name found in both source columns,
    alongside its assigned category. Run after connecting to spot
    anything landing in ``unclassified`` that needs a new pattern.
    """
    names: set[str] = set()

    for row in home_training_df.iter_rows(named=True):
        for item in _extract_from_struct(row.get("training_methods_therapies")):
            names.add(item["name"])

    for row in intensive_therapies_df.iter_rows(named=True):
        for item in _extract_from_struct(
            row.get("methods_applied_during_intense_training")
        ):
            names.add(item["name"])

    return pl.DataFrame([
        {"training_name": n, "assigned_category": classify_training_name(n)}
        for n in sorted(names)
    ]).sort(["assigned_category", "training_name"])


# ════════════════════════════════════════════════════════════════════════════
# Smoke-test
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from dataloader import load_data
    from connect_db import get_connection

    conn = get_connection()
    data = load_data(conn)

    audit = audit_classifications(data["home_training"], data["intensive_therapies"])
    print("=" * 60)
    print("Classification audit")
    print("=" * 60)
    print(audit)

    unclassified = audit.filter(pl.col("assigned_category") == "unclassified")
    if unclassified.height:
        print(f"\n⚠  {unclassified.height} unclassified — consider adding patterns:")
        print(unclassified)
    else:
        print("\n✓  All names classified.")

    cat_table = build_category_hours_table(
        data["home_training"],
        data["intensive_therapies"],
    )
    print(f"\nCategory hours table  →  {cat_table.shape[0]} rows × {cat_table.shape[1]} columns")
    print(cat_table.head(10))