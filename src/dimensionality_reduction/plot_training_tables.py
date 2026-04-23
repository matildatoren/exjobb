"""
Plot: Training summary tables for Group A, B and C
=====================================================
Produces matplotlib table figures showing, per group:
  - % of participants who have ever attended intensive neurohabilitation training
  - Average training hours per therapy category (home + intensive)
  - Overall average active total hours (home + other + neurohabilitation)

Paste participant IDs into GROUP_A, GROUP_B and GROUP_C below.
Leave all groups empty to use all participants as a single group.

Usage:
    cd /home/matilda/exjobb/src
    python dimensionality_reduction/plot_training_tables.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import polars as pl

SRC = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC))

from src.connect_db import get_connection
from src.dataloader import load_data
from src.preprocessing.master_preprocessing import build_master_feature_table
from src.preprocessing.training_categories import CATEGORY_COLS

FIGURES_DIR = Path(__file__).resolve().parents[2] / "outputs" / "plot_training_tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# GROUPS — paste IDs here.
# ════════════════════════════════════════════════════════════════════════════

GROUP_A: list[str] = [
    "771d12c3-bc1a-4a97-ad27-00d35b24f87e",
    "e30d335e-3a7a-484d-951d-f8e3f17ccfb3",
    "df67e7ea-0b50-408b-9342-4c29d0efa839",
    "30302f7a-c470-47bf-8f0e-d104b3065d99",
    "1950325f-99da-47b4-b49d-735253ba0aaa",
    "8dba1f55-9e79-4e62-90c3-02e9609d3feb",
    "65ab3206-7371-4471-845c-6d238050494f",
    "0a584ba1-cdf4-4251-9168-5f8ccc0240e3",
]

GROUP_B: list[str] = [
    "cd26a009-6e51-4372-b151-b7d2bb8b7183",
    "c0990a55-916e-47ba-b29a-aee83d9f33c9",
    "c8f4ec50-18b6-47ed-92a3-919da180a10d",
    "44cd783c-b33d-4553-89cd-2a73b59e1982",
    "16f3f961-07a2-4099-8498-1bad9c2faa19",
    "7e42b31a-c597-4418-9bf6-a8c3286d049f",
    "578adb11-a12f-4121-a567-afe67c25640b",
    "f9231c8d-2ade-4c0e-a878-a9524ccc3d65",
    "6e7aeec2-2846-433d-a4ac-0e753da08530",
    "1d0afd8d-6945-488a-964c-724e95db6696",
    "1019fb0a-480d-4bef-b8f9-493b9dfe253b",
    "d2703a20-7b4a-4624-b31a-306eebe4caa0",
    "f1856ef8-2fe0-480d-9635-cfc0be308458",
    "89e4bf27-9a6f-45e8-a415-ef53f23f7931",
]

GROUP_C: list[str] = [
        "42475b28-2dfd-4114-ac53-d8619881dd2f",
        "7e68f3b3-509b-4352-8eb1-400c9407ac9b",
        "4be3b41c-a0b4-4e7b-ae49-896b37ea2052",
        "52dac13b-a335-449d-a7db-a58e40b5e213",
]

# ────────────────────────────────────────────────────────────────────────────

# Pretty display names for each therapy category column
CATEGORY_LABELS: dict[str, str] = {
    "cat_neurodevelopmental_reflex":  "Neurodevelopmental / Reflex",
    "cat_motor_learning_task":        "Motor Learning / Task-oriented",
    "cat_technology_assisted":        "Technology-assisted",
    "cat_suit_based":                 "Suit-based",
    "cat_physical_conditioning":      "Physical Conditioning",
    "cat_complementary":              "Complementary",
    "cat_unclassified":               "Unclassified",
}

# Hours columns to include in the "overall" row
TOTAL_HOURS_COLS = [
    ("total_home_training_hours",  "Home training hours"),
    ("total_other_training_hours", "Other training hours"),
    ("neurohab_hours",             "Neurohab hours"),
    ("active_total_hours",         "Active total hours"),
]

GROUP_COLORS = {
    "Group A": "#2948b9",
    "Group B": "#e63622",
    "Group C": "#27ae60",
}


# ─── Data preparation ────────────────────────────────────────────────────────

def _has_intensive(intensive_therapies: pl.DataFrame, ids: list[str]) -> float:
    """
    Fraction of IDs in *ids* who answered 'Yes' to
    participate_therapies_neurohabilitation at least once across all years.
    Returns 0.0 if ids is empty.
    """
    if not ids:
        return 0.0

    df = intensive_therapies.filter(
        pl.col("introductory_id").is_in(ids)
    )

    yes_ids = (
        df
        .filter(
            pl.col("participate_therapies_neurohabilitation")
            .cast(pl.Utf8)
            .str.to_lowercase()
            .is_in(["yes", "ja", "1", "true"])
        )
        ["introductory_id"]
        .unique()
        .to_list()
    )

    return len(yes_ids) / len(ids)


def _group_stats(master_pd: pd.DataFrame, ids: list[str]) -> dict:
    """
    Compute summary statistics for the given list of participant IDs.

    Returns a dict with:
      - 'n'                : number of IDs actually found in the data
      - 'category_avg'     : {col: mean hours across all observations}
      - 'total_hours_avg'  : {col_label: mean hours}
    """
    sub = master_pd[master_pd["introductory_id"].isin(ids)]

    # Per-participant averages across years, then group mean
    def _per_participant_mean(col: str) -> float:
        if col not in sub.columns:
            return float("nan")
        return sub.groupby("introductory_id")[col].mean().mean()

    cat_avg = {
        col: _per_participant_mean(col)
        for col in CATEGORY_COLS
        if col in master_pd.columns
    }

    total_avg = {
        label: _per_participant_mean(col)
        for col, label in TOTAL_HOURS_COLS
    }

    return {
        "n":               sub["introductory_id"].nunique(),
        "category_avg":    cat_avg,
        "total_hours_avg": total_avg,
    }


# ─── Plotting ────────────────────────────────────────────────────────────────

def _fmt(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    if isinstance(val, float):
        return f"{val:.1f}"
    return str(val)


def _pct_fmt(frac: float) -> str:
    return f"{frac * 100:.0f}%"


def plot_tables(
    groups: dict[str, dict],           # {group_label: stats_dict}
    intensive_fracs: dict[str, float], # {group_label: fraction with intensive}
) -> None:
    """Render and save three table figures."""

    group_labels = list(groups.keys())
    colors = [GROUP_COLORS.get(lbl, "#555555") for lbl in group_labels]

    # ── Figure 1: Intensive training participation ────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(max(5, len(group_labels) * 2.5), 2.8))
    ax1.axis("off")

    rows_data = [
        ["Group", "n", "% attended intensive training"],
        *[
            [
                lbl,
                str(groups[lbl]["n"]),
                _pct_fmt(intensive_fracs[lbl]),
            ]
            for lbl in group_labels
        ],
    ]

    col_widths = [0.35, 0.15, 0.50]
    table1 = ax1.table(
        cellText=rows_data[1:],
        colLabels=rows_data[0],
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(11)
    table1.scale(1, 1.6)

    for j, color in enumerate(colors):
        cell = table1[j + 1, 0]
        cell.set_facecolor(color + "33")   # light tint

    ax1.set_title("Intensive Neurohabilitation Training Participation",
                  fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    p1 = FIGURES_DIR / "table_intensive_participation.png"
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    print(f"  Saved: {p1.name}")
    plt.show()

    # ── Figure 2: Average hours per therapy category ─────────────────────────
    cat_cols = [c for c in CATEGORY_COLS if c in next(iter(groups.values()))["category_avg"]]

    row_labels = [CATEGORY_LABELS.get(c, c) for c in cat_cols]
    table_data = []
    for row_lbl, col in zip(row_labels, cat_cols):
        row = [row_lbl] + [_fmt(groups[lbl]["category_avg"].get(col)) for lbl in group_labels]
        table_data.append(row)

    col_headers = ["Therapy category"] + [f"{lbl}\n(avg h/yr)" for lbl in group_labels]
    col_w = [0.45] + [0.18] * len(group_labels)

    fig2, ax2 = plt.subplots(figsize=(max(6, len(group_labels) * 2.5 + 4), 0.5 * len(cat_cols) + 1.5))
    ax2.axis("off")

    table2 = ax2.table(
        cellText=table_data,
        colLabels=col_headers,
        cellLoc="center",
        loc="center",
        colWidths=col_w,
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1, 1.5)

    # Color header cells per group
    for j, color in enumerate(colors):
        table2[0, j + 1].set_facecolor(color + "55")

    ax2.set_title("Average Training Hours per Therapy Category (per participant per year)",
                  fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    p2 = FIGURES_DIR / "table_category_hours.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    print(f"  Saved: {p2.name}")
    plt.show()

    # ── Figure 3: Overall training hours ────────────────────────────────────
    overall_rows = []
    for lbl_col, lbl_display in TOTAL_HOURS_COLS:
        row = [lbl_display] + [_fmt(groups[lbl]["total_hours_avg"].get(lbl_display)) for lbl in group_labels]
        overall_rows.append(row)

    col_headers3 = ["Training measure"] + [f"{lbl}\n(avg h/yr)" for lbl in group_labels]
    col_w3 = [0.45] + [0.18] * len(group_labels)

    fig3, ax3 = plt.subplots(figsize=(max(6, len(group_labels) * 2.5 + 4), 2.5))
    ax3.axis("off")

    table3 = ax3.table(
        cellText=overall_rows,
        colLabels=col_headers3,
        cellLoc="center",
        loc="center",
        colWidths=col_w3,
    )
    table3.auto_set_font_size(False)
    table3.set_fontsize(10)
    table3.scale(1, 1.6)

    for j, color in enumerate(colors):
        table3[0, j + 1].set_facecolor(color + "55")

    ax3.set_title("Overall Average Training Hours per Year (per participant)",
                  fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    p3 = FIGURES_DIR / "table_overall_hours.png"
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    print(f"  Saved: {p3.name}")
    plt.show()


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Loading data …")
    conn = get_connection()
    data = load_data(conn)

    print("  Building master feature table …")
    master = build_master_feature_table(data)
    master_pd = master.to_pandas()

    intensive_therapies = data["intensive_therapies"].to_pandas()

    all_ids = sorted(master_pd["introductory_id"].unique())

    # Resolve groups
    if GROUP_A or GROUP_B or GROUP_C:
        resolved = {
            "Group A": [i for i in GROUP_A if i in all_ids],
            "Group B": [i for i in GROUP_B if i in all_ids],
            "Group C": [i for i in GROUP_C if i in all_ids],
        }
        # Warn about missing IDs
        for label, raw in [("Group A", GROUP_A), ("Group B", GROUP_B), ("Group C", GROUP_C)]:
            missing = [i for i in raw if i not in all_ids]
            if missing:
                print(f"  Warning: {label} IDs not found in data: {missing}")
        # Drop empty groups
        resolved = {k: v for k, v in resolved.items() if v}
    else:
        resolved = {"All participants": all_ids}

    print(f"\n  Groups: { {k: len(v) for k, v in resolved.items()} }")

    # Compute stats
    intensive_therapies_pl = data["intensive_therapies"]
    groups_stats = {}
    intensive_fracs = {}

    for label, ids in resolved.items():
        groups_stats[label] = _group_stats(master_pd, ids)
        intensive_fracs[label] = _has_intensive(intensive_therapies_pl, ids)
        print(f"  {label}: n={groups_stats[label]['n']}, "
              f"intensive={intensive_fracs[label]:.0%}")

    print("\n  Generating tables …")
    plot_tables(groups_stats, intensive_fracs)
    print("\nDone.")


if __name__ == "__main__":
    main()
