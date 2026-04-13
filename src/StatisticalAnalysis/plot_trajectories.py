"""
Plot: Individual score trajectories (age 1–4)
=======================================================
Plots one panel per participant, split into two separate images
using GROUP_A and GROUP_B, plus a combined image with colored borders.

Leave both groups empty to auto-split all participants in half.

Usage:
    cd /home/matilda/exjobb/src
    python ../plot_trajectories.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SRC = Path(__file__).resolve().parent / "src"
sys.path.append(str(SRC))

from src.connect_db import get_connection
from src.dataloader import load_data
from src.preprocessing.master_preprocessing import build_master_feature_table

IMAGES_DIR = Path(__file__).resolve().parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# GROUPS — paste IDs into GROUP_A and GROUP_B.
# Each group is saved as a separate image, plus a combined image.
# Leave both empty to auto-split all participants in half.
# ════════════════════════════════════════════════════════════════════════════

GROUP_A = [
    "16f3f961-07a2-4099-8498-1bad9c2faa19",
    "cd26a009-6e51-4372-b151-b7d2bb8b7183",
    "7e42b31a-c597-4418-9bf6-a8c3286d049f",
    "c8f4ec50-18b6-47ed-92a3-919da180a10d",
    "c0990a55-916e-47ba-b29a-aee83d9f33c9",
    "44cd783c-b33d-4553-89cd-2a73b59e1982",
    "f1856ef8-2fe0-480d-9635-cfc0be308458",
    "65ab3206-7371-4471-845c-6d238050494f",
    "89e4bf27-9a6f-45e8-a415-ef53f23f7931",
    "0a584ba1-cdf4-4251-9168-5f8ccc0240e3",
    "8dba1f55-9e79-4e62-90c3-02e9609d3feb",
]

GROUP_B = [
    "df67e7ea-0b50-408b-9342-4c29d0efa839",
    "578adb11-a12f-4121-a567-afe67c25640b",
    "f9231c8d-2ade-4c0e-a878-a9524ccc3d65",
    "e30d335e-3a7a-484d-951d-f8e3f17ccfb3",
    "771d12c3-bc1a-4a97-ad27-00d35b24f87e",
    "6e7aeec2-2846-433d-a4ac-0e753da08530",
    "1d0afd8d-6945-488a-964c-724e95db6696",
    "1019fb0a-480d-4bef-b8f9-493b9dfe253b",
    "d2703a20-7b4a-4624-b31a-306eebe4caa0",
]

# ────────────────────────────────────────────────────────────────────────────

SCORE_COL = "combined_score"
GMFCS_COL = "gmfcs_int"

# Border colors used in the combined plot to distinguish groups
COLOR_A = "#2980b9"   # blue
COLOR_B = "#e67e22"   # orange

GMFCS_COLORS = {
    1: "#3498db",
    2: "#2ecc71",
    3: "#f39c12",
    4: "#e74c3c",
    5: "#8e44ad",
}


# ─── Data ─────────────────────────────────────────────────────────────────────

def prepare_trajectories(master) -> pd.DataFrame:
    df = master.to_pandas()

    keep = ["introductory_id", "age", SCORE_COL]
    if GMFCS_COL in df.columns:
        keep.append(GMFCS_COL)

    df = df[keep].copy()
    df = df.dropna(subset=[SCORE_COL])

    counts    = df.groupby("introductory_id")[SCORE_COL].count()
    valid_ids = counts[counts >= 2].index
    df        = df[df["introductory_id"].isin(valid_ids)]

    print(f"  Total participants with ≥2 time points: {df['introductory_id'].nunique()}")
    return df


def split_into_groups(df: pd.DataFrame):
    all_ids = sorted(df["introductory_id"].unique())

    if GROUP_A or GROUP_B:
        ids_a   = [i for i in GROUP_A if i in all_ids]
        ids_b   = [i for i in GROUP_B if i in all_ids]
        label_a = "Group A"
        label_b = "Group B"
    else:
        mid     = len(all_ids) // 2
        ids_a   = all_ids[:mid]
        ids_b   = all_ids[mid:]
        label_a = "Part 1 (auto)"
        label_b = "Part 2 (auto)"

    df_a = df[df["introductory_id"].isin(ids_a)]
    df_b = df[df["introductory_id"].isin(ids_b)]

    print(f"  {label_a}: {df_a['introductory_id'].nunique()} participants")
    print(f"  {label_b}: {df_b['introductory_id'].nunique()} participants")

    missing_a = [i for i in GROUP_A if i not in all_ids]
    missing_b = [i for i in GROUP_B if i not in all_ids]
    if missing_a:
        print(f"  ⚠️  GROUP_A IDs not found in data: {missing_a}")
    if missing_b:
        print(f"  ⚠️  GROUP_B IDs not found in data: {missing_b}")

    return df_a, df_b, label_a, label_b


# ─── Shared panel drawing helper ──────────────────────────────────────────────

def _draw_panel(ax, pid, group, all_ages, y_min, y_max, line_color, border_color=None, title=None):
    ages   = group["age"].values
    scores = group[SCORE_COL].values

    ax.plot(ages, scores, color=line_color, linewidth=2, marker="o", markersize=6, zorder=3)

    for age, score in zip(ages, scores):
        ax.annotate(
            f"{score:.2f}",
            xy=(age, score),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center", fontsize=8, color=line_color,
        )

    if len(scores) >= 2:
        delta       = scores[-1] - scores[0]
        delta_str   = f"Δ {delta:+.2f}"
        delta_color = "#27ae60" if delta > 0 else "#e74c3c" if delta < 0 else "#7f8c8d"
        ax.annotate(
            delta_str,
            xy=(0.97, 0.06), xycoords="axes fraction",
            ha="right", va="bottom", fontsize=9,
            color=delta_color, fontweight="bold",
        )

    ax.axhline(0, color="lightgray", linewidth=0.8, linestyle=":")
    ax.set_xticks(all_ages)
    ax.set_xticklabels([f"Year {a}" for a in all_ages], fontsize=8)
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel(SCORE_COL.replace("_", " ").title(), fontsize=8)

    # Optionally color the panel border
    if border_color:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2.5)
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if title:
        ax.set_title(title, fontsize=9)


# ─── Individual group plots ───────────────────────────────────────────────────

def plot_individual_trajectories(
    df: pd.DataFrame,
    group_label: str,
    filename: str,
    y_min: float,
    y_max: float,
    all_ages: list,
):
    ids = sorted(df["introductory_id"].unique())
    n   = len(ids)

    if n == 0:
        print(f"  No participants in {group_label} — skipping.")
        return

    n_cols = min(4, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for i, pid in enumerate(ids):
        ax    = axes[i // n_cols][i % n_cols]
        group = df[df["introductory_id"] == pid].sort_values("age")
        gmfcs = group[GMFCS_COL].iloc[0] if GMFCS_COL in group.columns else None
        color = GMFCS_COLORS.get(int(gmfcs), "#555555") if pd.notna(gmfcs) else "#555555"
        gmfcs_label = f"  GMFCS {int(gmfcs)}" if pd.notna(gmfcs) else ""

        _draw_panel(ax, pid, group, all_ages, y_min, y_max,
                    line_color=color,
                    title=f"{pid[:8]}…{gmfcs_label}")

    for j in range(n, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_visible(False)

    legend_elements = [
        plt.Line2D([0], [0], color=c, linewidth=2, marker="o", label=f"GMFCS {lvl}")
        for lvl, c in GMFCS_COLORS.items()
        if lvl in df[GMFCS_COL].dropna().astype(int).unique()
    ] if GMFCS_COL in df.columns else []

    if legend_elements:
        fig.legend(handles=legend_elements, loc="lower center",
                   ncol=len(legend_elements), fontsize=10,
                   title="GMFCS level", bbox_to_anchor=(0.5, 0))
        plt.subplots_adjust(bottom=0.08)

    fig.suptitle(f"Individual Trajectories — {group_label}  (n={n})", fontsize=14, y=1.01)
    plt.tight_layout()
    path = IMAGES_DIR / filename
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path.name}")
    plt.show()


# ─── Combined plot (both groups, colored borders) ─────────────────────────────

def plot_combined_groups(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    filename: str,
    y_min: float,
    y_max: float,
    all_ages: list,
):
    ids_a  = set(df_a["introductory_id"].unique())
    ids_b  = set(df_b["introductory_id"].unique())
    df_all = pd.concat([df_a, df_b])
    ids    = sorted(df_all["introductory_id"].unique())
    n      = len(ids)

    if n == 0:
        print("  No participants to plot.")
        return

    n_cols = min(4, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for i, pid in enumerate(ids):
        ax    = axes[i // n_cols][i % n_cols]
        group = df_all[df_all["introductory_id"] == pid].sort_values("age")
        gmfcs = group[GMFCS_COL].iloc[0] if GMFCS_COL in group.columns else None
        color = GMFCS_COLORS.get(int(gmfcs), "#555555") if pd.notna(gmfcs) else "#555555"

        border      = COLOR_A if pid in ids_a else COLOR_B
        group_tag   = "[A]" if pid in ids_a else "[B]"
        gmfcs_label = f"  GMFCS {int(gmfcs)}" if pd.notna(gmfcs) else ""

        _draw_panel(ax, pid, group, all_ages, y_min, y_max,
                    line_color=color,
                    border_color=border,
                    title=f"{group_tag} {pid[:8]}…{gmfcs_label}")

    for j in range(n, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_visible(False)

    # Legend: GMFCS line colors + group border colors
    gmfcs_elements = [
        plt.Line2D([0], [0], color=c, linewidth=2, marker="o", label=f"GMFCS {lvl}")
        for lvl, c in GMFCS_COLORS.items()
        if lvl in df_all[GMFCS_COL].dropna().astype(int).unique()
    ] if GMFCS_COL in df_all.columns else []

    group_elements = [
        plt.Line2D([0], [0], color=COLOR_A, linewidth=3, label=f"Border = {label_a}"),
        plt.Line2D([0], [0], color=COLOR_B, linewidth=3, label=f"Border = {label_b}"),
    ]

    fig.legend(
        handles=gmfcs_elements + group_elements,
        loc="lower center",
        ncol=len(gmfcs_elements) + 2,
        fontsize=9,
        bbox_to_anchor=(0.5, 0),
    )
    plt.subplots_adjust(bottom=0.06)

    fig.suptitle(
        f"All Trajectories — {label_a} (blue border) vs {label_b} (orange border)  (n={n})",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    path = IMAGES_DIR / filename
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path.name}")
    plt.show()


# ─── Group average plot ───────────────────────────────────────────────────────

def plot_group_averages(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    filename: str,
    y_min: float,
    y_max: float,
    all_ages: list,
):
    """
    Single plot showing the mean trajectory +/- std for each group.
    Individual trajectories shown as faint lines in the background.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for df, color, label in [
        (df_a, COLOR_A, label_a),
        (df_b, COLOR_B, label_b),
    ]:
        # Faint individual lines
        for pid, group in df.groupby("introductory_id"):
            g = group.sort_values("age")
            ax.plot(g["age"], g[SCORE_COL], color=color, alpha=0.15, linewidth=1)

        # Mean and std per age
        stats = df.groupby("age")[SCORE_COL].agg(["mean", "std", "count"]).reindex(all_ages)
        means = stats["mean"]
        stds  = stats["std"].fillna(0)
        ns    = stats["count"].fillna(0)

        ax.plot(means.index, means.values, color=color, linewidth=3,
                marker="o", markersize=8, zorder=5, label=f"{label} mean")
        ax.fill_between(means.index,
                        means.values - stds.values,
                        means.values + stds.values,
                        color=color, alpha=0.15, label=f"{label} +/-1 SD")

        # Annotate mean value and n at each age point
        for age in means.index:
            if pd.notna(means[age]):
                n = int(ns[age])
                ax.annotate(
                    f"{means[age]:.2f}\n(n={n})",
                    xy=(age, means[age]),
                    xytext=(0, 12),
                    textcoords="offset points",
                    ha="center", fontsize=8, color=color, fontweight="bold",
                )

    ax.axhline(0, color="lightgray", linewidth=0.8, linestyle=":")
    ax.set_xticks(all_ages)
    ax.set_xticklabels([f"Year {a}" for a in all_ages], fontsize=11)
    ax.set_ylabel(SCORE_COL.replace("_", " ").title(), fontsize=12)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"Group Average Trajectories: {label_a} vs {label_b}", fontsize=13)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = IMAGES_DIR / filename
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path.name}")
    plt.show()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Loading data …")
    conn = get_connection()
    data = load_data(conn)

    print("  Building master feature table …")
    master = build_master_feature_table(data)

    print("\n  Preparing trajectory data …")
    df = prepare_trajectories(master)

    if df.empty:
        print("  No data to plot.")
        return

    print("\n  Splitting into groups …")
    df_a, df_b, label_a, label_b = split_into_groups(df)

    all_ages = sorted(df["age"].unique())
    y_min    = df[SCORE_COL].min() - 0.05
    y_max    = df[SCORE_COL].max() + 0.05

    print("\n  Plotting …")
    plot_individual_trajectories(
        df_a, label_a,
        filename="trajectories_group_a.png",
        y_min=y_min, y_max=y_max, all_ages=all_ages,
    )
    plot_individual_trajectories(
        df_b, label_b,
        filename="trajectories_group_b.png",
        y_min=y_min, y_max=y_max, all_ages=all_ages,
    )
    plot_combined_groups(
        df_a, df_b, label_a, label_b,
        filename="trajectories_combined.png",
        y_min=y_min, y_max=y_max, all_ages=all_ages,
    )

    plot_group_averages(
        df_a, df_b, label_a, label_b,
        filename="trajectories_group_averages.png",
        y_min=y_min, y_max=y_max, all_ages=all_ages,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()