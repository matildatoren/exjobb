"""
Plot: Individual motorical score trajectories (age 1–4)
=======================================================
Plots one panel per participant. Use FILTER_IDS to include only
specific participants — leave it empty to include everyone.

Usage:
    cd /home/matilda/exjobb/src
    python ../plot_trajectories.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SRC = Path(__file__).resolve().parent / "src"
sys.path.append(str(SRC))

from src.connect_db import get_connection
from src.dataloader import load_data
from src.preprocessing.master_preprocessing import build_master_feature_table

IMAGES_DIR = Path(__file__).resolve().parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# FILTER — paste the introductory_ids you want to include.
# Leave empty to include all participants.
# ════════════════════════════════════════════════════════════════════════════

FILTER_IDS = [
        "771d12c3-bc1a-4a97-ad27-00d35b24f87e",
        "1d0afd8d-6945-488a-964c-724e95db6696",
        "1019fb0a-480d-4bef-b8f9-493b9dfe253b",
        "6e7aeec2-2846-433d-a4ac-0e753da08530",
        "e30d335e-3a7a-484d-951d-f8e3f17ccfb3",
        "578adb11-a12f-4121-a567-afe67c25640b",
        "0a584ba1-cdf4-4251-9168-5f8ccc0240e3",
        "7e42b31a-c597-4418-9bf6-a8c3286d049f",
        "8dba1f55-9e79-4e62-90c3-02e9609d3feb",
        "f9231c8d-2ade-4c0e-a878-a9524ccc3d65",
        "df67e7ea-0b50-408b-9342-4c29d0efa839",
        "16f3f961-07a2-4099-8498-1bad9c2faa19",
        "44cd783c-b33d-4553-89cd-2a73b59e1982",
        "cd26a009-6e51-4372-b151-b7d2bb8b7183",
        "c0990a55-916e-47ba-b29a-aee83d9f33c9",
        "c8f4ec50-18b6-47ed-92a3-919da180a10d",
        "d2703a20-7b4a-4624-b31a-306eebe4caa0",
        "89e4bf27-9a6f-45e8-a415-ef53f23f7931",
        "65ab3206-7371-4471-845c-6d238050494f",
        "f1856ef8-2fe0-480d-9635-cfc0be308458"
]

# ────────────────────────────────────────────────────────────────────────────

SCORE_COL = "milestone_score_setvalue"
GMFCS_COL = "gmfcs_int"

SCORE_LABELS = {
    "milestone_score_setvalue": "Milestone score (setvalue)",
}

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

    # ── Apply ID filter ──────────────────────────────────────────────────────
    if FILTER_IDS:
        df = df[df["introductory_id"].isin(FILTER_IDS)]
        print(f"  Filter active — keeping {df['introductory_id'].nunique()} participants")
    else:
        print(f"  No filter — using all {df['introductory_id'].nunique()} participants")

    # Keep only participants with at least 2 time points
    counts = df.groupby("introductory_id")[SCORE_COL].count()
    valid_ids = counts[counts >= 2].index
    df = df[df["introductory_id"].isin(valid_ids)]

    print(f"  Participants with ≥2 time points: {df['introductory_id'].nunique()}")
    return df


# ─── Individual panel plot ────────────────────────────────────────────────────

def plot_individual_trajectories(df: pd.DataFrame):
    """
    One subplot per participant, arranged in a grid.
    Each panel shows the motorical score over years, colored by GMFCS level.
    """
    ids = sorted(df["introductory_id"].unique())
    n   = len(ids)

    n_cols = min(4, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    all_ages   = sorted(df["age"].unique())
    all_scores = df[SCORE_COL].dropna()
    y_min = all_scores.min() - 0.05
    y_max = all_scores.max() + 0.05

    for i, pid in enumerate(ids):
        row_i = i // n_cols
        col_i = i % n_cols
        ax    = axes[row_i][col_i]

        group  = df[df["introductory_id"] == pid].sort_values("age")
        ages   = group["age"].values
        scores = group[SCORE_COL].values

        # Color by GMFCS if available
        gmfcs = group[GMFCS_COL].iloc[0] if GMFCS_COL in group.columns else None
        color = GMFCS_COLORS.get(int(gmfcs), "#555555") if pd.notna(gmfcs) else "#555555"

        ax.plot(ages, scores, color=color, linewidth=2, marker="o",
                markersize=6, zorder=3)

        # Annotate each point with its value
        for age, score in zip(ages, scores):
            ax.annotate(
                f"{score:.2f}",
                xy=(age, score),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center", fontsize=8, color=color,
            )

        # Delta annotation
        if len(scores) >= 2:
            delta = scores[-1] - scores[0]
            delta_str = f"Δ {delta:+.2f}"
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
        ax.set_ylabel("Milestone score (setvalue)", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        gmfcs_label = f"  GMFCS {int(gmfcs)}" if pd.notna(gmfcs) else ""
        short_id = pid[:8] + "…"
        ax.set_title(f"{short_id}{gmfcs_label}", fontsize=9)

    # Hide unused axes
    for j in range(len(ids), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_visible(False)

    # GMFCS legend
    legend_elements = [
        plt.Line2D([0], [0], color=c, linewidth=2, marker="o",
                   label=f"GMFCS {lvl}")
        for lvl, c in GMFCS_COLORS.items()
        if lvl in df[GMFCS_COL].dropna().astype(int).unique()
    ] if GMFCS_COL in df.columns else []

    if legend_elements:
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=len(legend_elements),
            fontsize=10,
            title="GMFCS level",
            bbox_to_anchor=(0.5, 0),
        )
        plt.subplots_adjust(bottom=0.08)

    filter_note = f" (filtered: {len(FILTER_IDS)} IDs)" if FILTER_IDS else ""
    fig.suptitle(
        f"Individual Milestone Score (Setvalue) Trajectories{filter_note}  —  n={n}",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()

    path = IMAGES_DIR / "milestone_trajectories_individual.png"
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
        print("  No data to plot — check your FILTER_IDS.")
        return

    print("\n  Plotting individual trajectories …")
    plot_individual_trajectories(df)

    print("\nDone.")


if __name__ == "__main__":
    main()