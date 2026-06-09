"""
intensive_therapy_analysis.py
==============================
Jämför motorisk utveckling mellan barn som deltagit i intensivträning
(intensive therapy) och de som inte gjort det.

Analysen görs per åldersgrupp och inkluderar:
- Beskrivande statistik per grupp (Yes/No)
- Gruppjämförelse av motoriska scores
- Visualisering: box plots, linjediagram, stapeldiagram

Körning:
    python src/intensive_therapy_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

plt.rcParams.update({
    "font.size":        13,
    "axes.titlesize":   16,
    "axes.labelsize":   14,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  12,
})

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))


FIGURES_DIR = Path(__file__).resolve().parents[2] / "outputs" / "intensive_analysis"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# ID-filter — sätt till None för att inkludera alla barn,
# eller till en lista med introductory_id-strängar för att begränsa analysen.
#
# Exempel:
#   INCLUDE_IDS = None                  # alla barn
#   INCLUDE_IDS = ["abc123", "def456"]  # bara dessa två
# ════════════════════════════════════════════════════════════════════════════

INCLUDE_IDS: list[str] | None = [
    "e0129e7b-c90f-4f49-87d3-6987eb577cdb",
    "d87153c9-75b3-4305-99a4-42abc0366651",
    "47a9e4ae-8d91-4070-ae18-f2d9af891299",
    "1950325f-99da-47b4-b49d-735253ba0aaa",
    "ee9e6fc1-a9d3-4f45-aa0e-01723ebb2930",

    "498a4d90-77c6-41b2-ad39-517a2c2a9702",
    "fcdc7e60-1c2f-4178-bafd-db1d42e869ee",
    "7e68f3b3-509b-4352-8eb1-400c9407ac9b",
    "c8f4ec50-18b6-47ed-92a3-919da180a10d",
    "f1856ef8-2fe0-480d-9635-cfc0be308458",

    "52dac13b-a335-449d-a7db-a58e40b5e213",
    "42475b28-2dfd-4114-ac53-d8619881dd2f",
    "4be3b41c-a0b4-4e7b-ae49-896b37ea2052",
    "30302f7a-c470-47bf-8f0e-d104b3065d99",
    "8dba1f55-9e79-4e62-90c3-02e9609d3feb",
    "d2703a20-7b4a-4624-b31a-306eebe4caa0",
    "771d12c3-bc1a-4a97-ad27-00d35b24f87e",
    "1d0afd8d-6945-488a-964c-724e95db6696",
    "1019fb0a-480d-4bef-b8f9-493b9dfe253b",
    "6e7aeec2-2846-433d-a4ac-0e753da08530",
    "e30d335e-3a7a-484d-951d-f8e3f17ccfb3",
    "4c89ca0a-f5c3-4b7b-96be-a7919c679303",
    "578adb11-a12f-4121-a567-afe67c25640b",
    "0a584ba1-cdf4-4251-9168-5f8ccc0240e3",
    "7e42b31a-c597-4418-9bf6-a8c3286d049f",
    "3c1f5e61-56fd-4ac3-af9e-0d6fe054ddb7",
    "f9231c8d-2ade-4c0e-a878-a9524ccc3d65",
    "df67e7ea-0b50-408b-9342-4c29d0efa839",
    "16f3f961-07a2-4099-8498-1bad9c2faa19",
    "44cd783c-b33d-4553-89cd-2a73b59e1982",
    "cd26a009-6e51-4372-b151-b7d2bb8b7183",
    "c0990a55-916e-47ba-b29a-aee83d9f33c9",
    "89e4bf27-9a6f-45e8-a415-ef53f23f7931",
    "65ab3206-7371-4471-845c-6d238050494f",
]


# ════════════════════════════════════════════════════════════════════════════
# ID-filterfunktion
# ════════════════════════════════════════════════════════════════════════════

def filter_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtrerar DataFrame till INCLUDE_IDS om listan är satt.
    Skriver ut en sammanfattning av hur många barn som återstår.
    """
    total = df["introductory_id"].nunique()
    if INCLUDE_IDS is not None:
        df = df[df["introductory_id"].isin(INCLUDE_IDS)]
        missing = [i for i in INCLUDE_IDS if i not in df["introductory_id"].values]
        if missing:
            print(f"  Varning: dessa IDs hittades inte i datan: {missing}")
    remaining = df["introductory_id"].nunique()
    suffix = " (alla barn inkluderade)" if INCLUDE_IDS is None else f"  (filterlista: {len(INCLUDE_IDS)} IDs)"
    print(f"  ID-filter: {total} → {remaining} barn{suffix}")
    return df


# ════════════════════════════════════════════════════════════════════════════
# 1. Bygg analys-dataset
# ════════════════════════════════════════════════════════════════════════════

def build_intensive_therapy_flag(intensive_therapies_df: pl.DataFrame) -> pl.DataFrame:
    """
    Returnerar en tabell med en binär kolumn 'has_intensive_therapy'
    (1 = deltog minst en gång det året, 0 = deltog inte) per (introductory_id, age).
    """
    from preprocessing.intensive_therapies import process_neurohab_hours_per_user_per_age

    neurohab = process_neurohab_hours_per_user_per_age(intensive_therapies_df)

    if neurohab.is_empty():
        return pl.DataFrame(
            schema={"introductory_id": pl.Utf8, "age": pl.Int64, "has_intensive_therapy": pl.Int32}
        )

    return (
        neurohab
        .group_by(["introductory_id", "age"])
        .agg(pl.sum("total_hours").alias("neurohab_total_hours"))
        .with_columns(
            pl.when(pl.col("neurohab_total_hours") > 0)
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .cast(pl.Int32)
            .alias("has_intensive_therapy")
        )
        .select(["introductory_id", "age", "neurohab_total_hours", "has_intensive_therapy"])
        .sort(["introductory_id", "age"])
    )


def build_analysis_dataset(data: dict[str, pl.DataFrame]) -> pd.DataFrame:
    """
    Kombinerar motorisk score med intensivträningsflagga.

    Returns
    -------
    pd.DataFrame med kolumner:
        introductory_id, age, gmfcs_int,
        milestone_score_setvalue, impairment_score_setvalue, combined_score_setvalue,
        milestone_score, impairment_score, combined_score,
        neurohab_total_hours, has_intensive_therapy, therapy_group
    """
    from src.preprocessing.master_preprocessing import build_master_feature_table

    master = build_master_feature_table(data)

    therapy_flag = build_intensive_therapy_flag(data["intensive_therapies"])

    df = (
        master
        .join(therapy_flag, on=["introductory_id", "age"], how="left")
        .with_columns(
            pl.col("has_intensive_therapy").fill_null(0),
            pl.col("neurohab_total_hours").fill_null(0.0),
        )
        .with_columns(
            pl.when(pl.col("has_intensive_therapy") == 1)
            .then(pl.lit("Intensive therapy: Yes"))
            .otherwise(pl.lit("Intensive therapy: No"))
            .alias("therapy_group")
        )
    )

    df_pd = df.to_pandas()

    # ── Filtrera på godkända IDs ─────────────────────────────────────────────
    df_pd = filter_ids(df_pd)

    return df_pd


# ════════════════════════════════════════════════════════════════════════════
# 2. Beskrivande statistik
# ════════════════════════════════════════════════════════════════════════════
SCORE_COLS = {
    "milestone_score_setvalue":    "Milestone score (normalised)",
    "impairment_score_setvalue":   "Impairment score (normalised)",
    "combined_score_setvalue":     "Combined score (normalised)",
    "milestone_score":             "Milestone score (age-relative)",
    "impairment_score":            "Impairment score (age-relative)",
    "combined_score":              "Combined score (age-relative)",
}


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Beskrivande statistik per grupp (Yes/No) och score.
    """
    rows = []
    for col, label in SCORE_COLS.items():
        if col not in df.columns:
            continue
        for group_val, group_name in [(1, "Yes"), (0, "No")]:
            subset = df[df["has_intensive_therapy"] == group_val][col].dropna()
            rows.append({
                "Score":  label,
                "Grupp":  group_name,
                "n":      len(subset),
                "Medel":  round(subset.mean(), 4) if len(subset) else None,
                "Median": round(subset.median(), 4) if len(subset) else None,
                "Std":    round(subset.std(), 4) if len(subset) else None,
                "Min":    round(subset.min(), 4) if len(subset) else None,
                "Max":    round(subset.max(), 4) if len(subset) else None,
            })
    return pd.DataFrame(rows)


def stats_per_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Medelvärde och n per (ålder, grupp) för varje score-kolumn.
    """
    rows = []
    for col, label in SCORE_COLS.items():
        if col not in df.columns:
            continue
        for age in sorted(df["age"].unique()):
            for group_val, group_name in [(1, "Yes"), (0, "No")]:
                subset = df[
                    (df["age"] == age) & (df["has_intensive_therapy"] == group_val)
                ][col].dropna()
                if len(subset) == 0:
                    continue
                rows.append({
                    "Score":  label,
                    "Score_col": col,
                    "Grupp":  group_name,
                    "age":    age,
                    "n":      len(subset),
                    "mean":   subset.mean(),
                    "se":     subset.sem(),
                })
    return pd.DataFrame(rows)


def group_comparison_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mann-Whitney U-test per score-kolumn (hela kohorten, oavsett ålder).
    """
    rows = []
    for col, label in SCORE_COLS.items():
        if col not in df.columns:
            continue
        yes = df[df["has_intensive_therapy"] == 1][col].dropna()
        no  = df[df["has_intensive_therapy"] == 0][col].dropna()
        if len(yes) < 3 or len(no) < 3:
            continue
        stat, p = stats.mannwhitneyu(yes, no, alternative="two-sided")
        rows.append({
            "Score":     label,
            "n_Ja":      len(yes),
            "n_Nej":     len(no),
            "Mean_Ja":   round(yes.mean(), 4),
            "Mean_Nej":  round(no.mean(), 4),
            "Diff (Ja−Nej)": round(yes.mean() - no.mean(), 4),
            "U-statistic":  round(stat, 1),
            "p-value":      round(p, 4),
            "Signifikant":  "✓" if p < 0.05 else "",
        })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# 3. Print-sammanfattning
# ════════════════════════════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame) -> None:
    sep  = "=" * 72
    line = "-" * 72

    n_yes = (df["has_intensive_therapy"] == 1).sum()
    n_no  = (df["has_intensive_therapy"] == 0).sum()
    n_ids_yes = df[df["has_intensive_therapy"] == 1]["introductory_id"].nunique()
    n_ids_no  = df[df["has_intensive_therapy"] == 0]["introductory_id"].nunique()

    print(f"\n{sep}")
    print("  INTENSIVTRÄNING vs MOTORISK UTVECKLING — SAMMANFATTNING")
    print(sep)
    print(f"\n  Observationer totalt:  {len(df)}")
    print(f"  Med intensivträning:   {n_yes} rader  ({n_ids_yes} unika barn)")
    print(f"  Utan intensivträning:  {n_no} rader  ({n_ids_no} unika barn)")

    # ── Beskrivande statistik ──────────────────────────────────────────────
    print(f"\n  {line}")
    print("  BESKRIVANDE STATISTIK PER GRUPP\n")
    desc = descriptive_stats(df)
    for col_label in desc["Score"].unique():
        subset = desc[desc["Score"] == col_label]
        print(f"  {col_label}")
        for _, row in subset.iterrows():
            print(
                f"    {row['Grupp']:4s}  n={row['n']:>4}  "
                f"medel={row['Medel']:>7.4f}  median={row['Median']:>7.4f}  "
                f"std={row['Std']:>7.4f}"
            )
        print()

    # ── Statistiska tester ─────────────────────────────────────────────────
    print(f"  {line}")
    print("  GRUPPTESTER (Mann-Whitney U, two-sided)\n")
    tests = group_comparison_tests(df)
    header = (
        f"  {'Score':<40} {'n_Ja':>5} {'n_Nej':>6} "
        f"{'Ja':>7} {'Nej':>7} {'Diff':>7} {'p':>7} {'*':>3}"
    )
    print(header)
    print(f"  {line}")
    for _, row in tests.iterrows():
        print(
            f"  {row['Score']:<40} {row['n_Ja']:>5} {row['n_Nej']:>6} "
            f"{row['Mean_Ja']:>7.4f} {row['Mean_Nej']:>7.4f} "
            f"{row['Diff (Ja−Nej)']:>+7.4f} {row['p-value']:>7.4f} {row['Signifikant']:>3}"
        )

    # ── Per ålder ──────────────────────────────────────────────────────────
    print(f"\n  {line}")
    print("  MEDELVÄRDE PER ÅLDER OCH GRUPP (combined_score_setvalue)\n")
    age_stats = stats_per_age(df)
    col_filter = "combined_score_setvalue"
    age_sub = age_stats[age_stats["Score_col"] == col_filter]
    if not age_sub.empty:
        print(f"  {'Ålder':>6}  {'Grupp':>22}  {'n':>5}  {'Medel':>8}  {'SD':>7}")
        print(f"  {line}")
        for _, row in age_sub.sort_values(["age", "Grupp"]).iterrows():
            print(
                f"  {int(row['age']):>6}  {row['Grupp']:>22}  "
                f"{int(row['n']):>5}  {row['mean']:>8.4f}  {row['se']:>7.4f}"
            )

    print(f"\n{sep}\n")


# ════════════════════════════════════════════════════════════════════════════
# 4. Visualisering
# ════════════════════════════════════════════════════════════════════════════

COLORS = {
    "Yes": "#2196F3",
    "No":  "#FF9800",
}


def plot_boxplots(df: pd.DataFrame) -> None:
    """
    Box plots: distribution av varje motorisk score per grupp.
    """
    score_pairs = [
        ("milestone_score_setvalue",  "Milestone score\n(normalised)"),
        ("impairment_score_setvalue", "Impairment score\n(normalised)"),
    ]
    score_pairs = [(c, l) for c, l in score_pairs if c in df.columns]

    fig, axes = plt.subplots(1, len(score_pairs), figsize=(5 * len(score_pairs), 6), sharey=False)
    if len(score_pairs) == 1:
        axes = [axes]

    for ax, (col, label) in zip(axes, score_pairs):
        data_yes = df[df["has_intensive_therapy"] == 1][col].dropna()
        data_no  = df[df["has_intensive_therapy"] == 0][col].dropna()

        bp = ax.boxplot(
            [data_no, data_yes],
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="black", linewidth=2),
        )
        bp["boxes"][0].set_facecolor(COLORS["No"])
        bp["boxes"][1].set_facecolor(COLORS["Yes"])
        bp["boxes"][0].set_alpha(0.75)
        bp["boxes"][1].set_alpha(0.75)

        ax.set_xticks([1, 2])
        ax.set_xticklabels([
            f"No\n(n={len(data_no)})",
            f"Yes\n(n={len(data_yes)})",
        ])
        ax.set_title(label)
        ax.set_xlabel("Intensive therapy")

        # annotate mean diff
        diff = data_yes.mean() - data_no.mean()
        _, p = stats.mannwhitneyu(data_yes, data_no, alternative="two-sided") if (len(data_yes) > 2 and len(data_no) > 2) else (None, 1.0)
        annot = f"Δ={diff:+.3f}\np={p:.3f}"
        ax.annotate(
            annot,
            xy=(0.97, 0.97), xycoords="axes fraction",
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )

    axes[0].set_ylabel("Score")
    fig.suptitle("Motor Scores: Intensive Therapy Yes vs No", fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "intensive_therapy_boxplots.png", dpi=300, bbox_inches="tight")
    print("  → Sparad: intensive_therapy_boxplots.png")


def plot_score_by_age(df: pd.DataFrame) -> None:
    """
    Linjediagram: medelscores per åldersgrupp, uppdelat på grupp.
    """
    score_pairs = [
        ("milestone_score_setvalue",  "Milestone score (normalised)"),
        ("impairment_score_setvalue", "Impairment score (normalised)"),
    ]
    score_pairs = [(c, l) for c, l in score_pairs if c in df.columns]

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 10)
    )
    axes = axes.flatten()

    ages = sorted(df["age"].unique())

    for ax, (col, label) in zip(axes, score_pairs):
        for group_val, group_name in [(1, "Yes"), (0, "No")]:
            means, sems, age_ticks = [], [], []
            for age in ages:
                subset = df[
                    (df["age"] == age) & (df["has_intensive_therapy"] == group_val)
                ][col].dropna()
                if len(subset) == 0:
                    continue
                means.append(subset.mean())
                sems.append(subset.sem())
                age_ticks.append(age)

            ax.plot(
                age_ticks, means,
                marker="o", linewidth=2,
                color=COLORS[group_name],
                label=f"Intensive therapy: {group_name}"
            )
            ax.fill_between(
                age_ticks,
                [m - s for m, s in zip(means, sems)],
                [m + s for m, s in zip(means, sems)],
                color=COLORS[group_name], alpha=0.15,
            )

        ax.set_title(label)
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("Mean score (±SD)")
        ax.set_xticks(ages)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    for j in range(len(score_pairs), 4):
        axes[j].set_visible(False)

    fig.suptitle("Motor Scores by Age — Intensive Therapy Yes vs No", fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "intensive_therapy_by_age.png", dpi=300, bbox_inches="tight")
    print("  → Sparad: intensive_therapy_by_age.png")


def plot_participation_rate(df: pd.DataFrame) -> None:
    """
    Stapeldiagram: andel barn med intensivträning per ålder.
    """
    age_rates = (
        df.groupby("age")["has_intensive_therapy"]
        .agg(["mean", "sum", "count"])
        .reset_index()
        .rename(columns={"mean": "rate", "sum": "n_yes", "count": "n_total"})
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        age_rates["age"],
        age_rates["rate"] * 100,
        color="#2196F3", alpha=0.8, width=0.6,
    )

    for bar, (_, row) in zip(bars, age_rates.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{int(row['n_yes'])}/{int(row['n_total'])}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Participation in intensive therapy (%)")
    ax.set_title("Participation in Intensive Therapy by Age Group", fontsize=16, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_xticks(age_rates["age"])
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "intensive_therapy_participation_rate.png", dpi=300, bbox_inches="tight")
    print("  → Sparad: intensive_therapy_participation_rate.png")


def plot_gmfcs_breakdown(df: pd.DataFrame) -> None:
    """
    Stapeldiagram: andel med intensivträning per GMFCS-nivå.
    """
    if "gmfcs_int" not in df.columns:
        return

    gmfcs_df = (
        df.dropna(subset=["gmfcs_int"])
        .groupby("gmfcs_int")["has_intensive_therapy"]
        .agg(["mean", "sum", "count"])
        .reset_index()
        .rename(columns={"mean": "rate", "sum": "n_yes", "count": "n_total"})
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        gmfcs_df["gmfcs_int"].astype(int),
        gmfcs_df["rate"] * 100,
        color="#9C27B0", alpha=0.8, width=0.6,
    )

    for bar, (_, row) in zip(bars, gmfcs_df.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{int(row['n_yes'])}/{int(row['n_total'])}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_xlabel("GMFCS level")
    ax.set_ylabel("Participation in intensive therapy (%)")
    ax.set_title("Participation in Intensive Therapy by GMFCS Level", fontsize=16, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_xticks(sorted(gmfcs_df["gmfcs_int"].astype(int).tolist()))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "intensive_therapy_by_gmfcs.png", dpi=300, bbox_inches="tight")
    print("  → Sparad: intensive_therapy_by_gmfcs.png")


# ════════════════════════════════════════════════════════════════════════════
# 5. Main
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from src.dataloader import load_data
    from src.connect_db import get_connection

    conn = get_connection()
    data = load_data(conn)

    print("\nBygger analys-dataset …")
    df = build_analysis_dataset(data)

    print(f"Dataset klart: {len(df)} rader, {df['introductory_id'].nunique()} unika barn.\n")

    # ── Sammanfattning i terminalen ──────────────────────────────────────────
    print_summary(df)

    # ── Figurer ──────────────────────────────────────────────────────────────
    print("\nSkapar figurer …")
    plot_boxplots(df)
    plot_score_by_age(df)
    plot_participation_rate(df)
    plot_gmfcs_breakdown(df)

    plt.show()
    print("\nKlar!")