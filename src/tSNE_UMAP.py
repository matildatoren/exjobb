import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import re

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap

BASE_DIR   = Path(__file__).resolve().parent.parent
IMAGES_DIR = BASE_DIR / "images"

GMFCS_COLORS = {
    "I":   "green",
    "II":  "blue",
    "III": "orange",
    "IV":  "red",
    "V":   "purple",
}


def _gmfcs_map(introductory_df: pl.DataFrame) -> dict:
    return dict(zip(
        introductory_df["id"].to_list(),
        introductory_df["gmfcs_lvl"].to_list(),
    ))


def _get_gmfcs_level(gmfcs_str: str) -> str:
    if gmfcs_str is None:
        return None
    match = re.search(r"Level\s+(I{1,3}V?|VI{0,3}|IV|V)", gmfcs_str)
    return match.group(1) if match else None


def _gmfcs_color(gmfcs_str: str) -> str:
    lvl = _get_gmfcs_level(gmfcs_str)
    return GMFCS_COLORS.get(lvl, "gray")


def _add_gmfcs_legend(ax):
    legend_elements = [
        Patch(facecolor=color, label=f"GMFCS {lvl}")
        for lvl, color in GMFCS_COLORS.items()
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="lower right")


def _scatter_panel(ax, X_2d, ids, gmfcs, title, xlabel, ylabel):
    """Återanvändbar scatter-panel med GMFCS-färger och etiketter."""
    for i, child_id in enumerate(ids):
        gmfcs_lvl = gmfcs.get(child_id, None)
        color     = _gmfcs_color(gmfcs_lvl)
        ax.scatter(X_2d[i, 0], X_2d[i, 1], color=color, s=100, zorder=3)
        ax.annotate(str(child_id)[:8], (X_2d[i, 0], X_2d[i, 1]), fontsize=7, ha="left", va="bottom")

    ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.6, linestyle=":")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _add_gmfcs_legend(ax)


# -------------------------------------------------------
# Bygg feature-matriser (återanvänds från pca_analysis)
# -------------------------------------------------------

def build_training_profile(home_df: pl.DataFrame, neurohab_df: pl.DataFrame) -> pd.DataFrame:
    category_hours = (
        home_df.group_by(["introductory_id", "training_category"])
        .agg(pl.sum("total_hours").alias("hours"))
        .pivot(values="hours", index="introductory_id", on="training_category")
        .fill_null(0)
    )
    neurohab_total = (
        neurohab_df.group_by("introductory_id")
        .agg(pl.sum("total_hours").alias("neurohab_hours"))
    )
    df = category_hours.join(neurohab_total, on="introductory_id", how="left").fill_null(0)
    feature_cols = [c for c in df.columns if c != "introductory_id"]
    df = df.with_columns([(pl.col(c) / 60).alias(c) for c in feature_cols])
    return df.to_pandas().set_index("introductory_id")


def build_motor_trajectory(
    milestone_df: pl.DataFrame,
    impairment_df: pl.DataFrame,
    ages: list[int] = [1, 2, 3, 4, 5, 6, 7],
) -> pd.DataFrame:
    all_ids = milestone_df["introductory_id"].unique().to_list()
    rows    = {}

    for child_id in all_ids:
        m_row  = milestone_df.filter(pl.col("introductory_id") == child_id)
        i_row  = impairment_df.filter(pl.col("introductory_id") == child_id)
        m_dict = dict(zip(m_row["age"].to_list(), m_row["milestone_score"].to_list()))
        i_dict = dict(zip(i_row["age"].to_list(), i_row["mms_normalized"].to_list()))

        feat = {}
        for age in ages:
            feat[f"milestone_age{age}"]  = m_dict.get(age, np.nan)
            feat[f"impairment_age{age}"] = i_dict.get(age, np.nan)

        rows[child_id] = feat

    df = pd.DataFrame(rows).T
    df.index.name = "introductory_id"

    milestone_cols  = [c for c in df.columns if c.startswith("milestone")]
    impairment_cols = [c for c in df.columns if c.startswith("impairment")]
    df[milestone_cols]  = df[milestone_cols].apply(lambda row: row.fillna(row.mean()), axis=1)
    df[impairment_cols] = df[impairment_cols].apply(lambda row: row.fillna(row.mean()), axis=1)

    return df.dropna()


# -------------------------------------------------------
# Kör t-SNE + UMAP på en given feature-matris
# -------------------------------------------------------

def run_tsne_umap(
    feature_df: pd.DataFrame,
    gmfcs: dict,
    suptitle: str,
    filename: str,
    perplexity: int = 5,
    n_neighbors: int = 5,
):
    """
    Kör t-SNE och UMAP på feature_df och plottar resultaten sida vid sida.

    Args:
        feature_df:   DataFrame med en rad per barn och features som kolumner.
        gmfcs:        Dict {introductory_id -> gmfcs_lvl-sträng}.
        suptitle:     Titel för hela figuren.
        filename:     Filnamn att spara i IMAGES_DIR.
        perplexity:   t-SNE perplexity — bör vara < n_samples. Default 5.
        n_neighbors:  UMAP n_neighbors — bör vara < n_samples. Default 5.
    """
    n = len(feature_df)
    # Säkerställ att parametrarna inte överstiger n
    perplexity  = min(perplexity,  n - 1)
    n_neighbors = min(n_neighbors, n - 1)

    ids = feature_df.index.tolist()

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df)

    # --- t-SNE ---
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)

    # --- UMAP ---
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
    X_umap  = reducer.fit_transform(X_scaled)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    _scatter_panel(
        axes[0], X_tsne, ids, gmfcs,
        title  = f"t-SNE (perplexity={perplexity})",
        xlabel = "Dim 1",
        ylabel = "Dim 2",
    )

    _scatter_panel(
        axes[1], X_umap, ids, gmfcs,
        title  = f"UMAP (n_neighbors={n_neighbors})",
        xlabel = "Dim 1",
        ylabel = "Dim 2",
    )

    plt.suptitle(suptitle, fontsize=13)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / filename, dpi=150)
    print(f"Saved: {filename}")


# -------------------------------------------------------
# Main
# -------------------------------------------------------

if __name__ == "__main__":
    from dataloader import load_data
    from connect_db import get_connection
    from preprocessing.motor_scores import (
        motorscore_milestones_setvalue,
        motorscore_impairments_setvalue,
    )
    from preprocessing.preprocessing_ht import process_training_per_type_per_year
    from preprocessing.preprocessing_it import process_neurohab_hours_per_user_per_age

    conn = get_connection()
    data = load_data(conn)

    milestone_df  = motorscore_milestones_setvalue(data["motorical_development"])
    impairment_df = motorscore_impairments_setvalue(data["motorical_development"])
    home_df       = process_training_per_type_per_year(data["home_training"])
    neurohab_df   = process_neurohab_hours_per_user_per_age(data["intensive_therapies"])

    gmfcs = _gmfcs_map(data["introductory"])

    # --- Träningsprofil ---
    training_df = build_training_profile(home_df, neurohab_df)
    run_tsne_umap(
        training_df, gmfcs,
        suptitle = "t-SNE & UMAP — Träningsprofil per barn",
        filename = "tsne_umap_training.png",
    )

    # --- Motorisk trajektorie ---
    motor_df = build_motor_trajectory(milestone_df, impairment_df)
    run_tsne_umap(
        motor_df, gmfcs,
        suptitle = "t-SNE & UMAP — Motorisk trajektorie per barn",
        filename = "tsne_umap_motor.png",
    )

    plt.show()