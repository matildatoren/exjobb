import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import re
from sklearn.cluster import KMeans
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap

IMAGES_DIR = Path(__file__).resolve().parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)

GMFCS_COLORS = {
    "I":   "green",
    "II":  "blue",
    "III": "orange",
    "IV":  "red",
    "V":   "purple",
}

CLUSTER_COLORS = {
    0: "blue",
    1: "red",
}

# ════════════════════════════════════════════════════════════════════════════
# FILTER — paste the introductory_ids you want to include.
# Leave empty to include all participants.
# ════════════════════════════════════════════════════════════════════════════

FILTER_IDS = [
        "c0990a55-916e-47ba-b29a-aee83d9f33c9",
        "65ab3206-7371-4471-845c-6d238050494f",
        "c8f4ec50-18b6-47ed-92a3-919da180a10d",
        "8dba1f55-9e79-4e62-90c3-02e9609d3feb",
        "f1856ef8-2fe0-480d-9635-cfc0be308458",
        "771d12c3-bc1a-4a97-ad27-00d35b24f87e",
        "1019fb0a-480d-4bef-b8f9-493b9dfe253b",
        "6e7aeec2-2846-433d-a4ac-0e753da08530",
        "e30d335e-3a7a-484d-951d-f8e3f17ccfb3",
        "578adb11-a12f-4121-a567-afe67c25640b",
        "0a584ba1-cdf4-4251-9168-5f8ccc0240e3",
        "7e42b31a-c597-4418-9bf6-a8c3286d049f",
        "89e4bf27-9a6f-45e8-a415-ef53f23f7931",
        "16f3f961-07a2-4099-8498-1bad9c2faa19",
        "44cd783c-b33d-4553-89cd-2a73b59e1982",
        "d2703a20-7b4a-4624-b31a-306eebe4caa0",
        "1d0afd8d-6945-488a-964c-724e95db6696",
        "f9231c8d-2ade-4c0e-a878-a9524ccc3d65",
        "cd26a009-6e51-4372-b151-b7d2bb8b7183",
        "df67e7ea-0b50-408b-9342-4c29d0efa839",
        "30302f7a-c470-47bf-8f0e-d104b3065d99",
]

# ────────────────────────────────────────────────────────────────────────────


def _apply_filter(df: pl.DataFrame, id_col: str = "introductory_id") -> pl.DataFrame:
    if FILTER_IDS:
        df = df.filter(pl.col(id_col).is_in(FILTER_IDS))
        print(f"  Filter active — {df[id_col].n_unique()} participants kept")
    else:
        print(f"  No filter — using all {df[id_col].n_unique()} participants")
    return df


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


def _add_cluster_legend(ax):
    legend_elements = [
        Patch(facecolor="blue", label="Cluster 1"),
        Patch(facecolor="red", label="Cluster 2"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))

def _scatter_panel(ax, X_2d, ids, gmfcs, title, xlabel, ylabel, cluster_labels=None):
    for i, child_id in enumerate(ids):
        if cluster_labels is None:
            gmfcs_lvl = gmfcs.get(child_id, None)
            color = _gmfcs_color(gmfcs_lvl)
        else:
            color = CLUSTER_COLORS[cluster_labels[i]]

        ax.scatter(X_2d[i, 0], X_2d[i, 1], color=color, s=100, zorder=3)
        ax.annotate(str(child_id)[:8], (X_2d[i, 0], X_2d[i, 1]), fontsize=7, ha="left", va="bottom")

    ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.6, linestyle=":")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if cluster_labels is None:
        _add_gmfcs_legend(ax)
    else:
        _add_cluster_legend(ax)


# ════════════════════════════════════════════════════════════════════════════
# Feature matrix builders
# ════════════════════════════════════════════════════════════════════════════

def build_training_profile(home_df: pl.DataFrame, neurohab_df: pl.DataFrame) -> pd.DataFrame:
    home_df     = _apply_filter(home_df)
    neurohab_df = _apply_filter(neurohab_df)

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
    ages: list[int] = [1, 2, 3, 4],
) -> pd.DataFrame:
    milestone_df  = _apply_filter(milestone_df)
    impairment_df = _apply_filter(impairment_df)

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


def build_combined_trajectory(
    combined_df: pl.DataFrame,
    ages: list[int] = [1, 2, 3, 4],
) -> pd.DataFrame:
    combined_df = _apply_filter(combined_df)
    all_ids = combined_df["introductory_id"].unique().to_list()
    rows = {}

    for child_id in all_ids:
        row    = combined_df.filter(pl.col("introductory_id") == child_id)
        c_dict = dict(zip(row["age"].to_list(), row["combined_score"].to_list()))
        rows[child_id] = {
            f"combined_age{age}": c_dict.get(age, np.nan)
            for age in ages
        }

    df = pd.DataFrame(rows).T
    df.index.name = "introductory_id"
    df = df.apply(lambda row: row.fillna(row.mean()), axis=1)
    return df.dropna()


# ════════════════════════════════════════════════════════════════════════════
# t-SNE + UMAP runner
# ════════════════════════════════════════════════════════════════════════════

def run_tsne_umap(
    feature_df: pd.DataFrame,
    gmfcs: dict,
    suptitle: str,
    filename: str,
    perplexity: int = 5,
    n_neighbors: int = 5,
    color_by_cluster: bool = False,
):
    n = len(feature_df)
    perplexity  = min(perplexity,  n - 1)
    n_neighbors = min(n_neighbors, n - 1)

    ids = feature_df.index.tolist()

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df)

    tsne   = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)

    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
    X_umap  = reducer.fit_transform(X_scaled)

    cluster_labels = None

    if color_by_cluster:
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    _scatter_panel(
        axes[0], X_tsne, ids, gmfcs,
        title  = f"t-SNE (perplexity={perplexity})",
        xlabel = "Dim 1",
        ylabel = "Dim 2",
        cluster_labels=cluster_labels,
    )
    _scatter_panel(
        axes[1], X_umap, ids, gmfcs,
        title  = f"UMAP (n_neighbors={n_neighbors})",
        xlabel = "Dim 1",
        ylabel = "Dim 2",
        cluster_labels=cluster_labels,
    )

    plt.suptitle(suptitle, fontsize=13)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / filename, dpi=150)
    print(f"Saved: {filename}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from src.dataloader import load_data
    from src.connect_db import get_connection
    from src.preprocessing.motor_scores import (
        motorscore_milestones_setvalue,
        motorscore_impairments_setvalue,
        motorscore_combined,
    )
    from preprocessing.preprocessing_ht import process_training_per_type_per_year
    from preprocessing.preprocessing_it import process_neurohab_hours_per_user_per_age

    conn = get_connection()
    data = load_data(conn)

    milestone_df  = motorscore_milestones_setvalue(data["motorical_development"], data["introductory"])
    impairment_df = motorscore_impairments_setvalue(data["motorical_development"], data["introductory"])
    combined_df   = motorscore_combined(milestone_df, impairment_df)
    home_df       = process_training_per_type_per_year(data["home_training"])
    neurohab_df   = process_neurohab_hours_per_user_per_age(data["intensive_therapies"])

    gmfcs = _gmfcs_map(data["introductory"])

    # ── Training profile ─────────────────────────────────────────────────────
    training_df = build_training_profile(home_df, neurohab_df)
    run_tsne_umap(
        training_df, gmfcs,
        suptitle = "t-SNE & UMAP — Träningsprofil per barn",
        filename = "tsne_umap_training.png",
        color_by_cluster = True,
    )

    # ── Combined score trajectory ─────────────────────────────────────────────
    combined_traj_df = build_combined_trajectory(combined_df)
    run_tsne_umap(
        combined_traj_df, gmfcs,
        suptitle = "t-SNE & UMAP — Motorical trajectory per child (combined score)",
        filename = "tsne_umap_combined.png",
        color_by_cluster = True,
    )

    plt.show()