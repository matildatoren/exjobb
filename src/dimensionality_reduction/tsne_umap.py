import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import re
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap

FIGURES_DIR = Path(__file__).resolve().parents[2] / "outputs" / "tsne_umap"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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
    2: "green",
    3: "purple",
}

# ════════════════════════════════════════════════════════════════════════════
# CONFIG — ändra här
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

# ── What to analyse ──────────────────────────────────────────────────────────
# Each entry in ANALYSES defines one t-SNE/UMAP run.
# - "cols"     : list of columns from master to use as features.
#                These are pivoted wide by age (one feature per col×age).
#                If the column has no age variation (e.g. gmfcs_int) it is
#                used as-is (one feature total).
# - "ages"     : which age buckets to include in the pivot (set to [] to skip pivot)
# - "title"    : suptitle for the plot
# - "filename" : output filename
# - "n_clusters": number of KMeans clusters (set to 0 to disable clustering)
#
# Examples of cols you can use from master:
#   Motor scores (per age):
#     "milestone_score_setvalue", "milestone_score"
#     "impairment_score_setvalue", "impairment_score"
#     "combined_score_setvalue", "combined_score"
#   Delta scores (per age):
#     "delta_milestone_score_setvalue", "delta_milestone_score"
#     "delta_impairment_score_setvalue", "delta_impairment_score"
#     "delta_combined_score_setvalue", "delta_combined_score"
#   Training (per age):
#     "log_total_home_training_hours", "log_neurohab_hours"
#     "log_active_total_hours", "log_total_other_training_hours"
#   Static (no age pivot needed — set ages=[]):
#     "gmfcs_int"

ANALYSES = [
    {
        "cols":       ["delta_milestone_score_setvalue", "delta_impairment_score_setvalue"],
        "ages":       [1, 2, 3, 4],
        "title":      "t-SNE & UMAP — Δ Milestone + Δ Impairment score (setvalue)",
        "filename":   "tsne_umap_delta_setvalue.png",
        "n_clusters": 2,
    },
    {
        "cols":       ["delta_milestone_score_setvalue"],
        "ages":       [1, 2, 3, 4],
        "title":      "t-SNE & UMAP — Δ Milestone score (setvalue)",
        "filename":   "tsne_umap_delta_milestone_setvalue.png",
        "n_clusters": 2,
    },
    {
        "cols":       ["delta_impairment_score_setvalue"],
        "ages":       [1, 2, 3, 4],
        "title":      "t-SNE & UMAP — Δ Impairment score (setvalue)",
        "filename":   "tsne_umap_delta_impairment_setvalue.png",
        "n_clusters": 2,
    },
    {
        "cols":       ["milestone_score_setvalue", "impairment_score_setvalue"],
        "ages":       [1, 2, 3, 4],
        "title":      "t-SNE & UMAP — Milestone + Impairment score (setvalue)",
        "filename":   "tsne_umap_scores_setvalue.png",
        "n_clusters": 2,
    },
    {
        "cols":       ["log_total_home_training_hours", "log_neurohab_hours", "log_total_other_training_hours"],
        "ages":       [1, 2, 3, 4],
        "title":      "t-SNE & UMAP — Träningsprofil per barn",
        "filename":   "tsne_umap_training.png",
        "n_clusters": 2,
    },
    {
        "cols":       ["combined_score_setvalue"],
        "ages":       [1, 2, 3, 4],
        "title":      "t-SNE & UMAP — Combined score trajectory (setvalue)",
        "filename":   "tsne_umap_combined.png",
        "n_clusters": 2,
    },
]

# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def _gmfcs_map(introductory_df: pl.DataFrame) -> dict:
    return dict(zip(
        introductory_df["id"].to_list(),
        introductory_df["gmfcs_lvl"].to_list(),
    ))


def _get_gmfcs_level(gmfcs_str: str) -> str:
    if gmfcs_str is None:
        return None
    match = re.search(r"Level\s+(I{1,3}V?|VI{0,3}|IV|V)", str(gmfcs_str))
    return match.group(1) if match else None


def _gmfcs_color(gmfcs_str: str) -> str:
    return GMFCS_COLORS.get(_get_gmfcs_level(gmfcs_str), "gray")


def _add_gmfcs_legend(ax):
    legend_elements = [
        Patch(facecolor=color, label=f"GMFCS {lvl}")
        for lvl, color in GMFCS_COLORS.items()
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="lower right")


def _add_cluster_legend(ax, n_clusters):
    legend_elements = [
        Patch(facecolor=CLUSTER_COLORS[k], label=f"Cluster {k+1}")
        for k in range(n_clusters)
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="lower right")


def _scatter_panel(ax, X_2d, ids, gmfcs, title, xlabel, ylabel, cluster_labels=None, n_clusters=2):
    for i, child_id in enumerate(ids):
        color = (
            CLUSTER_COLORS[cluster_labels[i]] if cluster_labels is not None
            else _gmfcs_color(gmfcs.get(child_id))
        )
        ax.scatter(X_2d[i, 0], X_2d[i, 1], color=color, s=100, zorder=3,
                   edgecolors="white", linewidths=0.5)
        ax.annotate(str(child_id)[:8], (X_2d[i, 0], X_2d[i, 1]),
                    fontsize=7, ha="left", va="bottom", alpha=0.75)

    ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.6, linestyle=":")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if cluster_labels is not None:
        _add_cluster_legend(ax, n_clusters)
    else:
        _add_gmfcs_legend(ax)


# ════════════════════════════════════════════════════════════════════════════
# Feature matrix builder — works directly from master
# ════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(
    master: pl.DataFrame,
    cols: list[str],
    ages: list[int],
) -> pd.DataFrame:
    """
    Build a per-child wide feature matrix from master.

    If ages is non-empty: pivot each col × age into separate features,
    then mean-impute missing age buckets within each child.
    If ages is empty: treat cols as static (one value per child, no pivot).
    """
    df = master.to_pandas()

    if FILTER_IDS:
        df = df[df["introductory_id"].isin(FILTER_IDS)]

    # Only keep cols that actually exist
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"  Warning: columns not found in master and will be skipped: {missing}")
    cols = [c for c in cols if c in df.columns]

    if not cols:
        raise ValueError("No valid columns found in master.")

    rows = {}

    if ages:
        # Pivot: one feature per (col, age)
        for child_id, grp in df.groupby("introductory_id"):
            feat = {}
            for col in cols:
                age_vals = dict(zip(grp["age"].tolist(), grp[col].tolist()))
                for age in ages:
                    feat[f"{col}_age{age}"] = age_vals.get(age, np.nan)
            rows[child_id] = feat
    else:
        # Static: one row per child (take first non-null value per col)
        for child_id, grp in df.groupby("introductory_id"):
            rows[child_id] = {
                col: grp[col].dropna().iloc[0] if not grp[col].dropna().empty else np.nan
                for col in cols
            }

    wide = pd.DataFrame(rows).T
    wide.index.name = "introductory_id"

    # Mean-impute per column across children
    for col in wide.columns:
        col_mean = wide[col].mean()
        wide[col] = wide[col].fillna(col_mean)

    before = len(wide)
    wide = wide.dropna()
    print(f"  Feature matrix: {len(wide)} children × {wide.shape[1]} features "
          f"({before - len(wide)} dropped after imputation)")

    return wide


# ════════════════════════════════════════════════════════════════════════════
# t-SNE + UMAP runner
# ════════════════════════════════════════════════════════════════════════════

def run_tsne_umap(
    feature_df: pd.DataFrame,
    gmfcs: dict,
    suptitle: str,
    filename: str,
    n_clusters: int = 2,
    perplexity: int = 5,
    n_neighbors: int = 5,
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
    if n_clusters > 0 and n >= n_clusters:
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = km.fit_predict(X_scaled)
        if n >= n_clusters + 1:
            sil = silhouette_score(X_scaled, cluster_labels)
            print(f"  Silhouette (k={n_clusters}): {sil:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    panels = [
        (axes[0], X_tsne, cluster_labels, f"t-SNE (perplexity={perplexity})"),
        (axes[1], X_umap, cluster_labels, f"UMAP (n_neighbors={n_neighbors})"),
    ]
    for ax, X_2d, cl, title in panels:
        _scatter_panel(ax, X_2d, ids, gmfcs, title,
                       xlabel="Dim 1", ylabel="Dim 2",
                       cluster_labels=cl, n_clusters=n_clusters)

    plt.suptitle(suptitle, fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150)
    print(f"  Saved: {filename}")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from src.dataloader import load_data
    from src.connect_db import get_connection
    from src.preprocessing.master_preprocessing import build_master_feature_table

    conn = get_connection()
    data = load_data(conn)

    master = build_master_feature_table(data)
    gmfcs  = _gmfcs_map(data["introductory"])

    for cfg in ANALYSES:
        print(f"\n── {cfg['title']} ──")
        feature_df = build_feature_matrix(master, cols=cfg["cols"], ages=cfg["ages"])
        run_tsne_umap(
            feature_df, gmfcs,
            suptitle   = cfg["title"],
            filename   = cfg["filename"],
            n_clusters = cfg["n_clusters"],
        )

    plt.show()