import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
    """Extrahera romersk siffra ur GMFCS-strängen oavsett format."""
    if gmfcs_str is None:
        return None
    import re
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


# -------------------------------------------------------
# PCA 1 — Träningsprofil per barn
# -------------------------------------------------------

def build_training_profile(home_df: pl.DataFrame, neurohab_df: pl.DataFrame) -> pd.DataFrame:
    """
    Bygg en feature-matris där varje rad = ett barn och
    kolumnerna = totala träningstimmar per kategori (över alla år).
    """
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
    df = df.with_columns(
        [(pl.col(c) / 60).alias(c) for c in feature_cols]
    )

    return df.to_pandas().set_index("introductory_id")


def run_pca_training(
    home_df: pl.DataFrame,
    neurohab_df: pl.DataFrame,
    introductory_df: pl.DataFrame,
    filename: str = "pca_training_profile.png",
):
    """
    PCA på träningsprofil per barn, färgad efter GMFCS-nivå.
    """
    profile_df = build_training_profile(home_df, neurohab_df)
    gmfcs      = _gmfcs_map(introductory_df)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(profile_df)

    pca   = PCA(n_components=min(2, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_ * 100
    loadings  = pca.components_
    ids       = profile_df.index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel 1: Scatter färgad efter GMFCS ---
    ax = axes[0]
    for i, child_id in enumerate(ids):
        gmfcs_lvl = gmfcs.get(child_id, None)
        color = _gmfcs_color(gmfcs_lvl)
        pc2_val   = X_pca[i, 1] if X_pca.shape[1] > 1 else 0
        ax.scatter(X_pca[i, 0], pc2_val, color=color, s=100, zorder=3)
        ax.annotate(str(child_id)[:8], (X_pca[i, 0], pc2_val), fontsize=7, ha="left", va="bottom")

    ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.6, linestyle=":")
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}% förklarad varians)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}% förklarad varians)" if len(explained) > 1 else "PC2")
    ax.set_title("Träningsprofil per barn")
    _add_gmfcs_legend(ax)

    # --- Panel 2: Feature loadings ---
    ax2 = axes[1]
    feature_names = profile_df.columns.tolist()
    for j, feat in enumerate(feature_names):
        ax2.arrow(
            0, 0,
            loadings[0, j], loadings[1, j] if pca.n_components_ > 1 else 0,
            head_width=0.03, head_length=0.02,
            fc="steelblue", ec="steelblue", alpha=0.8,
        )
        ax2.text(
            loadings[0, j] * 1.1,
            (loadings[1, j] if pca.n_components_ > 1 else 0) * 1.1,
            feat, fontsize=8, ha="center",
        )

    ax2.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    ax2.axvline(0, color="gray", linewidth=0.6, linestyle=":")
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_xlabel("PC1 loadings")
    ax2.set_ylabel("PC2 loadings")
    ax2.set_title("Feature loadings (träningskategorier)")

    plt.suptitle("PCA — Träningsprofil per barn", fontsize=13)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / filename, dpi=150)
    print(f"Saved: {filename}")

    print("\nFörklarad varians per komponent:")
    for i, ev in enumerate(explained):
        print(f"  PC{i+1}: {ev:.1f}%")


# -------------------------------------------------------
# PCA 2 — Motorisk trajektorie per barn
# -------------------------------------------------------

def build_motor_trajectory(
    milestone_df: pl.DataFrame,
    impairment_df: pl.DataFrame,
    ages: list[int] = [1, 2, 3, 4, 5, 6, 7],
) -> pd.DataFrame:
    """
    Bygg en feature-matris där varje rad = ett barn och
    kolumnerna = milestone_score och mms_normalized vid varje ålder.
    Saknade åldrar imputeras med barnets eget medelvärde.
    """
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


def run_pca_motor_trajectory(
    milestone_df: pl.DataFrame,
    impairment_df: pl.DataFrame,
    introductory_df: pl.DataFrame,
    filename: str = "pca_motor_trajectory.png",
):
    """
    PCA på motorisk trajektorie per barn, färgad efter GMFCS-nivå.
    """
    traj_df = build_motor_trajectory(milestone_df, impairment_df)
    gmfcs   = _gmfcs_map(introductory_df)

    if len(traj_df) < 2:
        print("För få barn med komplett data för PCA på trajektorier.")
        return

    scaler       = StandardScaler()
    X_scaled     = scaler.fit_transform(traj_df)
    n_components = min(2, X_scaled.shape[0] - 1, X_scaled.shape[1])

    pca   = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Lägg till direkt efter pca.fit_transform i run_pca_motor_trajectory:
    loadings_df = pd.DataFrame(
    pca.components_.T,
    index=traj_df.columns,
    columns=[f"PC{i+1}" for i in range(n_components)]
    )
    print("\nLoadings:")
    print(loadings_df.round(3).sort_values("PC1", ascending=False))

    explained = pca.explained_variance_ratio_ * 100
    ids       = traj_df.index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel 1: Scatter färgad efter GMFCS ---
    ax = axes[0]
    for i, child_id in enumerate(ids):
        gmfcs_lvl = gmfcs.get(child_id, None)
        color = _gmfcs_color(gmfcs_lvl)
        pc2_val   = X_pca[i, 1] if n_components > 1 else 0
        ax.scatter(X_pca[i, 0], pc2_val, color=color, s=100, zorder=3)
        ax.annotate(str(child_id)[:8], (X_pca[i, 0], pc2_val), fontsize=7, ha="left", va="bottom")

    ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.6, linestyle=":")
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}% förklarad varians)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}% förklarad varians)" if n_components > 1 else "PC2")
    ax.set_title("Motorisk trajektorie per barn")
    _add_gmfcs_legend(ax)

    # --- Panel 2: Scree plot ---
    ax2      = axes[1]
    all_pca  = PCA().fit(X_scaled)
    all_exp  = all_pca.explained_variance_ratio_ * 100
    cum_exp  = np.cumsum(all_exp)

    ax2.bar(range(1, len(all_exp) + 1), all_exp, color="steelblue", alpha=0.7, label="Per komponent")
    ax2.plot(range(1, len(cum_exp) + 1), cum_exp, color="orange", marker="o", label="Kumulativ")
    ax2.axhline(80, color="red", linewidth=0.8, linestyle="--", label="80%-gräns")
    ax2.set_xlabel("Komponent")
    ax2.set_ylabel("Förklarad varians (%)")
    ax2.set_title("Scree plot")
    ax2.legend()

    plt.suptitle("PCA — Motorisk trajektorie per barn", fontsize=13)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / filename, dpi=150)
    print(f"Saved: {filename}")

    print("\nFörklarad varians per komponent:")
    for i, ev in enumerate(explained):
        print(f"  PC{i+1}: {ev:.1f}%")


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

    run_pca_training(
        home_df, neurohab_df,
        introductory_df=data["introductory"],
    )
    run_pca_motor_trajectory(
        milestone_df, impairment_df,
        introductory_df=data["introductory"],
    )

    plt.show()