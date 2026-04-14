import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import re
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

IMAGES_DIR = Path(__file__).resolve().parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)

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
# Each entry defines one PCA run.
# - "cols"     : list of columns from master to use as features.
#                Pivoted wide by age (one feature per col×age).
#                Set ages=[] for static columns (no pivot).
# - "ages"     : which age buckets to include. Set to [] to skip pivot.
# - "title"    : suptitle for the plot
# - "filename" : output filename
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
#   Static (set ages=[]):
#     "gmfcs_int"

ANALYSES = [
    {
        "cols":     ["delta_milestone_score_setvalue", "delta_impairment_score_setvalue"],
        "ages":     [1, 2, 3, 4],
        "title":    "PCA — Δ Milestone + Δ Impairment score (setvalue)",
        "filename": "pca_delta_setvalue.png",
    },
    {
        "cols":     ["milestone_score_setvalue", "impairment_score_setvalue"],
        "ages":     [1, 2, 3, 4],
        "title":    "PCA — Milestone + Impairment score (setvalue)",
        "filename": "pca_scores_setvalue.png",
    },
    {
        "cols":     ["log_total_home_training_hours", "log_neurohab_hours", "log_total_other_training_hours"],
        "ages":     [1, 2, 3, 4],
        "title":    "PCA — Träningsprofil per barn",
        "filename": "pca_training.png",
    },
    {
        "cols":     ["combined_score_setvalue"],
        "ages":     [1, 2, 3, 4],
        "title":    "PCA — Combined score trajectory (setvalue)",
        "filename": "pca_combined.png",
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

    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"  Warning: columns not found in master and will be skipped: {missing}")
    cols = [c for c in cols if c in df.columns]

    if not cols:
        raise ValueError("No valid columns found in master.")

    rows = {}

    if ages:
        for child_id, grp in df.groupby("introductory_id"):
            feat = {}
            for col in cols:
                age_vals = dict(zip(grp["age"].tolist(), grp[col].tolist()))
                for age in ages:
                    feat[f"{col}_age{age}"] = age_vals.get(age, np.nan)
            rows[child_id] = feat
    else:
        for child_id, grp in df.groupby("introductory_id"):
            rows[child_id] = {
                col: grp[col].dropna().iloc[0] if not grp[col].dropna().empty else np.nan
                for col in cols
            }

    wide = pd.DataFrame(rows).T
    wide.index.name = "introductory_id"

    for col in wide.columns:
        wide[col] = wide[col].fillna(wide[col].mean())

    before = len(wide)
    wide = wide.dropna()
    print(f"  Feature matrix: {len(wide)} children × {wide.shape[1]} features "
          f"({before - len(wide)} dropped after imputation)")

    return wide


# ════════════════════════════════════════════════════════════════════════════
# PCA runner
# ════════════════════════════════════════════════════════════════════════════

def run_pca(
    feature_df: pd.DataFrame,
    title: str,
    filename: str,
):
    if len(feature_df) < 2:
        print(f"  Too few children for PCA — skipping {filename}")
        return

    ids      = feature_df.index.tolist()
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df)

    # ── Fit PCA for scatter (2 components) ──────────────────────────────────
    n_components = min(2, X_scaled.shape[0] - 1, X_scaled.shape[1])
    pca          = PCA(n_components=n_components)
    X_pca        = pca.fit_transform(X_scaled)
    explained    = pca.explained_variance_ratio_ * 100
    loadings     = pca.components_

    # ── Fit full PCA for scree ───────────────────────────────────────────────
    full_pca = PCA().fit(X_scaled)
    all_exp  = full_pca.explained_variance_ratio_ * 100
    cum_exp  = np.cumsum(all_exp)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Panel 1: scatter ─────────────────────────────────────────────────────
    ax = axes[0]
    for i, child_id in enumerate(ids):
        pc2_val = X_pca[i, 1] if n_components > 1 else 0
        ax.scatter(X_pca[i, 0], pc2_val, color="steelblue", s=100, zorder=3,
                   edgecolors="white", linewidths=0.5)
        ax.annotate(str(child_id)[:8], (X_pca[i, 0], pc2_val),
                    fontsize=7, ha="left", va="bottom", alpha=0.75)

    ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.6, linestyle=":")
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}% förklarad varians)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}% förklarad varians)" if n_components > 1 else "PC2")
    ax.set_title(title)

    # ── Panel 2: scree ───────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.bar(range(1, len(all_exp) + 1), all_exp,
            color="steelblue", alpha=0.7, label="Per komponent")
    ax2.plot(range(1, len(cum_exp) + 1), cum_exp,
             color="orange", marker="o", label="Kumulativ")
    ax2.axhline(80, color="red", linewidth=0.8, linestyle="--", label="80%-gräns")
    ax2.set_xlabel("Komponent")
    ax2.set_ylabel("Förklarad varians (%)")
    ax2.set_title("Scree plot")
    ax2.legend()

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / filename, dpi=150)
    print(f"  Saved: {filename}")
    plt.close()

    # ── Print loadings ───────────────────────────────────────────────────────
    loadings_df = pd.DataFrame(
        loadings.T,
        index=feature_df.columns,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )
    print("\n  Förklarad varians per komponent:")
    for i, ev in enumerate(explained):
        print(f"    PC{i+1}: {ev:.1f}%")
    print("\n  Loadings (sorterat på PC1):")
    print(loadings_df.round(3).sort_values("PC1", ascending=False).to_string())


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

    for cfg in ANALYSES:
        print(f"\n── {cfg['title']} ──")
        feature_df = build_feature_matrix(master, cols=cfg["cols"], ages=cfg["ages"])
        run_pca(
            feature_df,
            title    = cfg["title"],
            filename = cfg["filename"],
        )

    plt.show()