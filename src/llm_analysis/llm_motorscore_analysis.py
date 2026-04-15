"""
LLM Motor Score — Print-Ready Figures
=======================================
Generates clean, publication-quality figures from the merged dataset CSV.

Figures produced:
  1. Individual scatter + regression line for each log training variable (2×2)
     Y-axis: delta LLM motor score (year-over-year change)
  2. Spearman correlation bar chart
  3. LLM motor score distribution by GMFCS level (boxplot)
  4. Motor score trajectories per child over age
  5. Other training hours vs motor score, coloured by GMFCS

Usage:
    cd src/llm_analysis
    python llm_motorscore_analysis.py

Requires:
    outputs/llm_motorscore_regression/results/llm_motorscore_merged_dataset.csv  (from llm_motorscore_regression.py)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

# ─── Paths ────────────────────────────────────────────────────────────────────

_ROOT       = Path(__file__).resolve().parents[2]
MERGED_CSV  = _ROOT / "outputs" / "llm_motorscore_regression" / "results" / "llm_motorscore_merged_dataset.csv"
FIGURES_DIR = _ROOT / "outputs" / "llm_motorscore_analysis"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ─── Style ────────────────────────────────────────────────────────────────────

PALETTE = {
    "blue":       "#2B6CB0",
    "light_blue": "#BEE3F8",
    "green":      "#276749",
    "orange":     "#C05621",
    "purple":     "#553C9A",
    "gray":       "#718096",
    "light_gray": "#EDF2F7",
    "dark":       "#1A202C",
}

GMFCS_COLORS = {
    1: "#2B6CB0",
    2: "#38A169",
    3: "#D69E2E",
    4: "#C05621",
    5: "#702459",
}

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "axes.labelsize":     11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     1.1,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.major.size":   4,
    "ytick.major.size":   4,
    "legend.frameon":     False,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  "white",
})


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    if not MERGED_CSV.exists():
        raise FileNotFoundError(
            f"Merged dataset not found: {MERGED_CSV}\n"
            "Run llm_motorscore_regression.py first."
        )
    return pd.read_csv(MERGED_CSV)


def compute_delta_llm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute delta_llm_motor_score = score(age) - score(age-1) per child.
    Returns one row per year-to-year transition, keeping the training
    features from the later year (the year in which training occurred).
    """
    df = df.sort_values(["introductory_id", "age"]).copy()
    df["delta_llm_motor_score"] = (
        df.groupby("introductory_id")["llm_motor_score"].diff()
    )
    return df.dropna(subset=["delta_llm_motor_score"]).reset_index(drop=True)


def _add_regression_line(ax, x, y, color, lw=2.0):
    """Fit and draw OLS regression line with shaded 95% CI band."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 4:
        return None

    slope, intercept, r, p, _ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = intercept + slope * x_line

    n     = len(x)
    x_bar = x.mean()
    se    = np.sqrt(
        np.sum((y - (intercept + slope * x)) ** 2) / (n - 2)
        * (1 / n + (x_line - x_bar) ** 2 / np.sum((x - x_bar) ** 2))
    )
    t_crit = stats.t.ppf(0.975, df=n - 2)

    ax.fill_between(x_line, y_line - t_crit * se, y_line + t_crit * se,
                    alpha=0.12, color=color)
    ax.plot(x_line, y_line, color=color, linewidth=lw, zorder=3)

    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    return r, p, slope, sig


def _annotate_stats(ax, r, p, sig, loc="upper left"):
    txt   = f"ρ = {r:.2f}  {sig}"
    props = dict(boxstyle="round,pad=0.3", facecolor="white",
                 edgecolor="#CBD5E0", alpha=0.9)
    x_pos = 0.04 if loc == "upper left" else 0.96
    ha    = "left" if loc == "upper left" else "right"
    ax.text(x_pos, 0.95, txt, transform=ax.transAxes,
            fontsize=9.5, verticalalignment="top", ha=ha, bbox=props)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Individual regressions vs delta score (2×2 grid)
# ══════════════════════════════════════════════════════════════════════════════

def fig_individual_regressions(df: pd.DataFrame):
    delta_df = compute_delta_llm(df)
    n        = len(delta_df)
    n_kids   = delta_df["introductory_id"].nunique()

    variables = [
        ("log_total_home_training_hours",  "log Home training hours / year",          PALETTE["blue"]),
        ("log_total_other_training_hours", "log Other training hours / year",         PALETTE["green"]),
        ("log_neurohab_hours",             "log Intensive therapy hours / year",      PALETTE["orange"]),
        ("log_active_total_hours",         "log Total active training hours / year",  PALETTE["purple"]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, (feat, label, color) in zip(axes, variables):
        x = delta_df[feat].values.astype(float)
        y = delta_df["delta_llm_motor_score"].values.astype(float)

        jitter = np.random.default_rng(42).uniform(-0.06, 0.06, size=len(y))
        ax.scatter(x, y + jitter, color=color, alpha=0.55, s=45,
                   edgecolors="white", linewidths=0.5, zorder=2)

        out = _add_regression_line(ax, x, y, color)
        if out:
            r, p, slope, sig = out
            _annotate_stats(ax, r, p, sig,
                            loc="upper right" if slope < 0 else "upper left")

        ax.axhline(0, color=PALETTE["gray"], linewidth=0.8,
                   linestyle="--", alpha=0.6, zorder=1)
        ax.set_xlabel(label)
        ax.set_ylabel("Δ LLM motor score (year-over-year)")
        ax.set_title(label)
        ax.tick_params(labelsize=10)

    fig.suptitle(
        "Δ LLM Motor Score (year-over-year) vs Log Training Hours\n"
        f"(n={n} transitions, {n_kids} children, shaded = 95% CI)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout(h_pad=3.5, w_pad=3.0)

    path = FIGURES_DIR / "fig1_individual_regressions.pdf"
    plt.savefig(path)
    plt.savefig(path.with_suffix(".png"))
    print(f"  Saved: {path.name}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Spearman correlation bar chart
# ══════════════════════════════════════════════════════════════════════════════

def fig_spearman_bars(df: pd.DataFrame):
    features = {
        "log_total_home_training_hours":      "log Home training",
        "log_total_other_training_hours":     "log Other training",
        "log_neurohab_hours":                 "log Intensive therapy",
        "log_cat_neurodevelopmental_reflex":  "log Neurodevelopmental/reflex",
        "log_cat_motor_learning_task":        "log Motor learning",
        "log_cat_technology_assisted":        "log Technology assisted",
        "log_cat_suit_based":                 "log Suit based",
        "log_cat_physical_conditioning":      "log Physical conditioning",
        "log_cat_complementary":              "log Complementary",
        "has_any_device":                     "Uses any device",
        "has_any_medical_treatment":          "Any medical treatment",
        "gmfcs_int":                          "GMFCS level",
    }

    results = []
    for feat, label in features.items():
        if feat not in df.columns:
            continue
        sub = df[[feat, "llm_motor_score"]].dropna()
        r, p = stats.spearmanr(sub[feat], sub["llm_motor_score"])
        sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        results.append({"label": label, "rho": r, "p": p, "sig": sig, "n": len(sub)})

    res = pd.DataFrame(results).sort_values("rho")

    fig, ax = plt.subplots(figsize=(9, 7))

    colors = [PALETTE["green"] if r > 0 else PALETTE["orange"] for r in res["rho"]]
    bars   = ax.barh(res["label"], res["rho"], color=colors, alpha=0.82,
                     height=0.62, edgecolor="white", linewidth=0.5)

    for bar, (_, row) in zip(bars, res.iterrows()):
        if row["sig"]:
            x_pos = row["rho"] + (0.012 if row["rho"] >= 0 else -0.012)
            ha    = "left" if row["rho"] >= 0 else "right"
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                    row["sig"], va="center", ha=ha, fontsize=10,
                    color=PALETTE["dark"], fontweight="bold")

    ax.axvline(0, color=PALETTE["dark"], linewidth=1.0, zorder=5)
    ax.axvline( 0.3, color=PALETTE["gray"], linewidth=0.6, linestyle="--", alpha=0.5)
    ax.axvline(-0.3, color=PALETTE["gray"], linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xlabel("Spearman ρ  (with LLM motor score)")
    ax.set_title("Spearman Correlations with LLM Motor Score",
                 fontweight="bold", pad=12)
    ax.set_xlim(-0.65, 0.75)

    legend_elements = [
        Line2D([0], [0], color=PALETTE["green"],  lw=0, marker="s",
               markersize=10, label="Positive association"),
        Line2D([0], [0], color=PALETTE["orange"], lw=0, marker="s",
               markersize=10, label="Negative association"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    ax.text(0.99, -0.08, "* p<0.05  ** p<0.01  *** p<0.001",
            transform=ax.transAxes, ha="right", fontsize=8.5,
            color=PALETTE["gray"])

    plt.tight_layout()
    path = FIGURES_DIR / "fig2_spearman_correlations.pdf"
    plt.savefig(path)
    plt.savefig(path.with_suffix(".png"))
    print(f"  Saved: {path.name}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Motor score by GMFCS level (boxplot)
# ══════════════════════════════════════════════════════════════════════════════

def fig_gmfcs_boxplot(df: pd.DataFrame):
    if "gmfcs_int" not in df.columns:
        print("  gmfcs_int not found — skipping fig 3.")
        return

    gmfcs_levels = sorted(df["gmfcs_int"].dropna().unique())
    groups       = [df[df["gmfcs_int"] == lvl]["llm_motor_score"].dropna().values
                    for lvl in gmfcs_levels]
    labels       = [f"GMFCS {int(lvl)}\n(n={len(g)})" for lvl, g in zip(gmfcs_levels, groups)]

    fig, ax = plt.subplots(figsize=(9, 6))

    bp = ax.boxplot(groups, patch_artist=True, widths=0.52, notch=False,
                    medianprops=dict(color="white", linewidth=2.5),
                    whiskerprops=dict(linewidth=1.3),
                    capprops=dict(linewidth=1.3),
                    flierprops=dict(marker="o", markersize=4, alpha=0.5))

    for patch, lvl in zip(bp["boxes"], gmfcs_levels):
        patch.set_facecolor(GMFCS_COLORS.get(int(lvl), PALETTE["gray"]))
        patch.set_alpha(0.82)

    for i, (grp, lvl) in enumerate(zip(groups, gmfcs_levels), start=1):
        jitter = np.random.default_rng(int(lvl)).uniform(-0.18, 0.18, size=len(grp))
        ax.scatter(np.full(len(grp), i) + jitter, grp,
                   color=GMFCS_COLORS.get(int(lvl), PALETTE["gray"]),
                   alpha=0.45, s=30, zorder=3, edgecolors="white", linewidths=0.4)

    if len([g for g in groups if len(g) >= 3]) >= 2:
        stat, p = stats.kruskal(*[g for g in groups if len(g) >= 3])
        sig     = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.text(0.98, 0.97,
                f"Kruskal-Wallis\nH={stat:.2f}, p={p:.3f} {sig}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#CBD5E0", alpha=0.9))

    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("LLM motor score (1–10)")
    ax.set_ylim(0.2, 11.2)
    ax.set_yticks(range(1, 11))
    ax.set_title("LLM Motor Score Distribution by GMFCS Level",
                 fontweight="bold", pad=12)
    ax.tick_params(axis="x", length=0)

    plt.tight_layout()
    path = FIGURES_DIR / "fig3_gmfcs_boxplot.pdf"
    plt.savefig(path)
    plt.savefig(path.with_suffix(".png"))
    print(f"  Saved: {path.name}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Motor score trajectories per child over age
# ══════════════════════════════════════════════════════════════════════════════

def fig_trajectories(df: pd.DataFrame):
    if "gmfcs_int" not in df.columns:
        print("  gmfcs_int not found — skipping fig 4.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for child_id, grp in df.groupby("introductory_id"):
        grp = grp.sort_values("age")
        lvl = int(grp["gmfcs_int"].mode()[0]) if not grp["gmfcs_int"].isna().all() else 0
        col = GMFCS_COLORS.get(lvl, PALETTE["gray"])
        ax.plot(grp["age"], grp["llm_motor_score"],
                color=col, alpha=0.45, linewidth=1.4,
                marker="o", markersize=4, markeredgewidth=0.3,
                markeredgecolor="white", zorder=2)

    for lvl, col in GMFCS_COLORS.items():
        grp = df[df["gmfcs_int"] == lvl].groupby("age")["llm_motor_score"].mean()
        if len(grp) >= 2:
            ax.plot(grp.index, grp.values, color=col, linewidth=2.8,
                    marker="o", markersize=7, markeredgewidth=1.2,
                    markeredgecolor="white", zorder=4, label=f"GMFCS {lvl} (mean)")

    ax.set_xlabel("Age (years)")
    ax.set_ylabel("LLM motor score (1–10)")
    ax.set_title("LLM Motor Score Trajectories by Age\n(thin lines = individual children, thick = GMFCS group mean)",
                 fontweight="bold", pad=12)
    ax.set_yticks(range(1, 11))
    ax.set_xticks(sorted(df["age"].unique()))
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    path = FIGURES_DIR / "fig4_trajectories.pdf"
    plt.savefig(path)
    plt.savefig(path.with_suffix(".png"))
    print(f"  Saved: {path.name}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Other training hours vs motor score, coloured by GMFCS
# ══════════════════════════════════════════════════════════════════════════════

def fig_other_training_by_gmfcs(df: pd.DataFrame):
    if "gmfcs_int" not in df.columns or "total_other_training_hours" not in df.columns:
        print("  Required columns not found — skipping fig 5.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    for lvl in sorted(df["gmfcs_int"].dropna().unique()):
        grp   = df[df["gmfcs_int"] == lvl].dropna(
            subset=["total_other_training_hours", "llm_motor_score"])
        color = GMFCS_COLORS.get(int(lvl), PALETTE["gray"])

        jitter = np.random.default_rng(int(lvl) * 7).uniform(-0.1, 0.1, size=len(grp))
        ax.scatter(grp["total_other_training_hours"],
                   grp["llm_motor_score"] + jitter,
                   color=color, alpha=0.65, s=55,
                   edgecolors="white", linewidths=0.5,
                   label=f"GMFCS {int(lvl)} (n={len(grp)})", zorder=3)

        if len(grp) >= 5:
            x = grp["total_other_training_hours"].values.astype(float)
            y = grp["llm_motor_score"].values.astype(float)
            slope, intercept, *_ = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, intercept + slope * x_line,
                    color=color, linewidth=1.8, alpha=0.8, zorder=2)

    sub = df.dropna(subset=["total_other_training_hours", "llm_motor_score"])
    _add_regression_line(ax,
                         sub["total_other_training_hours"].values.astype(float),
                         sub["llm_motor_score"].values.astype(float),
                         color=PALETTE["dark"], lw=2.2)

    r, p = stats.spearmanr(sub["total_other_training_hours"], sub["llm_motor_score"])
    sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    ax.text(0.97, 0.05,
            f"Overall ρ = {r:.2f} {sig}",
            transform=ax.transAxes, ha="right", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#CBD5E0", alpha=0.9))

    ax.set_xlabel("Other training hours / year")
    ax.set_ylabel("LLM motor score (1–10)")
    ax.set_ylim(0.2, 11.2)
    ax.set_yticks(range(1, 11))
    ax.set_title("Other Training Hours vs LLM Motor Score by GMFCS Level",
                 fontweight="bold", pad=12)
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    path = FIGURES_DIR / "fig5_other_training_by_gmfcs.pdf"
    plt.savefig(path)
    plt.savefig(path.with_suffix(".png"))
    print(f"  Saved: {path.name}")
    plt.show()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Generating print-ready figures …")
    print("=" * 60)

    df = load_data()
    print(f"  Loaded {len(df)} rows, {df['introductory_id'].nunique()} children\n")

    np.random.seed(42)

    fig_individual_regressions(df)
    fig_spearman_bars(df)
    fig_gmfcs_boxplot(df)
    fig_trajectories(df)
    fig_other_training_by_gmfcs(df)

    print(f"\n  All figures saved to: {FIGURES_DIR}")
    print("  Both .pdf (print) and .png (preview) versions saved.")
    print("\nDone.")


if __name__ == "__main__":
    main()