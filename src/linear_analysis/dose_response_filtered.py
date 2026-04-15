import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

FIGURES_DIR = Path(__file__).resolve().parents[2] / "outputs" / "dose_response_filtered"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# CONFIG — ändra här
# ════════════════════════════════════════════════════════════════════════════

CONFIG = {
    "filter_ids": [
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
        "1950325f-99da-47b4-b49d-735253ba0aaa",
    ],

    # Output — vilket/vilka motorscores att analysera
    "scores": {
        "milestones": {
            "delta_col": "delta_milestone_score_setvalue",
            "title":     "Milestone Score",
        },
        "impairments": {
            "delta_col": "delta_impairment_score_setvalue",
            "title":     "Impairment Score",
        },
        "combined": {
            "delta_col": "delta_combined_score_setvalue",
            "title":     "Combined Score",
        },
        "simplified": {
            "delta_col": "delta_motorical_score",
            "title":     "Simplified Score",
        }
        
    },

    # Input — träningskomponenter att inkludera i komponentanalysen
    "hour_components": [
        ("log_total_home_training_hours",  "Home training"),
        ("log_total_other_training_hours", "Sports / other"),
        ("log_neurohab_hours",             "Intensive therapy"),
        ("log_active_total_hours",         "Combined active total"),
        # ("log_cat_neurodevelopmental_reflex",         "Neurodevelopmental and Reflex based therapies"),
        # ("log_cat_motor_learning_task",         "Motor learning and task oriented learning"),
        # ("log_cat_technology_assisted",         "Technology assisted therapies"),
        # ("log_cat_suit_based",         "Suit based therapies"),
        # ("log_cat_physical_conditioning",         "Physical conditioning and activity based therapies"),
        # ("log_cat_complementary",         "Complementary therapies"),
      ],

    # Input — kolumn som används i overall dose-response-plotten
    # Bör vara en av kolumnerna ovan
    "overall_feature": "log_active_total_hours",
}

# ════════════════════════════════════════════════════════════════════════════
# Dataset builder — uses master feature table directly
# ════════════════════════════════════════════════════════════════════════════


def _get_treatment_cols(master_df: pd.DataFrame) -> list[str]:
    """Return med_* columns that represent real treatments (exclude negations)."""
    exclude = {"med_no", "med_none", "med_nej", "has_any_medical_treatment"}
    return [
        c for c in master_df.columns
        if c.startswith("med_") and c.lower() not in exclude
    ]


def build_analysis_df(
    master: pl.DataFrame,
    delta_col: str,
) -> pd.DataFrame:
    """
    Extract the relevant columns from master for one analysis.

    Returns a pandas DataFrame with:
      delta_score, hour components, active_total_hours, med_* columns.

    Rows where no training was reported (all hour components == 0) are dropped,
    as these almost certainly represent missing data rather than true zero training.
    """
    overall = CONFIG["overall_feature"]

    keep = list(dict.fromkeys(
        ["introductory_id", "age", delta_col]
        + [col for col, _ in CONFIG["hour_components"] if col in master.columns]
        + ([overall] if overall in master.columns else [])
        + [c for c in master.columns if c.startswith("med_")]
    ))

    keep = [c for c in keep if c in master.columns]   # guard against missing cols

    df = (
        master
        .select(keep)
        .filter(pl.col(delta_col).is_not_null())
        .pipe(lambda df: (
        df.filter(pl.col("introductory_id").is_in(CONFIG["filter_ids"])) #FILTRERAR BARA DE SOM ÄR FÄRDIGA OCH SKRIVS UPPE I CONFIG!!
            if CONFIG["filter_ids"] else df
        ))
        .to_pandas()
        .rename(columns={delta_col: "delta_score"})
    )

    # ── Drop rows where no training was reported ─────────────────────────────
    hour_cols = [col for col, _ in CONFIG["hour_components"] if col in df.columns]
    has_training = df[hour_cols].sum(axis=1) > 0
    n_dropped = (~has_training).sum()
    print(f"  [build_analysis_df] Dropping {n_dropped} rows with no training reported")
    df = df[has_training].reset_index(drop=True)

    # ── Drop rows where delta score is zero (likely missing second assessment) ─
    n_zero = (df["delta_score"] == 0).sum()
    print(f"  [build_analysis_df] Dropping {n_zero} rows with delta_score == 0")
    df = df[df["delta_score"] != 0].reset_index(drop=True)
    # ─────────────────────────────────────────────────────────────────────────
    return df


# ════════════════════════════════════════════════════════════════════════════
# Regression helpers
# ════════════════════════════════════════════════════════════════════════════

def _fit_linear(X: pd.DataFrame, y: pd.Series):
    model = LinearRegression()
    model.fit(X, y)
    r2 = r2_score(y, model.predict(X))
    return model, r2


def fit_linear_dose_response(df: pd.DataFrame, feature: str = "active_total_hours"):
    subset = df[["delta_score", feature]].dropna()
    X, y = subset[[feature]], subset["delta_score"]
    return _fit_linear(X, y)


def fit_polynomial_dose_response(
    df: pd.DataFrame, feature: str = "active_total_hours", degree: int = 2
):
    subset = df[["delta_score", feature]].dropna()
    X, y = subset[[feature]], subset["delta_score"]
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False), LinearRegression()
    )
    model.fit(X, y)
    r2 = r2_score(y, model.predict(X))
    return model, r2


# ════════════════════════════════════════════════════════════════════════════
# Run analysis
# ════════════════════════════════════════════════════════════════════════════

def run_analysis(master: pl.DataFrame, delta_col: str) -> dict:
    """
    Run all dose-response regressions using the master feature table.

    Args:
        master:     Master feature table from build_master_feature_table().
        delta_col:  Which delta column to use as the target, e.g.
                    'delta_milestone_score_setvalue'.

    Returns:
        dict with active_df, component_results, treatment_results,
        linear/poly model and R² values.
    """
    active_df = build_analysis_df(master, delta_col)
    treatment_cols = _get_treatment_cols(active_df)

    # ── Component regressions ────────────────────────────────────────────────
    # Only include participants who actually used each component (> 0),
    # so the regression reflects dose-response among users only.
    component_results = []
    for col, label in CONFIG["hour_components"]:
        if col not in active_df.columns:
            continue
        subset = active_df[["delta_score", col]].dropna()
        subset = subset[subset[col] > 0]  # users only — excludes non-participants
        if len(subset) < 5 or subset[col].sum() == 0:
            continue
        model, r2 = _fit_linear(subset[[col]], subset["delta_score"])
        component_results.append({
            "col":        col,
            "label":      label,
            "coeff":      model.coef_[0],
            "intercept":  model.intercept_,
            "r2":         r2,
            "n":          len(subset),
            "mean_hours": round(subset[col].mean(), 1),
        })

    # ── Treatment comparisons ────────────────────────────────────────────────
    treatment_results = []
    for col in treatment_cols:
        received     = active_df[active_df[col] == 1]["delta_score"].dropna()
        not_received = active_df[active_df[col] == 0]["delta_score"].dropna()
        if len(received) == 0:
            continue
        treatment_results.append({
            "col":        col,
            "n_received": len(received),
            "n_not":      len(not_received),
            "mean_yes":   round(received.mean(), 3),
            "mean_no":    round(not_received.mean(), 3),
            "mean_diff":  round(received.mean() - not_received.mean(), 3),
        })

    # ── Overall dose-response ────────────────────────────────────────────────
    feature = CONFIG["overall_feature"]
    linear_model, linear_r2 = fit_linear_dose_response(active_df, feature)
    poly_model,   poly_r2   = fit_polynomial_dose_response(active_df, feature)

    return {
        "active_df":         active_df,
        "component_results": component_results,
        "treatment_results": treatment_results,
        "treatment_cols":    treatment_cols,
        "linear_model":      linear_model,
        "linear_r2":         linear_r2,
        "poly_model":        poly_model,
        "poly_r2":           poly_r2,
        "overall_feature":   feature,
    }


# ════════════════════════════════════════════════════════════════════════════
# Print summary
# ════════════════════════════════════════════════════════════════════════════

def print_summary(results: dict, title: str = "Motor Score"):
    sep  = "=" * 62
    line = "-" * 62

    print(f"\n{sep}")
    print(f"  DOSE-RESPONSE ANALYSIS — {title.upper()}")
    print(sep)

    print("\n  Active Hours Components (users only, devices excluded)\n")
    print(f"  {'Component':<26} {'Coeff':>8} {'R²':>7} {'N':>6} {'Mean hrs':>10}")
    print(f"  {line}")
    for r in results["component_results"]:
        print(
            f"  {r['label']:<26} "
            f"{r['coeff']:>8.4f} "
            f"{r['r2']:>7.3f} "
            f"{r['n']:>6} "
            f"{r['mean_hours']:>10.1f}"
        )

    print(f"\n  {line}")
    print(f"  Overall Dose-Response ({results['overall_feature']})\n")
    print(f"    Linear model   R² = {results['linear_r2']:.4f}")
    print(f"    Poly model     R² = {results['poly_r2']:.4f}")

    if results["treatment_results"]:
        print(f"\n  {line}")
        print("  Medical Treatments\n")
        print(f"  {'Treatment':<28} {'Yes (n)':>8} {'No (n)':>8} {'Diff':>8}")
        print(f"  {line}")
        for t in results["treatment_results"]:
            print(
                f"  {t['col']:<28} {t['n_received']:>8} {t['n_not']:>8} "
                f"{t['mean_diff']:>+8.3f}"
            )

    print(f"\n{sep}\n")


# ════════════════════════════════════════════════════════════════════════════
# Figures
# ════════════════════════════════════════════════════════════════════════════

def plot_training_components(
    results: dict,
    title: str = "Motor Score",
    filename: str = "training_components.png",
):
    active_df = results["active_df"]
    panels    = results["component_results"]

    n       = len(panels)
    n_cols  = 2                       
    n_rows  = (n + n_cols - 1) // n_cols 

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    for i, r in enumerate(panels):
        ax  = axes[i]
        col = r["col"]

        subset = active_df[["delta_score", col]].dropna()
        subset = subset[subset[col] > 0]  # users only — matches run_analysis

        ax.scatter(subset[col], subset["delta_score"], alpha=0.4, color="steelblue", s=30)

        if len(subset) >= 5 and subset[col].sum() > 0:
            model, _ = _fit_linear(subset[[col]], subset["delta_score"])
            x_range    = np.linspace(subset[col].min(), subset[col].max(), 100)
            x_range_df = pd.DataFrame(x_range, columns=[col])
            ax.plot(x_range, model.predict(x_range_df), color="orange", linewidth=2)

        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_xlabel("Training dose (hours / year, users only)")
        ax.set_ylabel(f"Δ {title}")
        ax.set_title(r["label"])
        ax.annotate(
            f"k = {r['coeff']:.3f}\nR² = {r['r2']:.3f}\nn = {r['n']}",
            xy=(0.97, 0.97), xycoords="axes fraction",
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
)

    for j in range(len(panels), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Motorscore improvement by Training-category — {title}", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150)
    print(f"Saved: {filename}")


def plot_treatment_effects(
    results: dict,
    title: str = "Motor Score",
    filename: str = "treatment_effects.png",
):
    active_df      = results["active_df"]
    treatment_cols = results["treatment_cols"]

    if not treatment_cols:
        print("No treatment columns found — skipping treatment plot.")
        return

    n_cols = len(treatment_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), sharey=True)
    if n_cols == 1:
        axes = [axes]

    for ax, col in zip(axes, treatment_cols):
        received     = active_df[active_df[col] == 1]["delta_score"].dropna()
        not_received = active_df[active_df[col] == 0]["delta_score"].dropna()

        bp = ax.boxplot([not_received, received], patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor("#AED6F1")
        bp["boxes"][1].set_facecolor("#A9DFBF")

        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"No\n(n={len(not_received)})", f"Yes\n(n={len(received)})"])
        ax.set_title(col.replace("med_", "").replace("_", " ").title())
        ax.set_xlabel("Received treatment")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")

        diff = received.mean() - not_received.mean()
        ax.annotate(
            f"Δ mean = {diff:+.2f}",
            xy=(0.5, 0.97), xycoords="axes fraction",
            ha="center", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )

    axes[0].set_ylabel(f"Δ {title}")
    plt.suptitle(f"Score Change by Medical Treatment — {title}", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150)
    print(f"Saved: {filename}")


def plot_overall_dose_response(
    results: dict,
    title: str = "Motor Score",
    filename: str = "overall_dose_response.png",
):
    active_df    = results["active_df"]
    feature      = results["overall_feature"]
    linear_model = results["linear_model"]
    poly_model   = results["poly_model"]

    subset = active_df[["delta_score", feature]].dropna()
    X, y   = subset[[feature]], subset["delta_score"]

    x_range    = np.linspace(X[feature].min(), X[feature].max(), 200).reshape(-1, 1)
    x_range_df = pd.DataFrame(x_range, columns=[feature])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X[feature], y, alpha=0.5, color="steelblue", s=30, label="Observed")
    ax.plot(
        x_range, linear_model.predict(x_range_df),
        color="orange", linewidth=2,
        label=f"Linear (R²={results['linear_r2']:.3f})",
    )
    ax.plot(
        x_range, poly_model.predict(x_range_df),
        color="red", linewidth=2, linestyle="--",
        label=f"Polynomial deg-2 (R²={results['poly_r2']:.3f})",
    )

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Total active hours / year")
    ax.set_ylabel(f"Δ {title}")
    ax.set_title(f"Overall Dose-Response: Active Hours vs {title} Change")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150)
    print(f"Saved: {filename}")


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

    for key, cfg in CONFIG["scores"].items():
        results = run_analysis(master, delta_col=cfg["delta_col"])

        print_summary(results, title=cfg["title"])

        plot_training_components(
            results,
            title    = cfg["title"],
            filename = f"training_components_{key}.png",
        )
        plot_overall_dose_response(
            results,
            title    = cfg["title"],
            filename = f"overall_dose_response_{key}.png",
        )
        plot_treatment_effects(
            results,
            title    = cfg["title"],
            filename = f"treatment_effects_{key}.png",
        )

    plt.show()