"""
llm_score_vs_own_ms.py
======================
Compares LLM-derived motor scores against the project's rule-based scores.

Outputs:
  - scatter plots (one per score type, grouped in one figure)
  - Bland-Altman plot for combined score
  - MAE-by-age line chart
  - summary metrics table figure
  - outlier tables (top rows with largest absolute delta, one figure per score type)
  - one joined CSV for manual investigation

Note: This is not a validation of clinical accuracy — the LLM scores are an
exploratory complement to the rule-based system, not a ground truth comparison.
"""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

# ── resolve imports ────────────────────────────────────────────────────────────
SRC_ROOT     = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(SRC_ROOT))

from connect_db import get_connection
from dataloader import load_data

from src.preprocessing.motor_scores import (
    motorscore_milestones_setvalue,
    motorscore_impairments_setvalue,
    motorscore_combined,
)

# ── paths ──────────────────────────────────────────────────────────────────────
LLM_CSV    = PROJECT_ROOT / "outputs" / "motorscore_analysis" / "llm_motorscore_results.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "motorscore_comparison"
IMAGES_DIR = OUTPUT_DIR / "images"
JOINED_CSV = OUTPUT_DIR / "llm_vs_manual_joined.csv"   # useful for manual outlier lookup

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# ── matplotlib style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        150,
})

# ══════════════════════════════════════════════════════════════════════════════
# Pure-Python statistics helpers (no scipy)
# ══════════════════════════════════════════════════════════════════════════════

def _clean(series: pl.Series) -> list[float]:
    return [float(x) for x in series.to_list() if x is not None]


def _mean(v: list[float]) -> float | None:
    return sum(v) / len(v) if v else None


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx, my = _mean(x), _mean(y)
    dx = [xi - mx for xi in x]
    dy = [yi - my for yi in y]
    num   = sum(a * b for a, b in zip(dx, dy))
    denom = math.sqrt(sum(a**2 for a in dx)) * math.sqrt(sum(b**2 for b in dy))
    return num / denom if denom else None


def _rank(v: list[float]) -> list[float]:
    pairs = sorted(enumerate(v), key=lambda t: t[1])
    ranks = [0.0] * len(v)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][1] == pairs[i][1]:
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[pairs[k][0]] = avg
        i = j + 1
    return ranks


def _spearman(x: list[float], y: list[float]) -> float | None:
    return _pearson(_rank(x), _rank(y)) if len(x) == len(y) and len(x) >= 2 else None


def _mae(x: list[float], y: list[float]) -> float | None:
    return sum(abs(a - b) for a, b in zip(x, y)) / len(x) if x and len(x) == len(y) else None


def _bias(x: list[float], y: list[float]) -> float | None:
    return sum(a - b for a, b in zip(x, y)) / len(x) if x and len(x) == len(y) else None


def _fmt(v, d: int = 3) -> str:
    if v is None:
        return "NA"
    return str(v) if isinstance(v, int) else f"{v:.{d}f}"


# ══════════════════════════════════════════════════════════════════════════════
# Data builders
# ══════════════════════════════════════════════════════════════════════════════

def build_manual_scores(data: dict) -> pl.DataFrame:
    md  = data["motorical_development"]
    inf = data["introductory"]

    ms  = motorscore_milestones_setvalue(md, inf)
    imp = motorscore_impairments_setvalue(md, inf)
    com = motorscore_combined(ms, imp)

    return (
        ms
        .join(imp.select(["introductory_id", "age", "mms_normalized"]),
              on=["introductory_id", "age"], how="inner")
        .join(com.select(["introductory_id", "age", "combined_score"]),
              on=["introductory_id", "age"], how="inner")
        .sort(["introductory_id", "age"])
    )


def load_llm_results() -> pl.DataFrame:
    if not LLM_CSV.exists():
        raise FileNotFoundError(f"LLM results not found: {LLM_CSV}")
    df = pl.read_csv(LLM_CSV)
    required = {"introductory_id", "age", "llm_milestone_score",
                "llm_impairment_score", "llm_combined_score"}
    if missing := required - set(df.columns):
        raise ValueError(f"Missing columns in LLM CSV: {sorted(missing)}")
    return df.sort(["introductory_id", "age"])


def build_comparison_df(manual: pl.DataFrame, llm: pl.DataFrame) -> pl.DataFrame:
    return (
        manual
        .join(llm, on=["introductory_id", "age"], how="inner")
        .with_columns([
            (pl.col("llm_milestone_score")  - pl.col("milestone_score")).alias("delta_milestone"),
            (pl.col("llm_impairment_score") - pl.col("mms_normalized") ).alias("delta_impairment"),
            (pl.col("llm_combined_score")   - pl.col("combined_score") ).alias("delta_combined"),
            (pl.col("llm_milestone_score")  - pl.col("milestone_score")).abs().alias("abs_delta_milestone"),
            (pl.col("llm_impairment_score") - pl.col("mms_normalized") ).abs().alias("abs_delta_impairment"),
            (pl.col("llm_combined_score")   - pl.col("combined_score") ).abs().alias("abs_delta_combined"),
        ])
        .sort(["introductory_id", "age"])
    )


# ══════════════════════════════════════════════════════════════════════════════
# Aggregate summaries
# ══════════════════════════════════════════════════════════════════════════════

def _eval_pair(df: pl.DataFrame, llm_col: str, man_col: str, label: str) -> dict:
    sub = df.select([llm_col, man_col]).drop_nulls()
    x   = _clean(sub[llm_col])
    y   = _clean(sub[man_col])
    return dict(
        score_type            = label,
        n                     = len(x),
        pearson_r             = _pearson(x, y),
        spearman_rho          = _spearman(x, y),
        mae                   = _mae(x, y),
        bias_llm_minus_manual = _bias(x, y),
        mean_llm              = _mean(x),
        mean_manual           = _mean(y),
    )


def build_summary_df(cdf: pl.DataFrame) -> pl.DataFrame:
    return pl.DataFrame([
        _eval_pair(cdf, "llm_milestone_score",  "milestone_score", "milestone"),
        _eval_pair(cdf, "llm_impairment_score", "mms_normalized",  "impairment"),
        _eval_pair(cdf, "llm_combined_score",   "combined_score",  "combined"),
    ])


def build_age_summary(cdf: pl.DataFrame) -> pl.DataFrame:
    return (
        cdf
        .group_by("age")
        .agg([
            pl.len().alias("n"),
            pl.col("abs_delta_milestone") .mean().alias("mae_milestone"),
            pl.col("abs_delta_impairment").mean().alias("mae_impairment"),
            pl.col("abs_delta_combined")  .mean().alias("mae_combined"),
            pl.col("delta_milestone")     .mean().alias("bias_milestone"),
            pl.col("delta_impairment")    .mean().alias("bias_impairment"),
            pl.col("delta_combined")      .mean().alias("bias_combined"),
        ])
        .sort("age")
    )


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════

def make_scatter_grid(cdf: pl.DataFrame) -> None:
    """3 scatter plots in one figure."""
    pairs = [
        ("milestone_score", "llm_milestone_score", "Milestone"),
        ("mms_normalized",  "llm_impairment_score","Impairment"),
        ("combined_score",  "llm_combined_score",  "Combined"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle("LLM vs rule-based motor scores", fontsize=12, fontweight="bold", y=1.01)

    for ax, (man_col, llm_col, label) in zip(axes, pairs):
        sub = cdf.select([man_col, llm_col]).drop_nulls()
        x   = _clean(sub[man_col])
        y   = _clean(sub[llm_col])

        ax.scatter(x, y, alpha=0.75, color="#4C72B0",
                   edgecolors="white", linewidths=0.4, s=55, zorder=3)
        ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1.2, zorder=2)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(f"Rule-based ({label})", fontsize=9)
        ax.set_ylabel(f"LLM ({label})", fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.annotate(
            f"r = {_fmt(_pearson(x, y))}\nρ = {_fmt(_spearman(x, y))}",
            xy=(0.05, 0.93), xycoords="axes fraction", fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75),
        )

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "scatter_all_scores.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved scatter_all_scores.png")


def make_bland_altman(cdf: pl.DataFrame) -> None:
    """Bland-Altman for the combined score."""
    sub = (
        cdf.select(["llm_combined_score", "combined_score"]).drop_nulls()
        .with_columns([
            ((pl.col("llm_combined_score") + pl.col("combined_score")) / 2).alias("avg"),
            (pl.col("llm_combined_score") - pl.col("combined_score")).alias("diff"),
        ])
    )
    avgs  = _clean(sub["avg"])
    diffs = _clean(sub["diff"])
    md    = _mean(diffs)
    sd    = math.sqrt(sum((d - md)**2 for d in diffs) / (len(diffs) - 1)) if len(diffs) > 1 else 0.0
    lo, hi = md - 1.96 * sd, md + 1.96 * sd

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(avgs, diffs, color="#DD8452", alpha=0.75,
               edgecolors="white", linewidths=0.4, s=55, zorder=3)
    ax.axhline(md, color="#333333", linestyle="--", linewidth=1.3,
               label=f"Mean diff = {md:.3f}")
    ax.axhline(hi, color="#aaaaaa", linestyle=":", linewidth=1.1,
               label=f"+1.96 SD = {hi:.3f}")
    ax.axhline(lo, color="#aaaaaa", linestyle=":", linewidth=1.1,
               label=f"−1.96 SD = {lo:.3f}")
    ax.set_xlabel("Mean of LLM and rule-based combined score", fontsize=9)
    ax.set_ylabel("LLM − rule-based", fontsize=9)
    ax.set_title("Bland-Altman: combined score", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "bland_altman_combined.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved bland_altman_combined.png")


def make_mae_by_age(age_sum: pl.DataFrame) -> None:
    sub  = age_sum.select(["age", "mae_milestone", "mae_impairment", "mae_combined"]).sort("age")
    ages = sub["age"].to_list()

    fig, ax = plt.subplots(figsize=(6, 4))
    for col, label, color in [
        ("mae_milestone",  "Milestone",  "#4C72B0"),
        ("mae_impairment", "Impairment", "#DD8452"),
        ("mae_combined",   "Combined",   "#55A868"),
    ]:
        ax.plot(ages, _clean(sub[col]), marker="o", label=label, color=color, linewidth=1.8)

    ax.set_xlabel("Age bucket", fontsize=9)
    ax.set_ylabel("MAE", fontsize=9)
    ax.set_title("Mean absolute error by age", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "mae_by_age.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved mae_by_age.png")


def make_summary_table(sum_df: pl.DataFrame) -> None:
    headers   = ["Score", "n", "Pearson r", "Spearman ρ", "MAE", "Bias (LLM−rule)"]
    cell_text = [
        [
            row["score_type"], str(row["n"]),
            _fmt(row["pearson_r"]), _fmt(row["spearman_rho"]),
            _fmt(row["mae"]),       _fmt(row["bias_llm_minus_manual"]),
        ]
        for row in sum_df.iter_rows(named=True)
    ]

    fig, ax = plt.subplots(figsize=(9, 2.2))
    ax.axis("off")
    t = ax.table(cellText=cell_text, colLabels=headers, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1, 1.6)
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f7f9fb" if r % 2 == 0 else "white")
    ax.set_title("LLM vs rule-based — summary metrics", fontsize=11,
                 fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "summary_table.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved summary_table.png")


def make_outlier_tables(cdf: pl.DataFrame, top_n: int = 15) -> None:
    """
    For each score type, produce a ranked table of the rows with the largest
    absolute discrepancy between LLM and rule-based scores.

    Δ > 0 (red)  → LLM scored higher than rule-based
    Δ < 0 (green)→ LLM scored lower than rule-based

    The full introductory_id is preserved in the tooltip column so the analyst
    can cross-reference JOINED_CSV for deeper investigation.
    """
    score_configs = [
        ("abs_delta_combined",   "delta_combined",   "llm_combined_score",   "combined_score",  "Combined"),
        ("abs_delta_milestone",  "delta_milestone",  "llm_milestone_score",  "milestone_score", "Milestone"),
        ("abs_delta_impairment", "delta_impairment", "llm_impairment_score", "mms_normalized",  "Impairment"),
    ]

    for abs_col, delta_col, llm_col, man_col, label in score_configs:
        top = (
            cdf
            .select(["introductory_id", "age", llm_col, man_col, delta_col, abs_col])
            .sort(abs_col, descending=True)
            .head(top_n)
        )

        cell_text   = []
        cell_colors = []

        for row in top.iter_rows(named=True):
            delta     = row[delta_col]
            short_id  = str(row["introductory_id"])[:8] + "…"
            d_bg      = "#ffd6d6" if delta > 0 else "#d6f5d6"
            abs_bg    = "#f0f0f0"
            cell_text.append([
                short_id, str(row["age"]),
                _fmt(row[llm_col]), _fmt(row[man_col]),
                f"{delta:+.3f}", _fmt(row[abs_col]),
            ])
            cell_colors.append(["white", "white", "white", "white", d_bg, abs_bg])

        headers = ["ID (first 8)", "Age", f"LLM {label}", f"Rule {label}", "Δ (LLM−rule)", "|Δ|"]

        fig, ax = plt.subplots(figsize=(10, 0.45 * top_n + 1.8))
        ax.axis("off")
        t = ax.table(
            cellText=cell_text, colLabels=headers,
            cellColours=cell_colors, loc="center", cellLoc="center",
        )
        t.auto_set_font_size(False)
        t.set_fontsize(8.5)
        t.scale(1, 1.5)
        for (r, c), cell in t.get_celld().items():
            if r == 0:
                cell.set_facecolor("#2c3e50")
                cell.set_text_props(color="white", fontweight="bold")

        ax.set_title(
            f"Top {top_n} largest discrepancies — {label} score\n"
            f"(red = LLM scored higher, green = LLM scored lower)  "
            f"| Full IDs in {JOINED_CSV.name}",
            fontsize=9, fontweight="bold", pad=10,
        )
        plt.tight_layout()
        out = IMAGES_DIR / f"outliers_{label.lower()}.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved outliers_{label.lower()}.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("Loading data…")
    conn = get_connection()
    data = load_data(conn)

    manual_df = build_manual_scores(data)
    llm_df    = load_llm_results()

    print("Building comparison table…")
    cdf = build_comparison_df(manual_df, llm_df)
    if cdf.height == 0:
        raise ValueError("No overlapping rows — check that introductory_ids match.")

    print(f"Comparing {cdf.height} rows across {cdf['introductory_id'].n_unique()} children.\n")

    sum_df  = build_summary_df(cdf)
    age_sum = build_age_summary(cdf)

    # ── save joined CSV — useful for manual investigation ─────────────────────
    cdf.write_csv(JOINED_CSV)
    print(f"Joined CSV saved: {JOINED_CSV}")
    print("  Tip: sort by abs_delta_combined to find the largest discrepancies.\n")

    # ── figures ───────────────────────────────────────────────────────────────
    print("Building figures…")
    make_scatter_grid(cdf)
    make_bland_altman(cdf)
    make_mae_by_age(age_sum)
    make_summary_table(sum_df)
    make_outlier_tables(cdf, top_n=15)

    # ── console summary ────────────────────────────────────────────────────────
    print("\n── Summary ─────────────────────────────────────────────────────────")
    for row in sum_df.iter_rows(named=True):
        print(
            f"  {row['score_type']:12s}  "
            f"r={_fmt(row['pearson_r'])}  "
            f"ρ={_fmt(row['spearman_rho'])}  "
            f"MAE={_fmt(row['mae'])}  "
            f"bias={_fmt(row['bias_llm_minus_manual'])}"
        )

    print(f"\nAll figures → {IMAGES_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()