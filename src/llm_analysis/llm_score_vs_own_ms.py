"""
llm_score_vs_own_ms.py
======================
Compares LLM-derived motor scores against the project's rule-based scores,
and visualises what the free-text story contributed beyond the checkboxes.

Kept figures:
  - summary_table
  - outliers_combined / outliers_milestone / outliers_impairment

New figures:
  - before_after_cards   : top story-driven adjustments shown as readable cards
  - delta_distribution   : histogram of LLM − struct-only delta per score type
  - scatter_adjustments  : struct-only (x) vs LLM (y), coloured by adjustment direction
"""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
JOINED_CSV = OUTPUT_DIR / "llm_vs_manual_joined.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         13,
    "axes.titlesize":    16,
    "axes.labelsize":    14,
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
    "legend.fontsize":   12,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        150,
})


# ══════════════════════════════════════════════════════════════════════════════
# Statistics helpers
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
    required = {"introductory_id", "age",
                "llm_milestone_score", "llm_impairment_score", "llm_combined_score",
                "struct_milestone_score", "struct_impairment_score", "struct_combined_score",
                "story_adjustments", "est_cum_milestones", "est_n_selected", "est_sum_ratings"}
    if missing := required - set(df.columns):
        raise ValueError(f"Missing columns in LLM CSV: {sorted(missing)}")
    return df.sort(["introductory_id", "age"])


def build_comparison_df(manual: pl.DataFrame, llm: pl.DataFrame) -> pl.DataFrame:
    return (
        manual
        .join(llm, on=["introductory_id", "age"], how="inner")
        .with_columns([
            # LLM (story-adjusted) vs rule-based
            (pl.col("llm_milestone_score")  - pl.col("milestone_score")).alias("delta_milestone"),
            (pl.col("llm_impairment_score") - pl.col("mms_normalized") ).alias("delta_impairment"),
            (pl.col("llm_combined_score")   - pl.col("combined_score") ).alias("delta_combined"),
            (pl.col("llm_milestone_score")  - pl.col("milestone_score")).abs().alias("abs_delta_milestone"),
            (pl.col("llm_impairment_score") - pl.col("mms_normalized") ).abs().alias("abs_delta_impairment"),
            (pl.col("llm_combined_score")   - pl.col("combined_score") ).abs().alias("abs_delta_combined"),
            # Story contribution: LLM vs struct-only (checkboxes alone)
            (pl.col("llm_milestone_score")  - pl.col("struct_milestone_score") ).alias("story_delta_milestone"),
            (pl.col("llm_impairment_score") - pl.col("struct_impairment_score")).alias("story_delta_impairment"),
            (pl.col("llm_combined_score")   - pl.col("struct_combined_score")  ).alias("story_delta_combined"),
        ])
        .sort(["introductory_id", "age"])
    )


# ══════════════════════════════════════════════════════════════════════════════
# Summary metrics
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


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 (kept): Summary table
# ══════════════════════════════════════════════════════════════════════════════

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
    ax.set_title("LLM vs Rule-based — Summary Metrics", fontsize=16,
                 fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "summary_table.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved summary_table.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figures 2–4 (kept): Outlier tables
# ══════════════════════════════════════════════════════════════════════════════

def make_outlier_tables(cdf: pl.DataFrame, top_n: int = 15) -> None:
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
            delta    = row[delta_col]
            d_bg     = "#ffd6d6" if delta > 0 else "#d6f5d6"
            cell_text.append([
                str(row["age"]),
                _fmt(row[llm_col]), _fmt(row[man_col]),
                f"{delta:+.3f}", _fmt(row[abs_col]),
            ])
            cell_colors.append(["white", "white", "white", d_bg, "#f0f0f0"])

        headers = ["Age", f"LLM {label}", f"Rule {label}", "Δ (LLM−rule)", "|Δ|"]

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
            f"(red = LLM scored higher, green = LLM scored lower)",
            fontsize=14, fontweight="bold", pad=10,
        )
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / f"outliers_{label.lower()}.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved outliers_{label.lower()}.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 (new): Before/after cards — top story-driven adjustments
# ══════════════════════════════════════════════════════════════════════════════

def make_before_after_cards(cdf: pl.DataFrame, n_cards: int = 8) -> None:
    """
    Show the rows where the story caused the largest score change
    (|story_delta_combined| is largest), displayed as readable cards.

    Each card shows:
      - Child ID (short) and age
      - Checkbox-only score  vs  story-adjusted score
      - The story_adjustments text (what the LLM actually changed)
    """
    # Filter rows where story actually changed something
    changed = cdf.filter(pl.col("story_adjustments") != "")

    if changed.height == 0:
        print("  No story adjustments found — skipping before/after cards.")
        return

    # Absolute combined delta caused by story
    changed = changed.with_columns(
        pl.col("story_delta_combined").abs().alias("abs_story_delta")
    )
    top = changed.sort("abs_story_delta", descending=True).head(n_cards)

    n_cols = 2
    n_rows = math.ceil(n_cards / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.6))
    axes = axes.flatten()
    fig.suptitle(
        "Story-driven adjustments — top cases where free text changed the score",
        fontsize=16, fontweight="bold", y=1.01,
    )

    GREEN = "#27ae60"
    RED   = "#c0392b"
    GREY  = "#7f8c8d"

    for i, row in enumerate(top.iter_rows(named=True)):
        ax = axes[i]
        ax.axis("off")

        age         = row["age"]
        s_ms        = row["struct_milestone_score"]
        s_imp       = row["struct_impairment_score"]
        s_com       = row["struct_combined_score"]
        l_ms        = row["llm_milestone_score"]
        l_imp       = row["llm_impairment_score"]
        l_com       = row["llm_combined_score"]
        delta_com   = row["story_delta_combined"]
        adjustments = str(row["story_adjustments"] or "")

        # Card background
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, 0), 1, 1, transform=ax.transAxes, clip_on=False,
            boxstyle="round,pad=0.02", linewidth=1.2,
            edgecolor="#bdc3c7", facecolor="#fafafa",
        ))

        # Header
        header_color = GREEN if delta_com > 0 else RED
        direction    = "▲ Story raised score" if delta_com > 0 else "▼ Story lowered score"
        ax.text(0.5, 0.95, f"Age: {age}  |  {direction}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=9, fontweight="bold", color=header_color)

        # Score comparison row
        score_y = 0.80
        for col_x, lbl, sv, lv in [
            (0.18, "Milestone",  s_ms,  l_ms),
            (0.50, "Impairment", s_imp, l_imp),
            (0.82, "Combined",   s_com, l_com),
        ]:
            d = lv - sv
            d_col = GREEN if d > 0 else (RED if d < 0 else GREY)
            ax.text(col_x, score_y + 0.06, lbl, transform=ax.transAxes,
                    ha="center", va="top", fontsize=7.5, color=GREY)
            ax.text(col_x, score_y - 0.01,
                    f"Checkbox: {sv:.2f}\nStory-adj: {lv:.2f}",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=8.5, color="#2c3e50")
            ax.text(col_x, score_y - 0.18, f"Δ {d:+.2f}",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=9, fontweight="bold", color=d_col)

        # Divider
        ax.axhline(0.55, xmin=0.03, xmax=0.97, color="#bdc3c7", linewidth=0.8)

        # Story adjustments text
        # Wrap each pipe-separated bullet into its own line
        bullets = [a.strip() for a in adjustments.split("|") if a.strip()]
        max_bullets = 4
        display = bullets[:max_bullets]
        if len(bullets) > max_bullets:
            display.append(f"… +{len(bullets) - max_bullets} more")

        text_block = "\n".join(f"• {b}" for b in display)
        ax.text(0.03, 0.50, text_block,
                transform=ax.transAxes, ha="left", va="top",
                fontsize=7.5, color="#2c3e50",
                wrap=True, linespacing=1.5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0))

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "before_after_cards.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved before_after_cards.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 (new): Delta distribution — how much did the story change scores?
# ══════════════════════════════════════════════════════════════════════════════

def make_delta_distribution(cdf: pl.DataFrame) -> None:
    """
    Histogram of (LLM − struct-only) for each score type.
    Distribution centred on 0 → story changes little.
    Wide spread → story carries real information.
    """
    cols = [
        ("story_delta_milestone",  "Milestone",  "#4C72B0"),
        ("story_delta_impairment", "Impairment", "#DD8452"),
        ("story_delta_combined",   "Combined",   "#55A868"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    fig.suptitle(
        "Story contribution: distribution of (story-adjusted − checkbox-only) score",
        fontsize=16, fontweight="bold", y=1.02,
    )

    for ax, (col, label, color) in zip(axes, cols):
        vals = _clean(cdf[col])
        if not vals:
            ax.set_visible(False)
            continue

        n_bins = min(25, max(10, len(vals) // 4))
        ax.hist(vals, bins=n_bins, color=color, alpha=0.82, edgecolor="white", linewidth=0.6)
        ax.axvline(0,        color="#333333", linewidth=1.4, linestyle="--", label="No change")
        ax.axvline(_mean(vals), color="#c0392b", linewidth=1.4, linestyle="-",
                   label=f"Mean = {_mean(vals):+.3f}")

        # Annotate proportion that changed
        n_changed   = sum(1 for v in vals if abs(v) > 0.01)
        pct_changed = 100 * n_changed / len(vals)
        ax.annotate(
            f"{pct_changed:.0f}% of rows\nadjusted by story",
            xy=(0.97, 0.97), xycoords="axes fraction", ha="right", va="top",
            fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )

        ax.set_xlabel("Score change from story")
        ax.set_ylabel("Count")
        ax.set_title(label, fontweight="bold")
        ax.legend()

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "delta_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved delta_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 (new): Scatter struct-only vs LLM, coloured by adjustment direction
# ══════════════════════════════════════════════════════════════════════════════

def make_scatter_adjustments(cdf: pl.DataFrame) -> None:
    """
    X = struct-only (checkbox) score
    Y = LLM story-adjusted score
    Colour:
      green  → story raised the score  (delta > +0.02)
      red    → story lowered the score (delta < -0.02)
      grey   → story changed nothing   (|delta| ≤ 0.02)
    """
    score_triples = [
        ("struct_milestone_score",  "llm_milestone_score",  "story_delta_milestone",  "Milestone"),
        ("struct_impairment_score", "llm_impairment_score", "story_delta_impairment", "Impairment"),
        ("struct_combined_score",   "llm_combined_score",   "story_delta_combined",   "Combined"),
    ]

    THRESHOLD = 0.02
    COL_UP   = "#27ae60"   # story added information → score went up
    COL_DOWN = "#c0392b"   # story corrected downward
    COL_NONE = "#95a5a6"   # no meaningful change

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(
        "Checkbox-only score (x) vs story-adjusted score (y)\n"
        "Colour shows direction of story-driven adjustment",
        fontsize=16, fontweight="bold", y=1.03,
    )

    for ax, (struct_col, llm_col, delta_col, label) in zip(axes, score_triples):
        sub = cdf.select([struct_col, llm_col, delta_col]).drop_nulls()

        xs     = _clean(sub[struct_col])
        ys     = _clean(sub[llm_col])
        deltas = _clean(sub[delta_col])

        colors = [
            COL_UP   if d >  THRESHOLD else
            COL_DOWN if d < -THRESHOLD else
            COL_NONE
            for d in deltas
        ]

        n_up   = sum(1 for d in deltas if d >  THRESHOLD)
        n_down = sum(1 for d in deltas if d < -THRESHOLD)
        n_none = len(deltas) - n_up - n_down

        ax.scatter(xs, ys, c=colors, alpha=0.80, s=55,
                   edgecolors="white", linewidths=0.4, zorder=3)
        ax.plot([0, 1], [0, 1], color="#aaaaaa", linestyle="--",
                linewidth=1.1, zorder=2, label="No change line")

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Checkbox-only score")
        ax.set_ylabel("Story-adjusted score")
        ax.set_title(label, fontweight="bold")

        legend_handles = [
            mpatches.Patch(color=COL_UP,   label=f"Story raised  (n={n_up})"),
            mpatches.Patch(color=COL_DOWN, label=f"Story lowered (n={n_down})"),
            mpatches.Patch(color=COL_NONE, label=f"Unchanged     (n={n_none})"),
        ]
        ax.legend(handles=legend_handles, loc="upper left")

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "scatter_adjustments.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved scatter_adjustments.png")


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

    sum_df = build_summary_df(cdf)

    # Save joined CSV for manual investigation
    cdf.write_csv(JOINED_CSV)
    print(f"Joined CSV saved: {JOINED_CSV}\n")

    print("Building figures…")
    make_summary_table(sum_df)
    make_outlier_tables(cdf, top_n=15)
    make_before_after_cards(cdf, n_cards=8)
    make_delta_distribution(cdf)
    make_scatter_adjustments(cdf)

    # Console summary
    print("\n── Summary ─────────────────────────────────────────────────────────")
    for row in sum_df.iter_rows(named=True):
        print(
            f"  {row['score_type']:12s}  "
            f"r={_fmt(row['pearson_r'])}  "
            f"ρ={_fmt(row['spearman_rho'])}  "
            f"MAE={_fmt(row['mae'])}  "
            f"bias={_fmt(row['bias_llm_minus_manual'])}"
        )

    # Story contribution summary
    print("\n── Story contribution ───────────────────────────────────────────────")
    for col, label in [
        ("story_delta_milestone",  "Milestone"),
        ("story_delta_impairment", "Impairment"),
        ("story_delta_combined",   "Combined"),
    ]:
        vals      = _clean(cdf[col])
        n_changed = sum(1 for v in vals if abs(v) > 0.01)
        print(
            f"  {label:12s}  mean Δ={_fmt(_mean(vals))}  "
            f"rows changed={n_changed}/{len(vals)} ({100*n_changed//len(vals)}%)"
        )

    print(f"\nAll figures → {IMAGES_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()