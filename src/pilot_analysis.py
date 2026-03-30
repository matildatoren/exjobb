"""
Pilot Study Analysis — Descriptive Statistics & Hypothesis Generation
======================================================================
Appropriate for n=10 children. Does NOT attempt predictive modelling.

Analyses:
  1. Summary statistics of all key variables
  2. Spearman correlations (training hours → delta milestone score) with CIs
  3. Individual child trajectory plots (milestone score over age)
  4. Botox group comparison (Mann-Whitney U + effect size)
  5. Effect size estimates → power analysis for a future study

Usage:
    cd /your/repo
    python pilot_analysis.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import spearmanr, mannwhitneyu

# ── resolve src/ ─────────────────────────────────────────────────────────────
SRC = Path(__file__).resolve().parent / "src"
sys.path.append(str(SRC))

from connect_db import get_connection
from dataloader import load_data
from preprocessing.master_preprocessing import build_master_feature_table

IMAGES_DIR = Path(__file__).resolve().parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)

BOTOX_KEYWORDS = ["botox", "botulinum", "botulin"]
TRAINING_COLS = [
    "total_home_training_hours",
    "total_other_training_hours",
    "neurohab_hours",
]
TARGET = "delta_milestone_score_setvalue"
SCORE  = "milestone_score_setvalue"


# ── helpers ───────────────────────────────────────────────────────────────────

def find_botox_col(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if col.startswith("med_") and any(kw in col.lower() for kw in BOTOX_KEYWORDS):
            return col
    return None


def spearman_ci(x: pd.Series, y: pd.Series, alpha: float = 0.05):
    """Spearman r with 95% CI via Fisher z-transform."""
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 4:
        return np.nan, np.nan, np.nan, n
    r, p = spearmanr(x, y)
    # Fisher z-transform
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    lo = np.tanh(z - z_crit * se)
    hi = np.tanh(z + z_crit * se)
    return r, lo, hi, n


def cohens_d(group1: pd.Series, group2: pd.Series) -> float:
    """Cohen's d for two independent groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    pooled_std = np.sqrt(
        ((n1 - 1) * group1.std() ** 2 + (n2 - 1) * group2.std() ** 2)
        / (n1 + n2 - 2)
    )
    if pooled_std == 0:
        return np.nan
    return (group1.mean() - group2.mean()) / pooled_std


def required_n_per_group(d: float, power: float = 0.80, alpha: float = 0.05) -> int:
    """
    Approximate sample size per group for a two-sample t-test
    using the formula: n ≈ 2 * (z_α/2 + z_β)² / d²
    """
    if np.isnan(d) or d == 0:
        return np.nan
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta  = stats.norm.ppf(power)
    n = 2 * ((z_alpha + z_beta) / d) ** 2
    return int(np.ceil(n))


def sep(title=""):
    w = 60
    if title:
        pad = (w - len(title) - 2) // 2
        print("=" * pad + f" {title} " + "=" * (w - pad - len(title) - 2))
    else:
        print("=" * w)


# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════

sep("Loading data")
conn = get_connection()
data = load_data(conn)
master = build_master_feature_table(data)
df = master.to_pandas()

botox_col = find_botox_col(df)
print(f"Botox column : {botox_col or 'NOT FOUND'}")
print(f"Master shape : {df.shape[0]} rows × {df.shape[1]} columns")
n_children = df["introductory_id"].nunique()
print(f"Unique children: {n_children}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

sep("Summary Statistics")

summary_cols = TRAINING_COLS + [TARGET, SCORE]
if botox_col:
    summary_cols.append(botox_col)

desc = df[summary_cols].describe().T
desc["missing"] = df[summary_cols].isna().sum()
print(desc.round(4).to_string())

if botox_col:
    print(f"\nBotox exposure (row-level):")
    print(df[botox_col].value_counts().to_string())
    botox_children = df.groupby("introductory_id")[botox_col].max()
    print(f"\nChildren ever receiving Botox: {int(botox_children.sum())} / {n_children}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. SPEARMAN CORRELATIONS WITH CONFIDENCE INTERVALS
# ═══════════════════════════════════════════════════════════════════════════

sep("Spearman Correlations → delta milestone score")

rows_with_target = df[TARGET].notna()
print(f"\nRows with delta score: {rows_with_target.sum()} / {len(df)}")
print(f"{'Feature':<45} {'r':>7} {'95% CI':>20} {'n':>6}")
print("-" * 82)

corr_results = []
for col in TRAINING_COLS:
    r, lo, hi, n = spearman_ci(df.loc[rows_with_target, col],
                                df.loc[rows_with_target, TARGET])
    p = spearmanr(df.loc[rows_with_target, col].dropna(),
                  df.loc[rows_with_target, TARGET].dropna())[1]
    ci_str = f"[{lo:+.3f}, {hi:+.3f}]" if not np.isnan(lo) else "n/a"
    sig = "*" if p < 0.05 else ""
    print(f"{col:<45} {r:>+7.3f} {ci_str:>20} {n:>6}  {sig}")
    corr_results.append({"feature": col, "r": r, "ci_lo": lo, "ci_hi": hi, "n": n, "p": p})

print("\n* p < 0.05")
print("\nNote: With n=10 children these are exploratory estimates only.")
print("      Confidence intervals are wide — do not over-interpret direction.")


# ── Correlation plot ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, len(TRAINING_COLS), figsize=(5 * len(TRAINING_COLS), 5))
if len(TRAINING_COLS) == 1:
    axes = [axes]

for ax, col in zip(axes, TRAINING_COLS):
    subset = df[[col, TARGET]].dropna()
    ax.scatter(subset[col], subset[TARGET],
               alpha=0.6, color="steelblue", edgecolors="white", s=60)

    # regression line (visual only, not inferential)
    if len(subset) > 2:
        m, b = np.polyfit(subset[col], subset[TARGET], 1)
        x_line = np.linspace(subset[col].min(), subset[col].max(), 100)
        ax.plot(x_line, m * x_line + b, "r--", linewidth=1.5, alpha=0.7)

    r_row = next((r for r in corr_results if r["feature"] == col), None)
    r_str = f"r = {r_row['r']:+.3f}" if r_row and not np.isnan(r_row["r"]) else ""
    ax.set_xlabel(col, fontsize=9)
    ax.set_ylabel("Δ milestone score (setvalue)")
    ax.set_title(f"{col}\n{r_str}", fontsize=9)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")

plt.suptitle("Training Hours vs Δ Milestone Score (Spearman, n=10 children)", fontsize=11)
plt.tight_layout()
path = IMAGES_DIR / "correlations_scatter.png"
plt.savefig(path, dpi=150)
print(f"\nSaved: {path.name}")
plt.show()


# ── Forest plot of correlations ───────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 3.5))
labels = [r["feature"].replace("total_", "").replace("_hours", "").replace("_", " ")
          for r in corr_results]
rs  = [r["r"]    for r in corr_results]
los = [r["ci_lo"] for r in corr_results]
his = [r["ci_hi"] for r in corr_results]
y_pos = range(len(labels))

for i, (r, lo, hi) in enumerate(zip(rs, los, his)):
    color = "steelblue" if not np.isnan(r) else "gray"
    ax.plot([lo, hi], [i, i], color=color, linewidth=2.5)
    ax.plot(r, i, "o", color=color, markersize=8)

ax.axvline(0, color="black", linewidth=1, linestyle="--")
ax.set_yticks(list(y_pos))
ax.set_yticklabels(labels)
ax.set_xlabel("Spearman r  (95% CI)")
ax.set_title("Correlation with Δ Milestone Score — Forest Plot\n(wide CIs expected at n=10)")
ax.set_xlim(-1.05, 1.05)
plt.tight_layout()
path = IMAGES_DIR / "correlation_forest_plot.png"
plt.savefig(path, dpi=150)
print(f"Saved: {path.name}")
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# 4. INDIVIDUAL CHILD TRAJECTORIES
# ═══════════════════════════════════════════════════════════════════════════

sep("Individual Trajectories")

children = df["introductory_id"].unique()
n_children = len(children)
ncols = min(5, n_children)
nrows = int(np.ceil(n_children / ncols))

fig = plt.figure(figsize=(4.5 * ncols, 4 * nrows))
gs  = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.5, wspace=0.4)

for idx, child_id in enumerate(sorted(children)):
    child_df = df[df["introductory_id"] == child_id].sort_values("age")
    row, col = divmod(idx, ncols)
    ax = fig.add_subplot(gs[row, col])

    # milestone score trajectory
    valid = child_df[[SCORE, "age"]].dropna()
    if not valid.empty:
        ax.plot(valid["age"], valid[SCORE],
                "o-", color="steelblue", linewidth=2, markersize=5,
                label="milestone score")

    # shade years with Botox
    if botox_col:
        botox_years = child_df[child_df[botox_col] == 1]["age"]
        for yr in botox_years:
            ax.axvspan(yr - 0.4, yr + 0.4, alpha=0.2, color="orange",
                       label="Botox" if yr == botox_years.iloc[0] else "")

    # training hours on secondary axis
    ax2 = ax.twinx()
    for t_col, color in zip(TRAINING_COLS, ["#2ca02c", "#d62728", "#9467bd"]):
        valid_t = child_df[[t_col, "age"]].dropna()
        if not valid_t.empty and valid_t[t_col].sum() > 0:
            ax2.bar(valid_t["age"], valid_t[t_col],
                    alpha=0.25, color=color, width=0.3,
                    label=t_col.replace("total_", "").replace("_hours", ""))
    ax2.set_ylabel("Training hrs", fontsize=6)
    ax2.tick_params(labelsize=6)

    short_id = str(child_id)[:8]
    ax.set_title(f"Child {short_id}…", fontsize=8)
    ax.set_xlabel("Age (years)", fontsize=7)
    ax.set_ylabel("Milestone score", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_ylim(0, 1.05)

    # legend only on first panel
    if idx == 0:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc="upper left")

fig.suptitle("Individual Motor Milestone Trajectories\n"
             "(blue line = score, bars = training hours, orange = Botox year)",
             fontsize=11)
path = IMAGES_DIR / "individual_trajectories.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"Saved: {path.name}")
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# 5. BOTOX GROUP COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

if botox_col:
    sep("Botox Group Comparison")

    comp = df[[botox_col, TARGET]].dropna(subset=[TARGET])
    botox_yes = comp[comp[botox_col] == 1][TARGET]
    botox_no  = comp[comp[botox_col] == 0][TARGET]

    print(f"\n  Botox = YES : n={len(botox_yes)}, "
          f"mean={botox_yes.mean():.4f}, median={botox_yes.median():.4f}, "
          f"std={botox_yes.std():.4f}")
    print(f"  Botox = NO  : n={len(botox_no)}, "
          f"mean={botox_no.mean():.4f}, median={botox_no.median():.4f}, "
          f"std={botox_no.std():.4f}")

    mean_diff = botox_yes.mean() - botox_no.mean()
    d = cohens_d(botox_yes, botox_no)

    if len(botox_yes) >= 2 and len(botox_no) >= 2:
        stat, p = mannwhitneyu(botox_yes, botox_no, alternative="two-sided")
        print(f"\n  Mann-Whitney U = {stat:.1f},  p = {p:.4f}")
    else:
        p = np.nan
        print("\n  ⚠  Too few observations per group for Mann-Whitney U.")

    print(f"  Mean difference (Botox - No Botox): {mean_diff:+.4f}")
    print(f"  Cohen's d (effect size)            : {d:+.4f}")

    if not np.isnan(d):
        d_abs = abs(d)
        if d_abs < 0.2:
            label = "negligible"
        elif d_abs < 0.5:
            label = "small"
        elif d_abs < 0.8:
            label = "medium"
        else:
            label = "large"
        print(f"  Effect size interpretation         : {label}")

    print(f"\n  ⚠  With row-level grouping (not child-level) these results")
    print(f"     are descriptive only — rows within a child are not independent.")

    # ── Boxplot ──────────────────────────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(6, 5))
    data_groups = [botox_no, botox_yes]
    bp = ax.boxplot(data_groups, patch_artist=True, widths=0.5,
                    medianprops={"color": "black", "linewidth": 2})
    bp["boxes"][0].set_facecolor("#AED6F1")
    bp["boxes"][1].set_facecolor("#A9DFBF")

    # overlay individual points
    for i, grp in enumerate([botox_no, botox_yes], start=1):
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(grp))
        ax.scatter(np.full(len(grp), i) + jitter, grp,
                   color="black", s=20, alpha=0.5, zorder=5)

    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"No Botox\n(n={len(botox_no)})",
                        f"Botox\n(n={len(botox_yes)})"])
    ax.set_ylabel("Δ Milestone Score (setvalue)")
    ax.set_title(f"Δ Milestone Score by Botox Exposure\n"
                 f"Cohen's d = {d:+.3f}  ({label}),  p = {p:.3f}")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    plt.tight_layout()
    path = IMAGES_DIR / "botox_comparison.png"
    plt.savefig(path, dpi=150)
    print(f"\nSaved: {path.name}")
    plt.show()
else:
    d = np.nan
    print("\nNo Botox column found — skipping group comparison.")


# ═══════════════════════════════════════════════════════════════════════════
# 6. POWER ANALYSIS — HOW MANY CHILDREN FOR A FUTURE STUDY?
# ═══════════════════════════════════════════════════════════════════════════

sep("Power Analysis — Required Sample Size")

print("\nBased on effect sizes estimated from this pilot data:")
print(f"{'Scenario':<35} {'d':>7} {'N per group':>12} {'Total N':>10}")
print("-" * 66)

scenarios = []

# From Botox comparison
if not np.isnan(d):
    scenarios.append(("Botox vs No-Botox (observed d)", d))

# Hypothetical small/medium effects for training hours
scenarios += [
    ("Small effect  (d=0.20) — training", 0.20),
    ("Medium effect (d=0.50) — training", 0.50),
    ("Large effect  (d=0.80) — training", 0.80),
]

for label_s, d_val in scenarios:
    n_pg = required_n_per_group(abs(d_val))
    total = n_pg * 2 if not np.isnan(n_pg) else np.nan
    d_str   = f"{d_val:+.3f}" if not np.isnan(d_val) else "n/a"
    n_str   = str(n_pg)   if not np.isnan(n_pg)   else "n/a"
    tot_str = str(total)  if not np.isnan(total)   else "n/a"
    print(f"{label_s:<35} {d_str:>7} {n_str:>12} {tot_str:>10}")

print(f"\nAssumptions: two-sided α=0.05, 80% power, two-sample comparison.")
print(f"For a repeated-measures design (children over time) the required N")
print(f"will be somewhat lower — but at minimum 50–70 unique children is")
print(f"recommended for a medium effect size in this context.")

# ── Power curve plot ──────────────────────────────────────────────────────────

d_range = np.linspace(0.1, 1.5, 200)
n_range = [required_n_per_group(d_val) for d_val in d_range]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(d_range, n_range, color="steelblue", linewidth=2.5)
ax.axhline(10, color="red",    linewidth=1.5, linestyle="--", label="Current n=10")
ax.axhline(50, color="orange", linewidth=1.5, linestyle="--", label="Target n=50")

# mark observed Botox d
if not np.isnan(d):
    n_botox = required_n_per_group(abs(d))
    ax.axvline(abs(d), color="green", linewidth=1.5, linestyle=":",
               label=f"Observed Botox d={abs(d):.2f} → n={n_botox}")
    ax.scatter([abs(d)], [n_botox], color="green", s=80, zorder=5)

ax.set_xlabel("Effect size (Cohen's d)")
ax.set_ylabel("Required children per group")
ax.set_title("Power Curve — Children Needed per Group\n"
             "(two-sided α=0.05, 80% power)")
ax.set_ylim(0, 300)
ax.set_xlim(0.1, 1.5)
ax.legend()
ax.fill_between(d_range, 0, n_range,
                where=[n <= 50 for n in n_range],
                alpha=0.1, color="green", label="Feasible zone")
plt.tight_layout()
path = IMAGES_DIR / "power_curve.png"
plt.savefig(path, dpi=150)
print(f"\nSaved: {path.name}")
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════════════════

sep()
print("Pilot analysis complete.")
print(f"All plots saved to: {IMAGES_DIR}")
print()
print("Key takeaways for your thesis:")
print("  1. This is a pilot study — frame it as hypothesis generation.")
print("  2. Report Spearman r values with 95% CIs, not p-values alone.")
print("  3. The Botox Cohen's d gives you a concrete effect size estimate")
print("     to justify a target sample size in a future study.")
print("  4. Individual trajectories reveal heterogeneity that group")
print("     statistics cannot capture — include them in the thesis.")
sep()