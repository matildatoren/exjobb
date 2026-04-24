# Important note:
# This is not an objective clinical measure, but an LLM-derived summary assessment.
# The LLM evaluates each child's full training history alongside motor outcomes
# and estimates whether training appears to be associated with motor progress.

from typing import List, Literal
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import time
import json
import re

import polars as pl
from ollama import chat
from pydantic import BaseModel

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # adjust if needed
sys.path.append(str(ROOT))

from connect_db import get_connection
from dataloader import load_data
from preprocessing.home_training import (
    process_home_training_hours_per_user_per_year,
    process_other_training_hours_per_user_per_year,
    extract_training_details,
)
from preprocessing.intensive_therapies import (
    process_neurohab_hours_per_user_per_age,
)


# ════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ════════════════════════════════════════════════════════════════════════════

MODEL_NAME = "gemma4:26b"

CHAT_TIMEOUT_SECONDS = 120   # longer timeout — full child timeline is bigger
CHAT_MAX_RETRIES = 2

INTRODUCTORY_IDS = [
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
]

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "training_effect_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_TXT_PATH = OUTPUT_DIR / "llm_training_effect_results.txt"
OUTPUT_CSV_PATH = OUTPUT_DIR / "llm_training_effect_results.csv"


# ════════════════════════════════════════════════════════════════════════════
# STRUCTURED OUTPUT
# ════════════════════════════════════════════════════════════════════════════

class TrainingEffectAssessment(BaseModel):
    training_effect_score: int          # 1–10: how strong is the apparent training effect
    trajectory: Literal[                # overall motor trajectory over the observed years
        "clearly_improving",
        "slightly_improving",
        "stable",
        "slightly_declining",
        "clearly_declining",
        "mixed_unclear",
    ]
    training_response: Literal[         # does training appear to correlate with progress
        "strong_positive",
        "moderate_positive",
        "unclear",
        "no_apparent_effect",
        "possible_negative",
    ]
    confidence: Literal["low", "medium", "high"]
    summary: str                        # 2–4 sentence English summary
    key_observations: List[str]         # bullet-point evidence list


# ════════════════════════════════════════════════════════════════════════════
# FEW-SHOT EXAMPLES
# ════════════════════════════════════════════════════════════════════════════

FEW_SHOT_EXAMPLES = [
    # Example A — clear improvement, intensive training
    {
        "input": """Evaluate training effect for this child's full history:

[Child Training & Motor Timeline]

GMFCS level: Level II – Walks with some limitations

--- Age 1 ---
Home training hours: 120 (NDT physiotherapy, bimanual training)
Intensive therapy hours: 40 (intensive NDT block)
Medical treatments: none
Gross motor milestones: rolls, sits independently, pulls to stand (3 milestones)
Fine motor milestones: reaches, pincer grasp (2 milestones)
Impairments (lower): spasticity 3, muscle_weakness 2
Impairments (upper): coordination 2
Story: "Barnet deltar aktivt i sjukgymnastik och visar god respons. Föräldrar tränar dagligen."

--- Age 2 ---
Home training hours: 180 (NDT physiotherapy, bimanual training, hydrotherapy)
Intensive therapy hours: 80 (two intensive blocks)
Medical treatments: none
Gross motor milestones: rolls, sits, pulls to stand, first steps, climbs stairs (5 milestones)
Fine motor milestones: reaches, pincer grasp, stacks blocks, scribbles (4 milestones)
Impairments (lower): spasticity 2, muscle_weakness 1
Impairments (upper): coordination 2
Story: "Barnet gör tydliga framsteg. Går några steg med stöd. Stark motivation."

--- Age 3 ---
Home training hours: 200 (NDT physiotherapy, bimanual training, hydrotherapy, strength training)
Intensive therapy hours: 100 (three blocks, including suit therapy)
Medical treatments: botox injection
Gross motor milestones: rolls, sits, pulls to stand, first steps, climbs stairs, runs, jumps (7 milestones)
Fine motor milestones: reaches, pincer grasp, stacks blocks, scribbles, copies lines, scissors (6 milestones)
Impairments (lower): spasticity 1, muscle_weakness 1
Impairments (upper): coordination 1
Story: "Barnet springer nu självständigt. Motoriken har förbättrats markant sedan botox. Fortsatt hög träningsintensitet."
""",
        "output": '{"training_effect_score": 9, "trajectory": "clearly_improving", "training_response": "strong_positive", "confidence": "high", "summary": "This child shows a clear and consistent motor improvement across ages 1–3, coinciding with high and increasing training intensity. Gross and fine motor milestones have nearly doubled each year. Spasticity and weakness have reduced markedly. The combination of regular physiotherapy, intensive blocks, and botox appears strongly associated with the observed gains.", "key_observations": ["Gross motor milestones increased from 3 to 7 over 3 years", "Fine motor milestones increased from 2 to 6 over 3 years", "Spasticity (lower) declined from 3 to 1 as training intensity rose", "Training hours increased from 120 to 200/year with parallel intensive therapy blocks", "Botox at age 3 coincides with parent-reported marked improvement in running", "Story confirms daily parental training and high engagement throughout"]}',
    },

    # Example B — no clear effect, low training, stable/declining
    {
        "input": """Evaluate training effect for this child's full history:

[Child Training & Motor Timeline]

GMFCS level: Level IV – Limited mobility, primarily uses a wheelchair

--- Age 1 ---
Home training hours: 10
Intensive therapy hours: 0
Medical treatments: none
Gross motor milestones: rolls (1 milestone)
Fine motor milestones: reaches (1 milestone)
Impairments (lower): spasticity 4, muscle_weakness 4
Impairments (upper): spasticity 3, coordination 4
Story: "Barnet är mycket påverkat. Begränsad tillgång till habilitering."

--- Age 2 ---
Home training hours: 8
Intensive therapy hours: 0
Medical treatments: none
Gross motor milestones: rolls (1 milestone)
Fine motor milestones: reaches, transfers (2 milestones)
Impairments (lower): spasticity 4, muscle_weakness 4
Impairments (upper): spasticity 4, coordination 4
Story: "Ingen förändring. Familjen saknar resurser för intensiv träning."

--- Age 3 ---
Home training hours: 12
Intensive therapy hours: 0
Medical treatments: none
Gross motor milestones: rolls (1 milestone)
Fine motor milestones: reaches, transfers (2 milestones)
Impairments (lower): spasticity 5, muscle_weakness 4
Impairments (upper): spasticity 4, coordination 5
Story: "Spasticiteten har ökat. Barnet behöver nu hjälp med alla förflyttningar."
""",
        "output": '{"training_effect_score": 2, "trajectory": "slightly_declining", "training_response": "no_apparent_effect", "confidence": "medium", "summary": "This child has received very little training across all three observed years, with no intensive therapy and minimal home training hours. Motor milestones have remained nearly static and lower-limb spasticity has increased. The absence of observable progress is consistent with insufficient training exposure rather than a negative training effect. Confidence is medium because we cannot distinguish natural disease progression from lack of training response.", "key_observations": ["Training hours extremely low across all years (8–12 hrs/year) and no intensive therapy", "Gross motor milestones unchanged at 1 across all three years", "Lower-body spasticity increased from 4 to 5 at age 3", "No medical treatments received", "Stories consistently describe resource limitations and caregiver burden", "Cannot conclude training is ineffective — dose has been too low to evaluate"]}',
    },
]


# ════════════════════════════════════════════════════════════════════════════
# TEXT / JSON HELPERS
# ════════════════════════════════════════════════════════════════════════════

def clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    return text


def extract_json_content(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text.removeprefix("```json").strip()
    elif text.startswith("```"):
        text = text.removeprefix("```").strip()
    if text.endswith("```"):
        text = text.removesuffix("```").strip()
    return text


def repair_json_text(text: str) -> str:
    text = text.strip()
    text = re.sub(
        r'"([A-Za-z0-9_]+):\s*(\[|\{|"|-|\d)',
        r'"\1": \2',
        text,
    )
    return text


# ════════════════════════════════════════════════════════════════════════════
# DATA ASSEMBLY — build per-child timeline
# ════════════════════════════════════════════════════════════════════════════

def _extract_training_method_names(struct) -> list[str]:
    """Return a list of training method names from a training_methods_therapies struct."""
    if not struct or not isinstance(struct, dict):
        return []
    details = struct.get("details", {})
    return [name for name in details.keys() if name]


def _count_milestones(struct) -> int:
    if not struct or not isinstance(struct, dict):
        return 0
    milestones = struct.get("milestones", [])
    return len(milestones) if milestones else 0


def _format_impairments(struct) -> str:
    """Format impairment details as 'key: value, ...' or 'none'."""
    if not struct or not isinstance(struct, dict):
        return "none"
    details = struct.get("details", {})
    if not details:
        return "none"
    parts = [f"{k}: {v}" for k, v in details.items() if v not in (None, 0, 0.0, "", "0")]
    return ", ".join(parts) if parts else "none"


def build_child_timeline(
    introductory_id: str,
    gmfcs_lvl: str,
    motorical_dev_df: pl.DataFrame,
    home_training_df: pl.DataFrame,
    neurohab_df: pl.DataFrame,
    medical_df: pl.DataFrame,
) -> str:
    """
    Build a formatted timeline string for one child covering all ages.

    Combines: GMFCS level, training hours (home / other / neurohab),
    training method names, medical treatments, motor milestones,
    impairments, and free-text stories.

    Returns a formatted string ready to embed in an LLM prompt.
    """
    # ── Pre-aggregate home training hours per (child, age) ──────────────────
    home_hours_wide = (
        home_training_df
        .filter(pl.col("introductory_id") == introductory_id)
    )

    # Raw training method names per (child, age)
    method_names_by_age: dict[int, list[str]] = {}
    for row in home_hours_wide.iter_rows(named=True):
        age = int(row["age"])
        names = _extract_training_method_names(row.get("training_methods_therapies"))
        names += _extract_training_method_names(row.get("other_training_methods_therapies"))
        method_names_by_age.setdefault(age, [])
        method_names_by_age[age].extend(names)

    # ── Home training hours per age ──────────────────────────────────────────
    home_h_by_age: dict[int, float] = {}
    for row in (
        home_hours_wide
        .with_columns(
            pl.Series(
                "h",
                [
                    sum(
                        float((info or {}).get("hours", 0) or 0)
                        * float((info or {}).get("weeks", 0) or 0)
                        for info in ((s or {}).get("details", {}) or {}).values()
                    )
                    for s in home_hours_wide["training_methods_therapies"].to_list()
                ]
            )
        )
        .group_by("age")
        .agg(pl.sum("h").alias("total_h"))
        .iter_rows(named=True)
    ):
        home_h_by_age[int(row["age"])] = round(row["total_h"], 1)

    # ── Neurohab hours per age ───────────────────────────────────────────────
    neuro_h_by_age: dict[int, float] = {}
    for row in (
        neurohab_df
        .filter(pl.col("introductory_id") == introductory_id)
        .group_by("age")
        .agg(pl.sum("total_hours").alias("total_h"))
        .iter_rows(named=True)
    ):
        neuro_h_by_age[int(row["age"])] = round(row["total_h"], 1)

    # ── Medical treatments per age ───────────────────────────────────────────
    med_by_age: dict[int, list[str]] = {}
    if medical_df is not None and medical_df.height > 0:
        child_med = medical_df.filter(pl.col("introductory_id") == introductory_id)
        treatment_cols = [c for c in child_med.columns if c not in ("introductory_id", "age")]
        for row in child_med.iter_rows(named=True):
            age = int(row["age"])
            treatments = [c for c in treatment_cols if row.get(c) and int(row[c]) > 0]
            med_by_age[age] = treatments

    # ── Motor development per age ─────────────────────────────────────────────
    md_child = (
        motorical_dev_df
        .filter(pl.col("introductory_id") == introductory_id)
        .sort("age")
    )

    # ── Assemble timeline ─────────────────────────────────────────────────────
    all_ages = sorted(set(
        list(home_h_by_age.keys())
        + list(neuro_h_by_age.keys())
        + md_child["age"].cast(pl.Int32).to_list()
    ))

    if not all_ages:
        return "(no data found for this child)"

    lines = [
        "[Child Training & Motor Timeline]",
        "",
        f"GMFCS level: {gmfcs_lvl or 'unknown'}",
    ]

    for age in all_ages:
        age_int = int(age)
        lines.append(f"\n--- Age {age_int} ---")

        # Training hours
        home_h = home_h_by_age.get(age_int, 0.0)
        neuro_h = neuro_h_by_age.get(age_int, 0.0)
        methods = list(dict.fromkeys(method_names_by_age.get(age_int, [])))  # deduplicated
        meds = med_by_age.get(age_int, [])

        lines.append(f"Home training hours: {home_h}" +
                     (f" ({', '.join(methods)})" if methods else ""))
        lines.append(f"Intensive therapy hours: {neuro_h}")
        lines.append(f"Medical treatments: {', '.join(meds) if meds else 'none'}")

        # Motor outcomes
        age_rows = md_child.filter(pl.col("age") == age)

        if age_rows.height == 0:
            lines.append("Motor data: not available for this age")
            continue

        # Take the first row for structured data (aggregate stories below)
        first_row = age_rows.row(0, named=True)

        gross_n = _count_milestones(first_row.get("gross_motor_development"))
        fine_n  = _count_milestones(first_row.get("fine_motor_development"))
        imp_lower = _format_impairments(first_row.get("motorical_impairments_lower"))
        imp_upper = _format_impairments(first_row.get("motorical_impairments_upper"))

        lines.append(f"Gross motor milestones: {gross_n}")
        lines.append(f"Fine motor milestones: {fine_n}")
        lines.append(f"Impairments (lower): {imp_lower}")
        lines.append(f"Impairments (upper): {imp_upper}")

        # Collect all non-empty stories for this age
        stories = [
            clean_text(r)
            for r in age_rows["story"].to_list()
            if clean_text(r)
        ]
        if stories:
            # Combine if multiple rows
            lines.append(f"Story: \"{' | '.join(stories)}\"")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# LLM CALL
# ════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
You are a clinical data analyst evaluating whether rehabilitation training has helped children with cerebral palsy improve their motor function.

TASK:
Given a child's full longitudinal timeline — covering training hours, training methods, medical treatments, and motor outcomes (milestones and impairments) across multiple ages — assess:
1. Whether the child's motor function has improved, stayed stable, or declined.
2. Whether training appears to be associated with motor progress.

IMPORTANT RULES:
- Base your assessment ONLY on the provided data. Do NOT invent or hallucinate information.
- Correlation is not causation. Be careful not to overstate causal claims.
- Low or absent training makes it impossible to evaluate training response — note this explicitly.
- Consider trajectory, dose, and timing when assessing the training–outcome relationship.
- Return raw JSON only. Do NOT wrap in markdown code fences.
- Use EXACTLY these field names:
  - training_effect_score
  - trajectory
  - training_response
  - confidence
  - summary
  - key_observations

SCALE — training_effect_score (1–10):
1–2 = No apparent training effect (minimal progress, very low dose, or steady decline)
3–4 = Possible small effect but weak signal
5–6 = Moderate apparent effect — some progress coinciding with training
7–8 = Fairly clear positive association between training and progress
9–10 = Strong, consistent positive association across multiple ages

TRAJECTORY options (pick one):
  clearly_improving | slightly_improving | stable | slightly_declining | clearly_declining | mixed_unclear

TRAINING_RESPONSE options (pick one):
  strong_positive | moderate_positive | unclear | no_apparent_effect | possible_negative

CONFIDENCE reflects data quality and consistency:
  low = sparse data, missing ages, contradictory signals
  medium = reasonable data but some gaps
  high = complete multi-year data with consistent signals

OUTPUT: Return ONLY valid JSON with the required fields.
LANGUAGE: Output must be in English regardless of input language.
"""


def analyze_child_timeline(
    timeline_text: str,
    model_name: str = MODEL_NAME,
) -> TrainingEffectAssessment:
    """
    Call the LLM with the full child timeline and return a structured assessment.
    Retries up to CHAT_MAX_RETRIES times on timeout.
    """
    user_prompt = f"Evaluate training effect for this child's full history:\n\n{timeline_text}"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for example in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user",      "content": example["input"]})
        messages.append({"role": "assistant", "content": example["output"]})

    messages.append({"role": "user", "content": user_prompt})

    def _call():
        return chat(
            model=model_name,
            messages=messages,
            format=TrainingEffectAssessment.model_json_schema(),
            options={"num_predict": 768},
        )

    for attempt in range(CHAT_MAX_RETRIES + 1):
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_call)
                response = future.result(timeout=CHAT_TIMEOUT_SECONDS)
            break
        except FuturesTimeoutError:
            print(
                f"    Timeout attempt {attempt + 1}/{CHAT_MAX_RETRIES + 1} "
                f"({CHAT_TIMEOUT_SECONDS}s), retrying..."
            )
            if attempt == CHAT_MAX_RETRIES:
                raise RuntimeError(
                    f"LLM timeout after {CHAT_MAX_RETRIES + 1} attempts"
                )

    raw = response.message.content
    cleaned = extract_json_content(raw)
    repaired = repair_json_text(cleaned)

    try:
        return TrainingEffectAssessment.model_validate_json(repaired)
    except Exception:
        print("\nRaw model output:")
        print(raw)
        print("\nRepaired output:")
        print(repaired)
        raise


# ════════════════════════════════════════════════════════════════════════════
# TEXT REPORT
# ════════════════════════════════════════════════════════════════════════════

def write_text_report(
    results_df: pl.DataFrame,
    output_path: Path,
    total_elapsed: float,
) -> None:
    lines = [
        "LLM TRAINING EFFECT ANALYSIS",
        "=" * 80,
        f"Total runtime: {total_elapsed:.2f} seconds",
        f"Children analysed: {results_df.height if results_df.height else 0}",
        "",
    ]

    if results_df.height == 0:
        lines.append("No results were generated.")
    else:
        for row in results_df.sort("training_effect_score", descending=True).iter_rows(named=True):
            lines += [
                f"Introductory ID     : {row['introductory_id']}",
                f"GMFCS level         : {row['gmfcs_lvl']}",
                f"Training effect score: {row['training_effect_score']} / 10",
                f"Trajectory          : {row['trajectory']}",
                f"Training response   : {row['training_response']}",
                f"Confidence          : {row['confidence']}",
                f"Summary             : {row['summary']}",
                f"Key observations    : {row['key_observations']}",
                "-" * 80,
            ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    total_start = time.perf_counter()

    conn = get_connection()
    data = load_data(conn)

    motorical_dev      = data["motorical_development"]
    home_training      = data["home_training"]
    intensive_therapies = data["intensive_therapies"]
    introductory       = data["introductory"]

    # Pre-process neurohab hours once
    neurohab_df = process_neurohab_hours_per_user_per_age(intensive_therapies)

    # Medical treatments (binary pivot)
    from preprocessing.intensive_therapies import process_medical_treatments_per_user_per_age
    medical_df = process_medical_treatments_per_user_per_age(intensive_therapies)

    # GMFCS lookup
    gmfcs_lookup: dict[str, str] = dict(
        zip(introductory["id"].to_list(), introductory["gmfcs_lvl"].to_list())
    )

    all_results: list[dict] = []

    for introductory_id in INTRODUCTORY_IDS:
        child_start = time.perf_counter()
        gmfcs_lvl = gmfcs_lookup.get(introductory_id, "unknown")
        print(f"\nProcessing: {introductory_id}  (GMFCS: {gmfcs_lvl})")

        try:
            timeline_text = build_child_timeline(
                introductory_id=introductory_id,
                gmfcs_lvl=gmfcs_lvl,
                motorical_dev_df=motorical_dev,
                home_training_df=home_training,
                neurohab_df=neurohab_df,
                medical_df=medical_df,
            )

            print(f"  Timeline built ({len(timeline_text)} chars). Sending to LLM...")

            result = analyze_child_timeline(timeline_text)

            elapsed = time.perf_counter() - child_start
            print(f"  Done in {elapsed:.2f}s  |  score={result.training_effect_score}  "
                  f"trajectory={result.trajectory}  response={result.training_response}")

            all_results.append({
                "introductory_id":       introductory_id,
                "gmfcs_lvl":             gmfcs_lvl,
                "training_effect_score": result.training_effect_score,
                "trajectory":            result.trajectory,
                "training_response":     result.training_response,
                "confidence":            result.confidence,
                "summary":               result.summary,
                "key_observations":      " | ".join(result.key_observations),
            })

        except RuntimeError as e:
            print(f"  SKIPPED (timeout): {e}")
        except Exception as e:
            print(f"  ERROR: {e}")

    results_df = pl.DataFrame(all_results) if all_results else pl.DataFrame()

    if results_df.height > 0:
        results_df.write_csv(OUTPUT_CSV_PATH)
        print(f"\nCSV saved → {OUTPUT_CSV_PATH}")

    total_elapsed = time.perf_counter() - total_start
    write_text_report(results_df, OUTPUT_TXT_PATH, total_elapsed)

    print(f"Text report saved → {OUTPUT_TXT_PATH}")
    print(f"Total runtime: {total_elapsed:.2f} seconds")


if __name__ == "__main__":
    main()