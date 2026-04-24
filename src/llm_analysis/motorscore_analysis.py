# Important note:
# This is not an objective clinical measure, but an LLM-derived summary score
# intentionally designed to be comparable to the project's rule-based motor scores.

from typing import List, Literal
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import time

import polars as pl
from ollama import chat
from pydantic import BaseModel, Field

import sys
from pathlib import Path

import json
import re

# Resolve imports regardless of where the script is called from
ROOT = Path(__file__).resolve().parents[1]  # src/
sys.path.append(str(ROOT))

from connect_db import get_connection
from dataloader import load_data


# ---------------------------
# SETTINGS
# ---------------------------

MODEL_NAME = "gemma4:26b"

CHAT_TIMEOUT_SECONDS = 60
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

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "motorscore_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_TXT_PATH = OUTPUT_DIR / "llm_motorscore_results.txt"
OUTPUT_CSV_PATH = OUTPUT_DIR / "llm_motorscore_results.csv"


# ---------------------------
# STRUCTURED OUTPUT
# ---------------------------

class MotorAssessment(BaseModel):
    llm_milestone_score: float = Field(ge=0.0, le=1.0)
    llm_impairment_score: float = Field(ge=0.0, le=1.0)
    llm_combined_score: float = Field(ge=0.0, le=1.0)
    confidence: Literal["low", "medium", "high"]
    summary: str
    supporting_evidence: List[str]


# ---------------------------
# FEW-SHOT EXAMPLES
# ---------------------------

FEW_SHOT_EXAMPLES = [
    # Example 1 — Very high function
    {
        "input": """Assess motor development for this record:

[Motorical Development Record]

Age: 4

Gross motor development:
{"milestones": [{"id": "rolls_both_directions"}, {"id": "pushes_up_straight_elbows"}, {"id": "sits_independently"}, {"id": "crawls_or_scoots"}, {"id": "stands_with_support"}, {"id": "pulls_to_stand_cruises"}, {"id": "first_independent_steps"}, {"id": "climbs_stairs_hands_knees"}, {"id": "squats_and_stands"}, {"id": "runs_and_climbs_furniture"}, {"id": "jumps_both_feet"}, {"id": "walks_stairs_one_foot_per_step"}, {"id": "balances_one_foot_briefly"}, {"id": "pedals_tricycle_hops"}, {"id": "throws_overhand_skips_hops"}, {"id": "balances_one_foot_5_seconds"}, {"id": "rides_bike_training_wheels"}, {"id": "hops_forward_rides_two_wheel_bike"}, {"id": "hops_jumps_confidently_10_seconds"}, {"id": "jumps_rope_sports"}, {"id": "runs_smoothly_throws_catches"}]}

Fine motor development:
{"milestones": [{"id": "reaches_grasps_both_hands"}, {"id": "transfers_hand_to_hand"}, {"id": "bangs_objects_points"}, {"id": "pincer_grasp"}, {"id": "places_in_containers_self_feeds_fingers"}, {"id": "turns_pages_several"}, {"id": "stacks_blocks_uses_spoon"}, {"id": "scribbles_stacks_6_8_blocks"}, {"id": "copies_lines_circles_spoon_fork"}, {"id": "turns_single_pages_copies_crosses"}, {"id": "scissors_large_buttons"}, {"id": "draws_simple_person"}, {"id": "copies_squares_attempts_letters"}, {"id": "fork_spoon_pencil_grasp"}]}

Motorical impairments (lower):
{"details": {}}

Motorical impairments (upper):
{"details": {}}

Story:
The child's motor development has consistently been at or above the expected level for their age. No motor concerns have been raised by parents, physiotherapists, or preschool staff. The child participates fully in all physical activities with peers, including running games, climbing, and ball sports.""",

        "output": """{
  "llm_milestone_score": 0.98,
  "llm_impairment_score": 1.0,
  "llm_combined_score": 0.99,
  "confidence": "high",
  "summary": "The child demonstrates near-maximal milestone attainment and no reported impairments.",
  "supporting_evidence": [
    "All or nearly all gross motor milestones are achieved",
    "All or nearly all fine motor milestones are achieved",
    "No lower-body impairments are reported",
    "No upper-body impairments are reported",
    "The story confirms full participation in age-expected physical activities"
  ]
}""",
    },

    # Example 2 — Very low function
    {
        "input": """Assess motor development for this record:

[Motorical Development Record]

Age: 4

Gross motor development:
{"milestones": []}

Fine motor development:
{"milestones": []}

Motorical impairments (lower):
{"details": {"spasticity": 5, "muscle_weakness": 5, "range_of_motion": 5}}

Motorical impairments (upper):
{"details": {"spasticity": 5, "muscle_weakness": 5, "coordination": 5}}

Story:
Barnet har spastisk tetrapares och är helt beroende av omvårdnad för alla förflyttningar och aktiviteter. Inga funktionella motoriska färdigheter har utvecklats. Barnet kan inte sitta utan stöd och har aldrig stått eller gått. Finmotoriken är extremt begränsad – inget grepp, ingen pekförmåga och inget självständigt hanterande av föremål. Intensiv habilitering pågår utan tydlig funktionell framgång hittills.""",

        "output": """{
  "llm_milestone_score": 0.03,
  "llm_impairment_score": 0.05,
  "llm_combined_score": 0.04,
  "confidence": "high",
  "summary": "The child shows extremely limited milestone attainment and very severe impairment burden.",
  "supporting_evidence": [
    "No gross motor milestones are reported",
    "No fine motor milestones are reported",
    "Lower-body impairments are severe across all reported domains",
    "Upper-body impairments are severe across all reported domains",
    "The story confirms complete dependence and absence of functional motor development"
  ]
}""",
    },

    # Example 3 — Moderate to good function
    {
        "input": """Assess motor development for this record:

[Motorical Development Record]

Age: 3

Gross motor development:
{"milestones": [{"id": "rolls_both_directions"}, {"id": "pushes_up_straight_elbows"}, {"id": "sits_independently"}, {"id": "crawls_or_scoots"}, {"id": "stands_with_support"}, {"id": "pulls_to_stand_cruises"}, {"id": "first_independent_steps"}, {"id": "climbs_stairs_hands_knees"}, {"id": "squats_and_stands"}, {"id": "runs_and_climbs_furniture"}, {"id": "jumps_both_feet"}, {"id": "walks_stairs_one_foot_per_step"}]}

Fine motor development:
{"milestones": [{"id": "reaches_grasps_both_hands"}, {"id": "transfers_hand_to_hand"}, {"id": "bangs_objects_points"}, {"id": "pincer_grasp"}, {"id": "places_in_containers_self_feeds_fingers"}, {"id": "turns_pages_several"}, {"id": "stacks_blocks_uses_spoon"}, {"id": "scribbles_stacks_6_8_blocks"}, {"id": "copies_lines_circles_spoon_fork"}]}

Motorical impairments (lower):
{"details": {"spasticity": 2, "muscle_weakness": 1}}

Motorical impairments (upper):
{"details": {"coordination": 2}}

Story:
Dziecko chodzi i biega samodzielnie, choć widoczna jest lekka sztywność w lewej nodze podczas szybkiego chodu. Wchodzi po schodach trzymając się poręczy. Zabawa z rówieśnikami przebiega dobrze, dziecko uczestniczy w większości aktywności fizycznych. Terapia fizyczna raz w tygodniu.""",

        "output": """{
  "llm_milestone_score": 0.72,
  "llm_impairment_score": 0.78,
  "llm_combined_score": 0.75,
  "confidence": "high",
  "summary": "The child has good milestone attainment with only mild impairment burden.",
  "supporting_evidence": [
    "Many gross motor milestones are achieved, including independent walking, running, jumping, and stair-related skills",
    "Fine motor milestones are broadly age-appropriate",
    "Only mild lower-body spasticity and minimal weakness are reported",
    "Only mild upper-body coordination difficulty is reported",
    "The story confirms independent mobility with only mild limitations"
  ]
}""",
    },

    # Example 4 — Moderately limited function
    {
        "input": """Assess motor development for this record:

[Motorical Development Record]

Age: 3

Gross motor development:
{"milestones": [{"id": "rolls_both_directions"}, {"id": "sits_independently"}, {"id": "stands_with_support"}, {"id": "pulls_to_stand_cruises"}, {"id": "first_independent_steps"}]}

Fine motor development:
{"milestones": [{"id": "reaches_grasps_both_hands"}, {"id": "transfers_hand_to_hand"}, {"id": "pincer_grasp"}, {"id": "places_in_containers_self_feeds_fingers"}]}

Motorical impairments (lower):
{"details": {"spasticity": 4, "muscle_weakness": 3, "range_of_motion": 3}}

Motorical impairments (upper):
{"details": {"spasticity": 3, "muscle_weakness": 3, "coordination": 4}}

Story:
Barnet tar några steg inomhus men behöver rollator för längre sträckor och utomhus. Klarar inte att springa, hoppa eller gå i trappor utan hjälp. Finmotoriken är påtagligt begränsad – kan ta föremål men har svårt med precision och bilaterala uppgifter. Behöver assistans vid de flesta ADL-moment. Går på habilitering två gånger i veckan.""",

        "output": """{
  "llm_milestone_score": 0.34,
  "llm_impairment_score": 0.28,
  "llm_combined_score": 0.31,
  "confidence": "high",
  "summary": "The child shows limited milestone attainment and moderate-to-severe impairment burden.",
  "supporting_evidence": [
    "Only early gross motor milestones are reported",
    "Fine motor development is limited to early skills",
    "Several lower-body impairments are present at moderate to high severity",
    "Several upper-body impairments are present at moderate severity",
    "The story confirms dependence on a walker and assistance in most daily activities"
  ]
}""",
    },

    # Example 5 — Very limited but not absent function
    {
        "input": """Assess motor development for this record:

[Motorical Development Record]

Age: 2

Gross motor development:
{"milestones": [{"id": "rolls_both_directions"}]}

Fine motor development:
{"milestones": [{"id": "reaches_grasps_both_hands"}]}

Motorical impairments (lower):
{"details": {"spasticity": 5, "muscle_weakness": 4, "range_of_motion": 4}}

Motorical impairments (upper):
{"details": {"spasticity": 4, "muscle_weakness": 4, "coordination": 5}}

Story:
The child cannot sit without support and has not developed any form of independent mobility. All positioning and transfers require full caregiver assistance. Reaching for objects is inconsistent and poorly controlled. Intensive physiotherapy and occupational therapy are ongoing.""",

        "output": """{
  "llm_milestone_score": 0.10,
  "llm_impairment_score": 0.12,
  "llm_combined_score": 0.11,
  "confidence": "high",
  "summary": "The child has very low milestone attainment and high impairment burden.",
  "supporting_evidence": [
    "Only one gross motor milestone is reported",
    "Only one fine motor milestone is reported",
    "Lower-body impairments are severe",
    "Upper-body impairments are severe",
    "The story confirms full dependence and very limited functional movement"
  ]
}""",
    },
]


# ---------------------------
# CLEAN TEXT
# ---------------------------

def clean_text(value: object) -> str | None:
    """
    Clean a database value and return a usable string.

    Args:
        value (object): Raw database value.

    Returns:
        str | None: Cleaned text, or None if the value is empty.
    """
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    if text.lower() in {"none", "null", "nan"}:
        return None

    return text

def _extract_row_milestone_keys(row: dict) -> set[str]:
    """
    Extract milestone id/value/label strings from one motorical_development row.
 
    Reads both gross_motor_development and fine_motor_development, which are
    stored as JSONB dicts with a 'milestones' list.  Returns a flat set of
    stable string keys — the same logic used by the rule-based
    extract_milestone_keys() in motor_development.py.
 
    Args:
        row (dict): One row from the motorical_development table.
 
    Returns:
        set[str]: Milestone keys found in this row.
    """
    keys: set[str] = set()
 
    for field in ("gross_motor_development", "fine_motor_development"):
        raw = row.get(field)
        if raw is None:
            continue
 
        # The DB value arrives either as a dict (already parsed) or as a string
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                continue
 
        if not isinstance(raw, dict):
            continue
 
        for m in raw.get("milestones", []) or []:
            if m is None:
                continue
            if isinstance(m, dict):
                mid = m.get("id")
                val = m.get("value")
                lab = m.get("label")
                if mid is None and val is None and lab is None:
                    continue
                key = str(mid or val or lab).strip()
            else:
                key = str(m).strip()
 
            if key and key.lower() not in ("none", ""):
                keys.add(key)
 
    return keys

def extract_json_content(text: str) -> str:
    """
    Remove markdown code fences if the model wraps the JSON output
    in ```json ... ```.

    Args:
        text (str): Raw model output.

    Returns:
        str: Clean JSON string.
    """
    text = text.strip()

    if text.startswith("```json"):
        text = text.removeprefix("```json").strip()
    elif text.startswith("```"):
        text = text.removeprefix("```").strip()

    if text.endswith("```"):
        text = text.removesuffix("```").strip()

    return text


def repair_json_text(text: str) -> str:
    """
    Repair a few common JSON formatting mistakes from LLM output.

    Args:
        text (str): Raw or cleaned JSON-like string.

    Returns:
        str: Repaired JSON string.
    """
    text = text.strip()

    # Fix keys accidentally emitted without quote before colon
    text = re.sub(
        r'"([A-Za-z0-9_]+):\s*(\[|\{|"|-|\d)',
        r'"\1": \2',
        text,
    )

    return text


def normalize_motor_assessment_keys(text: str) -> str:
    """
    Normalize common alternative field names into the schema expected by
    MotorAssessment.
    """
    data = json.loads(text)

    key_map = {
        "milestone_score": "llm_milestone_score",
        "llm_milestones_score": "llm_milestone_score",
        "motor_milestone_score": "llm_milestone_score",

        "impairment_score": "llm_impairment_score",
        "llm_impairments_score": "llm_impairment_score",
        "motor_impairment_score": "llm_impairment_score",

        "combined_score": "llm_combined_score",
        "llm_motor_score": "llm_combined_score",
        "motor_score": "llm_combined_score",
        "overall_score": "llm_combined_score",

        "reasoning": "summary",
        "motor_summary": "summary",
        "evidence": "supporting_evidence",
    }

    normalized = {}
    for key, value in data.items():
        normalized[key_map.get(key, key)] = value

    return json.dumps(normalized, ensure_ascii=False)


# ---------------------------
# BUILD INPUT FOR ONE ROW
# ---------------------------

def build_md_input(row: dict, prior_milestones: set[str] | None = None) -> str:
    """
    Build structured input for the LLM from one motorical_development row.
 
    Args:
        row (dict): One row from the motorical_development table.
        prior_milestones (set[str] | None): Cumulative milestone keys already
            achieved in earlier age periods for this child.  When provided,
            they are included in the prompt so the LLM can score cumulatively,
            matching the rule-based motorscore_milestones_setvalue logic.
 
    Returns:
        str: Formatted input text for the LLM.
    """
    prior_section = ""
    if prior_milestones:
        sorted_keys = sorted(prior_milestones)
        prior_section = (
            "\nMilestones already achieved in earlier age periods "
            "(carry these forward — do NOT ignore them when scoring):\n"
            + ", ".join(sorted_keys)
            + "\n"
        )
 
    return f"""
[Motorical Development Record]
 
Age: {row.get("age")}
{prior_section}
Gross motor development (this age period):
{clean_text(row.get("gross_motor_development")) or "unknown"}
 
Fine motor development (this age period):
{clean_text(row.get("fine_motor_development")) or "unknown"}
 
Motorical impairments (lower):
{clean_text(row.get("motorical_impairments_lower")) or "unknown"}
 
Motorical impairments (upper):
{clean_text(row.get("motorical_impairments_upper")) or "unknown"}
 
Story:
{clean_text(row.get("story")) or "unknown"}
""".strip()

# ---------------------------
# LLM ANALYSIS
# ---------------------------

def analyze_md_row(text: str, model_name: str = MODEL_NAME) -> MotorAssessment:
    """
    Analyze one motorical development record with the LLM.
    Retries up to CHAT_MAX_RETRIES times on timeout.

    Args:
        text (str): Formatted input text for one record.
        model_name (str): Ollama model name.

    Returns:
        MotorAssessment: Structured LLM assessment.

    Raises:
        RuntimeError: If all retry attempts time out.
    """
    system_prompt = """
You are evaluating motor development in children with cerebral palsy based on structured survey data.

Your goal is NOT to make a free clinical judgment.
Your goal is to produce scores that are as comparable as possible to an existing rule-based motor scoring system.

The existing rule-based system mainly uses:
1. milestone attainment
2. impairment burden
3. a combined score with roughly equal weight for milestone attainment and impairments

TASK:
Return exactly these fields:
- llm_milestone_score
- llm_impairment_score
- llm_combined_score
- confidence
- summary
- supporting_evidence

SCORING RULES:

1. llm_milestone_score
- Must be a float between 0.0 and 1.0
- Higher = more achieved motor milestones relative to expected function for the child's age
- Base this primarily on the structured gross and fine motor milestone data
- The story may slightly clarify function, but should not override clear structured milestone data

2. llm_impairment_score
- Must be a float between 0.0 and 1.0
- Higher = less impairment / better function
- Lower = more impairments and/or more severe impairments
- Base this primarily on the structured upper/lower impairment fields
- The story may slightly clarify functional impact, but should not override clear impairment data

3. llm_combined_score
- Must be a float between 0.0 and 1.0
- Should be close to the average of llm_milestone_score and llm_impairment_score
- Use approximately:
  llm_combined_score = (llm_milestone_score + llm_impairment_score) / 2
- Only make very small adjustments based on the free-text story if absolutely necessary
- Do NOT use large subjective adjustments

INTERPRETATION GUIDE:
Milestone score:
- 0.0-0.2 = very few milestones achieved for age
- 0.2-0.4 = clearly limited milestone attainment
- 0.4-0.6 = moderate milestone attainment
- 0.6-0.8 = good milestone attainment
- 0.8-1.0 = very high milestone attainment

Impairment score:
- 0.0-0.2 = very severe and/or numerous impairments
- 0.2-0.4 = marked impairment burden
- 0.4-0.6 = moderate impairment burden
- 0.6-0.8 = mild impairment burden
- 0.8-1.0 = minimal or no reported impairments

IMPORTANT:
- Base your assessment ONLY on the provided information
- Be conservative
- Prioritize structured data over narrative text
- Keep the scores numerically aligned with a rule-based scoring system, not with a clinical impression scale
- Return raw JSON only
- Do NOT wrap the JSON in markdown code fences
- Output MUST be in English
- Use EXACTLY the required field names and no others
"""

    user_prompt = f"""Assess motor development for this record:

{text}"""

    messages = [{"role": "system", "content": system_prompt}]

    for example in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": example["input"]})
        messages.append({"role": "assistant", "content": example["output"]})

    messages.append({"role": "user", "content": user_prompt})

    def _call_llm():
        return chat(
            model=model_name,
            messages=messages,
            format=MotorAssessment.model_json_schema(),
            options={"num_predict": 512, "temperature": 0.1},
        )

    for attempt in range(CHAT_MAX_RETRIES + 1):
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_call_llm)
                response = future.result(timeout=CHAT_TIMEOUT_SECONDS)
            break
        except FuturesTimeoutError:
            print(
                f"    Timeout on attempt {attempt + 1}/{CHAT_MAX_RETRIES + 1} "
                f"({CHAT_TIMEOUT_SECONDS}s), retrying..."
            )
            if attempt == CHAT_MAX_RETRIES:
                raise RuntimeError(
                    f"LLM timeout after {CHAT_MAX_RETRIES + 1} attempts "
                    f"({CHAT_TIMEOUT_SECONDS}s per attempt)"
                )

    cleaned_response = extract_json_content(response.message.content)
    repaired_response = repair_json_text(cleaned_response)
    normalized_response = normalize_motor_assessment_keys(repaired_response)

    try:
        return MotorAssessment.model_validate_json(normalized_response)
    except Exception:
        print("\nRaw model output:")
        print(response.message.content)
        print("\nCleaned / repaired output:")
        print(repaired_response)
        print("\nNormalized output:")
        print(normalized_response)
        raise


# ---------------------------
# ANALYZE ONE CHILD
# ---------------------------

def analyze_child(df, introductory_id: str) -> list[dict]:
    """
    Analyze all motorical development rows for one child, scoring cumulatively.
 
    Milestones observed in earlier age periods are extracted from the structured
    data and forwarded to each subsequent LLM call so that the milestone score
    is cumulative — matching the rule-based approach in
    motorscore_milestones_setvalue().
 
    Impairment scoring is intentionally NOT cumulative: impairments are assessed
    per age period, reflecting the child's current burden at that point in time,
    which is also how the rule-based system works.
 
    Args:
        df (pl.DataFrame): Motorical development table.
        introductory_id (str): Child/survey id.
 
    Returns:
        list[dict]: One result dict per age row.
    """
    import polars as pl
 
    df_child = (
        df.filter(pl.col("introductory_id") == introductory_id)
        .sort("age")
    )
 
    if df_child.height == 0:
        return []
 
    child_results: list[dict] = []
    seen_milestones: set[str] = set()   # cumulative across age periods
 
    for row in df_child.iter_rows(named=True):
        age = row.get("age")
 
        import time
        row_start = time.perf_counter()
 
        # Pass whatever has been seen so far (empty set on first iteration)
        text_input = build_md_input(row, prior_milestones=seen_milestones or None)
        print(f"    Sending input for age {age} "
              f"({len(text_input)} chars, "
              f"{len(seen_milestones)} prior milestones)...")
 
        try:
            result = analyze_md_row(text_input)
        except RuntimeError as e:
            print(f"    Age {age} SKIPPED: {e}")
            # Still update seen_milestones so subsequent ages are not affected
            seen_milestones |= _extract_row_milestone_keys(row)
            continue
        except Exception as e:
            print(f"    Age {age} ERROR (unexpected): {e}")
            seen_milestones |= _extract_row_milestone_keys(row)
            continue
 
        elapsed = time.perf_counter() - row_start
        print(f"    Age {age} done in {elapsed:.2f}s.")
 
        child_results.append({
            "introductory_id": row.get("introductory_id"),
            "age": age,
            "llm_milestone_score":  result.llm_milestone_score,
            "llm_impairment_score": result.llm_impairment_score,
            "llm_combined_score":   result.llm_combined_score,
            "confidence":           result.confidence,
            "summary":              result.summary,
            "supporting_evidence":  " | ".join(result.supporting_evidence),
        })
 
        # Accumulate milestones AFTER scoring this age period
        seen_milestones |= _extract_row_milestone_keys(row)
 
    return child_results
# ---------------------------
# WRITE TEXT REPORT
# ---------------------------

def write_text_report(
    results_df: pl.DataFrame,
    output_path: Path,
    total_elapsed_time: float,
) -> None:
    """
    Write a human-readable text report.

    Args:
        results_df (pl.DataFrame): Final results table.
        output_path (Path): Path to output text file.
        total_elapsed_time (float): Total runtime in seconds.

    Returns:
        None
    """
    lines: list[str] = []
    lines.append("LLM MOTOR SCORE ANALYSIS")
    lines.append("=" * 80)
    lines.append(f"Total runtime: {total_elapsed_time:.2f} seconds")
    lines.append("")

    if results_df.height == 0:
        lines.append("No results were generated.")
    else:
        for row in results_df.sort(["introductory_id", "age"]).iter_rows(named=True):
            lines.append(f"Introductory ID: {row['introductory_id']}")
            lines.append(f"Age: {row['age']}")
            lines.append(f"LLM milestone score: {row['llm_milestone_score']:.3f}")
            lines.append(f"LLM impairment score: {row['llm_impairment_score']:.3f}")
            lines.append(f"LLM combined score: {row['llm_combined_score']:.3f}")
            lines.append(f"Confidence: {row['confidence']}")
            lines.append(f"Summary: {row['summary']}")
            lines.append(f"Supporting evidence: {row['supporting_evidence']}")
            lines.append("-" * 80)

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------
# MAIN FUNCTION
# ---------------------------

def main() -> None:
    """
    Run LLM motor score analysis for all ids in INTRODUCTORY_IDS
    and save results to output files.

    Returns:
        None
    """
    total_start_time = time.perf_counter()

    conn = get_connection()
    data = load_data(conn)
    df = data["motorical_development"]

    all_results: list[dict] = []

    for introductory_id in INTRODUCTORY_IDS:
        child_start_time = time.perf_counter()
        print(f"Processing introductory_id: {introductory_id}")

        try:
            child_results = analyze_child(df, introductory_id)

            if not child_results:
                print(f"  No motorical development rows found for {introductory_id}")
                continue

            all_results.extend(child_results)

            child_elapsed_time = time.perf_counter() - child_start_time
            print(
                f"  Done. Processed {len(child_results)} row(s) "
                f"in {child_elapsed_time:.2f} seconds."
            )

        except Exception as e:
            print(f"  Error for {introductory_id}: {e}")

    results_df = pl.DataFrame(all_results) if all_results else pl.DataFrame()

    if results_df.height > 0:
        results_df.write_csv(OUTPUT_CSV_PATH)

    total_elapsed_time = time.perf_counter() - total_start_time

    write_text_report(results_df, OUTPUT_TXT_PATH, total_elapsed_time)

    print("\nFinished.")
    print(f"Total runtime: {total_elapsed_time:.2f} seconds.")
    print(f"Text report saved to: {OUTPUT_TXT_PATH}")
    if results_df.height > 0:
        print(f"CSV file saved to: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()