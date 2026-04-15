# Important note:
# This is not an objective clinical measure, but an LLM-derived summary score.

from typing import List, Literal
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import time

import polars as pl
from ollama import chat
from pydantic import BaseModel

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
    motor_score_1_to_10: int
    confidence: Literal["low", "medium", "high"]
    summary: str
    supporting_evidence: List[str]


# ---------------------------
# FEW-SHOT EXAMPLES
# ---------------------------

FEW_SHOT_EXAMPLES = [
    # Example 1 — Score 10: all milestones, no impairments
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

        "output": '{"motor_score_1_to_10": 10, "confidence": "high", "summary": "The child has achieved all 21 gross motor and all 14 age-appropriate fine motor milestones, with no lower or upper body impairments reported. Motor function is fully intact and consistent with the highest functional level.", "supporting_evidence": ["All 21 gross motor milestones achieved, including advanced skills: jumping rope, riding a two-wheeled bike, and accurate throwing and catching", "All 14 age-appropriate fine motor milestones achieved, including correct pencil grasp, scissor use, and copying letters and squares", "No lower-body impairments reported", "No upper-body impairments reported", "Story confirms typical or above-average motor development with no professional concerns"]}',
    },

    # Example 2 — Score 1: no milestones, maximum impairments
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

        "output": '{"motor_score_1_to_10": 1, "confidence": "high", "summary": "The child has achieved no gross or fine motor milestones at age 4, and presents with all lower and upper body impairment dimensions at maximum severity. Motor function is critically impaired with complete dependence on caregivers for all activities.", "supporting_evidence": ["No gross motor milestones achieved — child cannot roll, sit, stand, or walk", "No fine motor milestones achieved — no functional grasping, pointing, or object handling", "Lower-body impairments at maximum severity: spasticity 5/5, muscle weakness 5/5, range of motion 5/5", "Upper-body impairments at maximum severity: spasticity 5/5, muscle weakness 5/5, coordination 5/5", "Story confirms spastic tetraparesis with complete care dependence and no functional motor progress"]}',
    },

    # Example 3 — Score 7: good milestones, mild impairments
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

        "output": '{"motor_score_1_to_10": 7, "confidence": "high", "summary": "The child walks and runs independently and has achieved most gross and fine motor milestones expected at age 3. Mild lower-limb spasticity and mild coordination difficulties are present but do not substantially limit daily function.", "supporting_evidence": ["12 out of the most advanced age-3 gross motor milestones achieved, including running, jumping with both feet, and stair walking", "9 fine motor milestones achieved through copying lines and circles — developmentally appropriate for age 3", "Mild lower-body spasticity (2/5) and minimal muscle weakness (1/5) — functional ambulation maintained", "Mild upper-body coordination difficulty (2/5) — fine motor tasks largely intact", "Story confirms independent walking and running with mild stiffness in one leg; attends physical therapy once weekly"]}',
    },

    # Example 4 — Score 4: limited milestones, moderate impairments
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

        "output": '{"motor_score_1_to_10": 4, "confidence": "high", "summary": "The child has achieved only basic early gross motor milestones and a limited set of fine motor skills. Moderate-to-high lower-limb spasticity and moderate upper-body impairments significantly restrict independence, with reliance on a walker for longer distances.", "supporting_evidence": ["Only 5 gross motor milestones achieved — walking indoors without device possible but limited; running, jumping, and stair use not achieved", "Only 4 early fine motor milestones achieved — grasping and pincer present but precision and bilateral tasks impaired", "Lower-body impairments at moderate-to-high severity: spasticity 4/5, muscle weakness 3/5, range of motion 3/5", "Upper-body impairments at moderate severity: spasticity 3/5, muscle weakness 3/5, coordination 4/5", "Story confirms walker-dependent mobility outdoors and significant assistance needed for most daily activities"]}',
    },

    # Example 5 — Score 2: almost no milestones, severe impairments
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

        "output": '{"motor_score_1_to_10": 2, "confidence": "high", "summary": "The child has achieved only the most rudimentary gross and fine motor milestones at age 2, with severe lower and upper body impairments across all measured dimensions. Almost entirely dependent on caregivers for all motor activities.", "supporting_evidence": ["Only 1 gross motor milestone achieved (rolling) — no sitting, standing, or walking", "Only 1 fine motor milestone achieved (inconsistent reaching) — no transfer, pincer grasp, or self-feeding", "Lower-body impairments at high severity: spasticity 5/5, muscle weakness 4/5, range of motion 4/5", "Upper-body impairments at high severity: spasticity 4/5, muscle weakness 4/5, coordination 5/5", "Story confirms complete positional dependence and poorly controlled voluntary movement"]}',
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

    # Fix common mistake:
    # "supporting_evidence: [
    # -> "supporting_evidence": [
    text = re.sub(
        r'"([A-Za-z0-9_]+):\s*(\[|\{|"|-|\d)',
        r'"\1": \2',
        text,
    )

    return text


def normalize_motor_assessment_keys(text: str) -> str:
    data = json.loads(text)

    key_map = {
        "motor_function_level": "motor_score_1_to_10",
        "motor_function_score": "motor_score_1_to_10",
        "motor_score_1_and_10": "motor_score_1_to_10",
        "score": "motor_score_1_to_10",
        "reasoning": "summary",
        "motor_summary": "summary",
        "gross_motor": "gross_motor_level",
        "gross_motor_leg_level": "gross_motor_level",
        "fine_motor": "fine_motor_level",
        "impairment_level": "impairment_severity",
        "evidence": "supporting_evidence",
    }

    normalized = {}
    for key, value in data.items():
        normalized[key_map.get(key, key)] = value

    return json.dumps(normalized, ensure_ascii=False)


# ---------------------------
# BUILD INPUT FOR ONE ROW
# ---------------------------

def build_md_input(row: dict) -> str:
    """
    Build structured input for the LLM from one motorical_development row.

    Args:
        row (dict): One row from the motorical_development table.

    Returns:
        str: Formatted input text for the LLM.
    """
    return f"""
[Motorical Development Record]

Age: {row.get("age")}

Gross motor development:
{clean_text(row.get("gross_motor_development")) or "unknown"}

Fine motor development:
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

TASK:
Estimate the child's motor function level on a scale from 1 to 10.

IMPORTANT:
- Base your assessment ONLY on the provided information.
- Use both structured responses and free-text descriptions.
- Do NOT guess beyond what is stated.
- Be conservative.
- Return raw JSON only.
- Do NOT wrap the JSON in markdown code fences.
- Use EXACTLY these field names and no others:
  - motor_score_1_to_10
  - confidence
  - summary
  - supporting_evidence

SCALE GUIDELINES:
1 = very limited motor function, severe impairments
3 = major limitations, very few abilities
5 = moderate motor function with clear limitations
7 = relatively good motor abilities with some impairments
10 = very strong motor function, minimal limitations

FIELD REQUIREMENTS:
- motor_score_1_to_10: integer from 1 to 10
- confidence: one of "low", "medium", "high"
- summary: short English summary
- gross_motor_level: short English description
- fine_motor_level: short English description
- impairment_severity: short English description
- supporting_evidence: list of short English strings

LANGUAGE:
- Input may be multilingual (Swedish, English, Polish).
- Output MUST be in English.

Return ONLY valid JSON matching the required field names exactly.
"""

    user_prompt = f"""Assess motor development for this record:

{text}"""

    # Build messages list: system prompt → few-shot pairs → real user prompt
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
            options={"num_predict": 512},
        )

    for attempt in range(CHAT_MAX_RETRIES + 1):
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_call_llm)
                response = future.result(timeout=CHAT_TIMEOUT_SECONDS)
            break  # succeeded
        except FuturesTimeoutError:
            print(
                f"    Timeout på försök {attempt + 1}/{CHAT_MAX_RETRIES + 1} "
                f"({CHAT_TIMEOUT_SECONDS}s), försöker igen..."
            )
            if attempt == CHAT_MAX_RETRIES:
                raise RuntimeError(
                    f"LLM timeout efter {CHAT_MAX_RETRIES + 1} försök "
                    f"({CHAT_TIMEOUT_SECONDS}s per försök)"
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

def analyze_child(df: pl.DataFrame, introductory_id: str) -> list[dict]:
    """
    Analyze all motorical development rows for one child.

    Args:
        df (pl.DataFrame): Motorical development table.
        introductory_id (str): Child/survey id.

    Returns:
        list[dict]: One result dict per age row.
    """
    df_child = (
        df.filter(pl.col("introductory_id") == introductory_id)
        .sort("age")
    )

    if df_child.height == 0:
        return []

    child_results = []

    for row in df_child.iter_rows(named=True):
        age = row.get("age")
        row_start_time = time.perf_counter()

        text_input = build_md_input(row)
        print(f"    Skickar input för age {age} ({len(text_input)} tecken)...")

        try:
            result = analyze_md_row(text_input)
        except RuntimeError as e:
            print(f"    Age {age} HOPPADES ÖVER: {e}")
            continue
        except Exception as e:
            print(f"    Age {age} FEL (oväntat): {e}")
            continue

        row_elapsed_time = time.perf_counter() - row_start_time
        print(f"    Age {age} done in {row_elapsed_time:.2f} seconds.")

        child_results.append({
            "introductory_id": row.get("introductory_id"),
            "age": age,
            "llm_motor_score": result.motor_score_1_to_10,
            "confidence": result.confidence,
            "summary": result.summary,
            "supporting_evidence": " | ".join(result.supporting_evidence),
        })

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
            lines.append(f"LLM motor score: {row['llm_motor_score']}")
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