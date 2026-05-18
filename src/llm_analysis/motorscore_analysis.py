# Important note:
# This is not an objective clinical measure, but an LLM-derived summary score
# intentionally designed to be comparable to the project's rule-based motor scores,
# while also incorporating information from free-text narratives.
#
# Architecture (Roll 2):
#   1. LLM reads structured milestone/impairment data AND the free-text story.
#      It estimates three input values:
#        - estimated_cum_milestones  (may exceed what the checkboxes show)
#        - estimated_n_selected      (impairments with non-zero severity)
#        - estimated_sum_ratings     (sum of severity ratings)
#   2. Python applies the EXACT same formula as motorscore_milestones_setvalue /
#      motorscore_impairments_setvalue to those estimated inputs.
#   This means scores are comparable to the rule-based system but deviate
#   meaningfully when the narrative reveals information the checkboxes missed.

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from connect_db import get_connection
from dataloader import load_data


# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "gemma4:26b"
CHAT_TIMEOUT_SECONDS = 60
CHAT_MAX_RETRIES = 2

INTRODUCTORY_IDS = [
    "498a4d90-77c6-41b2-ad39-517a2c2a9702",
    "e0129e7b-c90f-4f49-87d3-6987eb577cdb",
    "d87153c9-75b3-4305-99a4-42abc0366651",
    "47a9e4ae-8d91-4070-ae18-f2d9af891299",
    "fcdc7e60-1c2f-4178-bafd-db1d42e869ee",
    "ee9e6fc1-a9d3-4f45-aa0e-01723ebb2930",
    "52dac13b-a335-449d-a7db-a58e40b5e213",
    "42475b28-2dfd-4114-ac53-d8619881dd2f",
    "7e68f3b3-509b-4352-8eb1-400c9407ac9b",
    "4be3b41c-a0b4-4e7b-ae49-896b37ea2052",
    "1950325f-99da-47b4-b49d-735253ba0aaa",
    "30302f7a-c470-47bf-8f0e-d104b3065d99",
    "c8f4ec50-18b6-47ed-92a3-919da180a10d",
    "8dba1f55-9e79-4e62-90c3-02e9609d3feb",
    "d2703a20-7b4a-4624-b31a-306eebe4caa0",
    "f1856ef8-2fe0-480d-9635-cfc0be308458",
    "771d12c3-bc1a-4a97-ad27-00d35b24f87e",
    "1d0afd8d-6945-488a-964c-724e95db6696",
    "1019fb0a-480d-4bef-b8f9-493b9dfe253b",
    "6e7aeec2-2846-433d-a4ac-0e753da08530",
    "e30d335e-3a7a-484d-951d-f8e3f17ccfb3",
    "4c89ca0a-f5c3-4b7b-96be-a7919c679303",
    "578adb11-a12f-4121-a567-afe67c25640b",
    "0a584ba1-cdf4-4251-9168-5f8ccc0240e3",
    "7e42b31a-c597-4418-9bf6-a8c3286d049f",
    "3c1f5e61-56fd-4ac3-af9e-0d6fe054ddb7",
    "f9231c8d-2ade-4c0e-a878-a9524ccc3d65",
    "df67e7ea-0b50-408b-9342-4c29d0efa839",
    "16f3f961-07a2-4099-8498-1bad9c2faa19",
    "44cd783c-b33d-4553-89cd-2a73b59e1982",
    "cd26a009-6e51-4372-b151-b7d2bb8b7183",
    "c0990a55-916e-47ba-b29a-aee83d9f33c9",
    "89e4bf27-9a6f-45e8-a415-ef53f23f7931",
    "65ab3206-7371-4471-845c-6d238050494f",
]

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "motorscore_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_TXT_PATH = OUTPUT_DIR / "llm_motorscore_results.txt"
OUTPUT_CSV_PATH = OUTPUT_DIR / "llm_motorscore_results.csv"


# ─────────────────────────────────────────────────────────────────────────────
# LOOKUP TABLES  (mirrors motor_scores.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

POSSIBLE_MILESTONES_BY_AGE_GMFCS: dict[int, dict[int, int]] = {
    1: {1: 12, 2: 11, 3:  9, 4:  7, 5:  4},
    2: {1: 19, 2: 17, 3: 13, 4:  9, 5:  5},
    3: {1: 27, 2: 23, 3: 16, 4: 11, 5:  6},
    4: {1: 35, 2: 29, 3: 18, 4: 12, 5:  7},
}

N_NAMED_BY_AGE_GMFCS: dict[int, dict[int, int]] = {
    1: {1:  9, 2:  9, 3:  8, 4:  7, 5:  5},
    2: {1: 16, 2: 16, 3: 14, 4: 11, 5:  7},
    3: {1: 17, 2: 17, 3: 15, 4: 11, 5:  8},
    4: {1: 18, 2: 18, 3: 16, 4: 11, 5:  8},
}

_GMFCS_STR_TO_INT: dict[str, int] = {
    "Level I – Walks without limitations": 1,
    "Level II – Walks with some limitations": 2,
    "Level III – Walks with assistive devices": 3,
    "Level IV – Limited mobility, primarily uses a wheelchair": 4,
    "Level V – Severe limitations, needs full assistance for mobility": 5,
    "Not sure / Don't know": 3,
}


def _gmfcs_int_from_str(gmfcs_str: str | None) -> int:
    return _GMFCS_STR_TO_INT.get(gmfcs_str or "", 3)

def _possible_milestones(age: int, gmfcs_int: int) -> int:
    return POSSIBLE_MILESTONES_BY_AGE_GMFCS[max(1, min(4, age))].get(gmfcs_int, 16)

def _n_named(age: int, gmfcs_int: int) -> int:
    return N_NAMED_BY_AGE_GMFCS[max(1, min(4, age))].get(gmfcs_int, 14)


# ─────────────────────────────────────────────────────────────────────────────
# EXACT FORMULA  (same math as motor_scores.py — applied in Python, not LLM)
# ─────────────────────────────────────────────────────────────────────────────

def apply_formula(
    cum_milestones: int,
    n_selected: int,
    sum_ratings: float,
    possible: int,
    n_named: int,
) -> dict[str, float]:
    """
    Apply the exact rule-based formula to estimated input values.

    Returns milestone_score, impairment_score (mms_normalized), and combined_score.
    """
    # Milestone score
    milestone_score = min(cum_milestones / possible, 1.0) if possible > 0 else 0.0

    # Impairment score
    presence_ratio = min(n_selected / n_named, 1.0) if n_named > 0 else 0.0

    if n_selected > 0:
        mean_severity  = sum_ratings / n_selected
        severity_ratio = max(0.0, min(1.0, (mean_severity - 1) / 4))
    else:
        severity_ratio = 0.0

    impairment_burden = (presence_ratio + severity_ratio) / 2
    mms_normalized    = max(0.0, min(1.0, 1.0 - impairment_burden))

    combined_score = (milestone_score + mms_normalized) / 2

    return {
        "milestone_score":    milestone_score,
        "impairment_score":   mms_normalized,
        "combined_score":     combined_score,
        # intermediates — useful for debugging / audit
        "presence_ratio":     presence_ratio,
        "severity_ratio":     severity_ratio,
        "impairment_burden":  impairment_burden,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURED OUTPUT  — LLM returns estimated INPUT VALUES, not scores
# ─────────────────────────────────────────────────────────────────────────────

class MotorInputEstimate(BaseModel):
    estimated_cum_milestones: int = Field(
        ge=0,
        description=(
            "Best estimate of cumulative unique milestones achieved up to and "
            "including this age period, combining checkbox data with any "
            "additional milestones inferred from the free-text story."
        ),
    )
    estimated_n_selected: int = Field(
        ge=0,
        description=(
            "Best estimate of the number of distinct impairment domains with a "
            "non-zero severity rating, combining structured fields with story."
        ),
    )
    estimated_sum_ratings: float = Field(
        ge=0.0,
        description=(
            "Best estimate of the total sum of all non-zero impairment severity "
            "ratings (each on a 1–5 scale), combining structured fields with story."
        ),
    )
    confidence: Literal["low", "medium", "high"]
    story_adjustments: List[str] = Field(
        description=(
            "List of specific adjustments made because of the free-text story "
            "(e.g. 'Story mentions child walks without walker — added 2 gross "
            "motor milestones not in checkboxes'). Empty list if story added "
            "nothing beyond what the structured data already captured."
        ),
    )
    summary: str = Field(
        description=(
            "One or two sentences explaining the final estimates and the key "
            "reasoning behind any story-driven adjustments."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# FEW-SHOT EXAMPLES
# Each example shows: structured baseline → story adjustment → final estimates
# ─────────────────────────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = [
    # Example 1 — Story confirms structured data, no adjustment needed
    {
        "input": """Estimate motor input values for this record:

[Motorical Development Record]

Age: 4
GMFCS level: 1 (Walks without limitations)
Milestone ceiling (possible_milestones): 35
Impairment ceiling (n_named): 18
Prior cumulative milestones (from earlier age periods): 0

Gross motor development (this period):
{"milestones": [{"id": "rolls_both_directions"}, {"id": "sits_independently"}, {"id": "stands_with_support"}, {"id": "first_independent_steps"}, {"id": "runs_and_climbs_furniture"}, {"id": "jumps_both_feet"}, {"id": "balances_one_foot_briefly"}, {"id": "pedals_tricycle_hops"}, {"id": "rides_bike_training_wheels"}, {"id": "jumps_rope_sports"}, {"id": "runs_smoothly_throws_catches"}]}

Fine motor development (this period):
{"milestones": [{"id": "reaches_grasps_both_hands"}, {"id": "pincer_grasp"}, {"id": "turns_pages_several"}, {"id": "stacks_blocks_uses_spoon"}, {"id": "copies_lines_circles_spoon_fork"}, {"id": "scissors_large_buttons"}, {"id": "draws_simple_person"}, {"id": "copies_squares_attempts_letters"}]}

Motorical impairments (lower): {"details": {}}
Motorical impairments (upper): {"details": {}}

Story:
The child participates fully in all physical activities. No motor concerns raised by anyone. Runs, jumps and uses scissors well for their age.""",

        "output": """{
  "estimated_cum_milestones": 19,
  "estimated_n_selected": 0,
  "estimated_sum_ratings": 0.0,
  "confidence": "high",
  "story_adjustments": [],
  "summary": "Checkboxes show 11 gross + 8 fine = 19 milestones; story confirms full participation but adds no specific new milestones beyond what is already captured. No impairments in structured data and story is consistent."
}""",
    },

    # Example 2 — Story reveals additional milestones not in checkboxes
    {
        "input": """Estimate motor input values for this record:

[Motorical Development Record]

Age: 3
GMFCS level: 2 (Walks with some limitations)
Milestone ceiling (possible_milestones): 23
Impairment ceiling (n_named): 17
Prior cumulative milestones (from earlier age periods): 8

Gross motor development (this period):
{"milestones": [{"id": "first_independent_steps"}, {"id": "squats_and_stands"}]}

Fine motor development (this period):
{"milestones": [{"id": "stacks_blocks_uses_spoon"}, {"id": "scribbles_stacks_6_8_blocks"}]}

Motorical impairments (lower): {"details": {"spasticity": 2}}
Motorical impairments (upper): {"details": {}}

Story:
Barnet går nu utan hjälpmedel och kan springa kortare sträckor, även om det haltar lite. Klättrar uppför trappor med stöd av räcke. Kan klippa med sax och rita enkla figurer. Leker med jämnåriga utan problem.""",

        "output": """{
  "estimated_cum_milestones": 17,
  "estimated_n_selected": 1,
  "estimated_sum_ratings": 2.0,
  "confidence": "high",
  "story_adjustments": [
    "Story mentions child runs short distances — added 'runs_and_climbs_furniture' milestone not checked in structured data (+1)",
    "Story mentions climbing stairs with railing — added 'climbs_stairs_hands_knees' milestone not checked (+1)",
    "Story confirms scissor use and drawing — consistent with fine motor checkboxes, no extra added"
  ],
  "summary": "Structured data gives 8 prior + 4 new = 12 cumulative milestones, but the story clearly describes running and stair-climbing that are not checked — raising the estimate to 14 new milestones and 17 cumulative. One impairment (spasticity rating 2) from lower body."
}""",
    },

    # Example 3 — Story reveals impairments are less severe than rated
    {
        "input": """Estimate motor input values for this record:

[Motorical Development Record]

Age: 4
GMFCS level: 3 (Walks with assistive devices)
Milestone ceiling (possible_milestones): 18
Impairment ceiling (n_named): 16
Prior cumulative milestones (from earlier age periods): 7

Gross motor development (this period):
{"milestones": [{"id": "first_independent_steps"}, {"id": "climbs_stairs_hands_knees"}]}

Fine motor development (this period):
{"milestones": [{"id": "copies_lines_circles_spoon_fork"}, {"id": "scissors_large_buttons"}]}

Motorical impairments (lower): {"details": {"spasticity": 5, "muscle_weakness": 5, "toe_walking": 4}}
Motorical impairments (upper): {"details": {"coordination": 3}}

Story:
Barnet gör stora framsteg. Fysioterapeuten noterade förra månaden att spasticiteten minskat påtagligt efter botoxbehandling och att barnet nu tar fler steg utan rollator än förut. Styrkan har också förbättrats märkbart. Koordinationen i händerna är fortfarande lite svag.""",

        "output": """{
  "estimated_cum_milestones": 11,
  "estimated_n_selected": 4,
  "estimated_sum_ratings": 11.0,
  "confidence": "medium",
  "story_adjustments": [
    "Story reports spasticity has decreased markedly after botox — adjusted spasticity rating from 5 to 3",
    "Story reports strength has improved noticeably — adjusted muscle_weakness rating from 5 to 3",
    "Toe-walking and upper coordination ratings unchanged as story does not specifically address them",
    "Story mentions more independent steps than before — no new milestone added since 'first_independent_steps' already captured"
  ],
  "summary": "7 prior + 4 new checkboxes = 11 cumulative milestones. The story credibly describes recent clinical improvement that makes the checkbox ratings for spasticity and weakness too high — adjusted both down from 5 to 3, changing sum_ratings from 17 to 13."
}""",
    },

    # Example 4 — Sparse checkboxes, story fills major gap
    {
        "input": """Estimate motor input values for this record:

[Motorical Development Record]

Age: 2
GMFCS level: 1 (Walks without limitations)
Milestone ceiling (possible_milestones): 19
Impairment ceiling (n_named): 16
Prior cumulative milestones (from earlier age periods): 0

Gross motor development (this period):
{"milestones": [{"id": "sits_independently"}]}

Fine motor development (this period):
{"milestones": []}

Motorical impairments (lower): {"details": {}}
Motorical impairments (upper): {"details": {}}

Story:
Föräldern glömde fylla i checkboxarna för den här perioden. Men i textrutan berättar de att barnet kröp, reste sig, tog sina första steg och nu springer runt hemma. Plockar upp föremål med pincettgrepp och matar sig själv med sked. Inga kliniska bekymmer.""",

        "output": """{
  "estimated_cum_milestones": 9,
  "estimated_n_selected": 0,
  "estimated_sum_ratings": 0.0,
  "confidence": "medium",
  "story_adjustments": [
    "Story mentions crawling — added 'crawls_or_scoots' (+1)",
    "Story mentions pulling to stand — added 'pulls_to_stand_cruises' (+1)",
    "Story mentions first independent steps — added 'first_independent_steps' (+1)",
    "Story mentions running — added 'runs_and_climbs_furniture' (+1)",
    "Story mentions pincer grip — added 'pincer_grasp' (+1)",
    "Story mentions self-feeding with spoon — added 'places_in_containers_self_feeds_fingers' and 'stacks_blocks_uses_spoon' (+2)",
    "Checkboxes only had 'sits_independently' — story makes it clear this was under-reported"
  ],
  "summary": "Only 1 checkbox ticked, but the story clearly describes 8 distinct achieved milestones. Confidence is medium rather than high because the story is retrospective parental recall, not direct observation."
}""",
    },

    # Example 5 — Story indicates worse function than checkboxes suggest
    {
        "input": """Estimate motor input values for this record:

[Motorical Development Record]

Age: 3
GMFCS level: 4 (Limited mobility, primarily uses a wheelchair)
Milestone ceiling (possible_milestones): 11
Impairment ceiling (n_named): 11
Prior cumulative milestones (from earlier age periods): 3

Gross motor development (this period):
{"milestones": [{"id": "sits_independently"}, {"id": "stands_with_support"}, {"id": "first_independent_steps"}]}

Fine motor development (this period):
{"milestones": [{"id": "pincer_grasp"}, {"id": "stacks_blocks_uses_spoon"}]}

Motorical impairments (lower): {"details": {"spasticity": 3}}
Motorical impairments (upper): {"details": {"coordination": 2}}

Story:
Vi är lite osäkra på hur vi skulle svara på frågorna. Barnet kan sitta med stöd men inte självständigt, och de "första stegen" är egentligen bara några steg med full handstöd från terapeuten på kliniken — ingenting hemma. Vi satte i kryss för att det ändå hänt. Spasticiteten är ett stort problem och vi hade svårt att ge ett rättvist betyg.""",

        "output": """{
  "estimated_cum_milestones": 5,
  "estimated_n_selected": 2,
  "estimated_sum_ratings": 11.0,
  "confidence": "medium",
  "story_adjustments": [
    "Story clarifies 'sits_independently' was actually with support — removed this milestone (-1)",
    "Story clarifies 'first_independent_steps' were only therapist-assisted clinic steps — removed this milestone (-1)",
    "Story says spasticity is a major problem and parents found it hard to rate fairly — adjusted spasticity from 3 to 5",
    "Upper coordination rating left unchanged as story does not specifically address it"
  ],
  "summary": "Checkboxes gave 3 prior + 5 new = 8 cumulative milestones, but the story reveals that 2 of the new ones were over-reported. Adjusted to 5 cumulative. Spasticity severity raised from 3 to 5 based on parent narrative, changing sum_ratings from 5 to 7."
}""",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    return text


def _extract_row_milestone_keys(row: dict) -> set[str]:
    keys: set[str] = set()
    for field in ("gross_motor_development", "fine_motor_development"):
        raw = row.get(field)
        if raw is None:
            continue
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


def _structured_impairment_values(row: dict) -> tuple[int, float]:
    """Return (n_selected, sum_ratings) from structured impairment fields."""
    n_selected = 0
    sum_ratings = 0.0
    for field in ("motorical_impairments_lower", "motorical_impairments_upper"):
        raw = row.get(field)
        if not raw or not isinstance(raw, dict):
            continue
        for v in raw.get("details", {}).values():
            try:
                fv = float(v)
                if fv > 0:
                    n_selected += 1
                    sum_ratings += fv
            except (TypeError, ValueError):
                pass
    return n_selected, sum_ratings


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
    return re.sub(r'"([A-Za-z0-9_]+):\s*(\[|\{|"|-|\d)', r'"\1": \2', text.strip())


# ─────────────────────────────────────────────────────────────────────────────
# BUILD PROMPT
# ─────────────────────────────────────────────────────────────────────────────

def build_md_input(
    row: dict,
    gmfcs_int: int,
    prior_milestones: set[str] | None = None,
) -> str:
    """
    Build the LLM prompt for one motorical_development row.

    Shows the structured baseline values so the LLM knows exactly what the
    checkboxes captured, and can focus its effort on what the story adds.
    """
    age = int(row.get("age") or 1)
    possible = _possible_milestones(age, gmfcs_int)
    n_named  = _n_named(age, gmfcs_int)

    gmfcs_labels = {
        1: "1 (Walks without limitations)",
        2: "2 (Walks with some limitations)",
        3: "3 (Walks with assistive devices)",
        4: "4 (Limited mobility, primarily uses a wheelchair)",
        5: "5 (Severe limitations, needs full assistance for mobility)",
    }

    n_prior = len(prior_milestones) if prior_milestones else 0
    n_struct_milestones = len(_extract_row_milestone_keys(row))
    struct_n_selected, struct_sum_ratings = _structured_impairment_values(row)

    prior_section = ""
    if prior_milestones:
        sorted_keys = sorted(prior_milestones)
        prior_section = (
            f"\nMilestones already achieved in earlier age periods "
            f"({n_prior} total — these are already cumulated):\n"
            + ", ".join(sorted_keys)
            + "\n"
        )

    return f"""Estimate motor input values for this record:

[Motorical Development Record]

Age: {age}
GMFCS level: {gmfcs_labels.get(gmfcs_int, str(gmfcs_int))}
Milestone ceiling (possible_milestones): {possible}
Impairment ceiling (n_named): {n_named}
Prior cumulative milestones (from earlier age periods): {n_prior}
{prior_section}
[Structured baseline — what the checkboxes captured]
  Milestones in checkboxes this period : {n_struct_milestones}
  Cumulative if only using checkboxes  : {n_prior + n_struct_milestones}
  Impairments with non-zero rating     : {struct_n_selected}
  Sum of impairment ratings            : {struct_sum_ratings:.1f}

Gross motor development (this period):
{clean_text(row.get("gross_motor_development")) or "unknown"}

Fine motor development (this period):
{clean_text(row.get("fine_motor_development")) or "unknown"}

Motorical impairments (lower):
{clean_text(row.get("motorical_impairments_lower")) or "unknown"}

Motorical impairments (upper):
{clean_text(row.get("motorical_impairments_upper")) or "unknown"}

Story:
{clean_text(row.get("story")) or "(no story provided)"}
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a specialist in pediatric motor development for children with cerebral palsy.

Your job is NOT to compute scores. Your job is to estimate three input values
as accurately as possible, combining the structured checkbox data with anything
the free-text story reveals. Python will apply the scoring formula afterwards.

══════════════════════════════════════════════════════════════
THE THREE VALUES YOU MUST ESTIMATE
══════════════════════════════════════════════════════════════

1. estimated_cum_milestones
   The total number of UNIQUE motor milestones this child has achieved
   cumulatively up to and including this age period.
   - Start from the checkbox data: prior milestones + new checkboxes this period.
   - Then adjust UPWARD if the story describes specific motor abilities that
     are not represented in the checkboxes (e.g. "runs short distances", 
     "climbs stairs", "uses scissors").
   - Adjust DOWNWARD if the story makes clear that a checked milestone was
     over-reported (e.g. "those first steps were only in the clinic with a
     therapist holding both hands").
   - Do NOT add vague story phrases ("doing well", "making progress") —
     only add milestones you can name specifically.
   - Never exceed the milestone ceiling given in the prompt.

2. estimated_n_selected
   The number of distinct impairment domains with a non-zero severity.
   - Start from the structured impairment fields (count non-zero entries).
   - Add an impairment domain if the story describes a specific functional
     limitation that is NOT already in the checkboxes.
   - Do NOT add impairments for vague phrases.

3. estimated_sum_ratings
   The sum of all non-zero severity ratings (each 1–5).
   - Start from the structured sum.
   - Adjust individual ratings up or down if the story provides credible
     clinical evidence that the checkbox rating was inaccurate
     (e.g. "spasticity reduced markedly after botox last month",
      "weakness is a major daily barrier").
   - Keep adjustments conservative — only change a rating if the story
     gives a specific, concrete reason.

══════════════════════════════════════════════════════════════
RULES
══════════════════════════════════════════════════════════════
- The structured baseline shown in the prompt is your starting point,
  not something to override lightly.
- Prefer small, justified adjustments over large unsupported ones.
- If the story adds nothing concrete, return the structured baseline values.
- List every adjustment in story_adjustments with a brief explanation.
- Return raw JSON only — no markdown fences.
- Output MUST be in English.
"""


def estimate_motor_inputs(text: str, model_name: str = MODEL_NAME) -> MotorInputEstimate:
    """
    Call the LLM and return estimated input values.

    Args:
        text:       Formatted prompt text for one record.
        model_name: Ollama model name.

    Returns:
        MotorInputEstimate with the three estimated input values.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user",      "content": ex["input"]})
        messages.append({"role": "assistant", "content": ex["output"]})
    messages.append({"role": "user", "content": text})

    def _call():
        return chat(
            model=model_name,
            messages=messages,
            format=MotorInputEstimate.model_json_schema(),
            options={"num_predict": 512, "temperature": 0.0},
        )

    for attempt in range(CHAT_MAX_RETRIES + 1):
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                response = executor.submit(_call).result(timeout=CHAT_TIMEOUT_SECONDS)
            break
        except FuturesTimeoutError:
            print(f"    Timeout attempt {attempt + 1}/{CHAT_MAX_RETRIES + 1}, retrying...")
            if attempt == CHAT_MAX_RETRIES:
                raise RuntimeError(
                    f"LLM timeout after {CHAT_MAX_RETRIES + 1} attempts"
                )

    raw      = extract_json_content(response.message.content)
    repaired = repair_json_text(raw)

    try:
        return MotorInputEstimate.model_validate_json(repaired)
    except Exception:
        print("\nRaw output:\n", response.message.content)
        print("\nRepaired:\n", repaired)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# ANALYZE ONE CHILD
# ─────────────────────────────────────────────────────────────────────────────

def analyze_child(
    df: pl.DataFrame,
    introductory_id: str,
    gmfcs_lookup: dict[str, int],
) -> list[dict]:
    """
    Analyze all motorical development rows for one child.

    For each age period:
      1. LLM estimates input values (with story adjustments).
      2. Python applies the exact formula to compute scores.

    Args:
        df:              Motorical development table.
        introductory_id: Child/survey id.
        gmfcs_lookup:    Mapping from introductory_id → GMFCS integer (1–5).

    Returns:
        list[dict]: One result dict per age row, containing both the LLM
                    estimates and the Python-computed scores.
    """
    df_child = df.filter(pl.col("introductory_id") == introductory_id).sort("age")

    if df_child.height == 0:
        return []

    gmfcs_int = gmfcs_lookup.get(introductory_id, 3)

    child_results: list[dict] = []
    seen_milestones: set[str] = set()   # cumulative keys from structured data

    for row in df_child.iter_rows(named=True):
        age = row.get("age")
        row_start = time.perf_counter()

        possible = _possible_milestones(int(age), gmfcs_int)
        n_named  = _n_named(int(age), gmfcs_int)

        text_input = build_md_input(row, gmfcs_int=gmfcs_int, prior_milestones=seen_milestones or None)
        print(
            f"    age={age}, GMFCS={gmfcs_int}, "
            f"prior={len(seen_milestones)}, ceiling={possible}, n_named={n_named}"
        )

        try:
            estimate = estimate_motor_inputs(text_input)
        except RuntimeError as e:
            print(f"    Age {age} SKIPPED: {e}")
            seen_milestones |= _extract_row_milestone_keys(row)
            continue
        except Exception as e:
            print(f"    Age {age} ERROR: {e}")
            seen_milestones |= _extract_row_milestone_keys(row)
            continue

        # ── Apply exact formula in Python ──────────────────────────────────
        scores = apply_formula(
            cum_milestones = estimate.estimated_cum_milestones,
            n_selected     = estimate.estimated_n_selected,
            sum_ratings    = estimate.estimated_sum_ratings,
            possible       = possible,
            n_named        = n_named,
        )

        elapsed = time.perf_counter() - row_start
        print(
            f"    Done in {elapsed:.2f}s. "
            f"milestones={estimate.estimated_cum_milestones}/{possible} → {scores['milestone_score']:.3f} | "
            f"imp={estimate.estimated_n_selected}/{n_named} sum={estimate.estimated_sum_ratings:.1f} → {scores['impairment_score']:.3f} | "
            f"combined={scores['combined_score']:.3f}"
        )

        # Structured baseline for audit
        struct_n_sel, struct_sum = _structured_impairment_values(row)
        struct_milestones = len(seen_milestones) + len(_extract_row_milestone_keys(row))
        struct_scores = apply_formula(struct_milestones, struct_n_sel, struct_sum, possible, n_named)

        child_results.append({
            "introductory_id":              row.get("introductory_id"),
            "age":                          age,
            "gmfcs_int":                    gmfcs_int,
            # LLM estimates
            "est_cum_milestones":           estimate.estimated_cum_milestones,
            "est_n_selected":               estimate.estimated_n_selected,
            "est_sum_ratings":              estimate.estimated_sum_ratings,
            # Scores from LLM-adjusted inputs (main output)
            "llm_milestone_score":          scores["milestone_score"],
            "llm_impairment_score":         scores["impairment_score"],
            "llm_combined_score":           scores["combined_score"],
            # Intermediate values for debugging
            "presence_ratio":               scores["presence_ratio"],
            "severity_ratio":               scores["severity_ratio"],
            "impairment_burden":            scores["impairment_burden"],
            # Structured-only baseline (for comparing LLM vs raw checkboxes)
            "struct_milestone_score":       struct_scores["milestone_score"],
            "struct_impairment_score":      struct_scores["impairment_score"],
            "struct_combined_score":        struct_scores["combined_score"],
            # Story adjustment metadata
            "confidence":                   estimate.confidence,
            "story_adjustments":            " | ".join(estimate.story_adjustments) if estimate.story_adjustments else "",
            "summary":                      estimate.summary,
        })

        # Accumulate structured milestone keys for the next age period
        seen_milestones |= _extract_row_milestone_keys(row)

    return child_results


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def write_text_report(
    results_df: pl.DataFrame,
    output_path: Path,
    total_elapsed_time: float,
) -> None:
    lines: list[str] = ["LLM MOTOR SCORE ANALYSIS (Roll 2 — story-adjusted inputs)", "=" * 80,
                        f"Total runtime: {total_elapsed_time:.2f} seconds", ""]

    if results_df.height == 0:
        lines.append("No results were generated.")
    else:
        for row in results_df.sort(["introductory_id", "age"]).iter_rows(named=True):
            lines.append(f"ID: {row['introductory_id']}  |  Age: {row['age']}  |  GMFCS: {row.get('gmfcs_int', '?')}")
            lines.append(
                f"  Milestones  : est={row['est_cum_milestones']}  "
                f"→ llm_score={row['llm_milestone_score']:.3f}  "
                f"(struct_only={row['struct_milestone_score']:.3f})"
            )
            lines.append(
                f"  Impairments : est_n={row['est_n_selected']}  est_sum={row['est_sum_ratings']:.1f}  "
                f"→ llm_score={row['llm_impairment_score']:.3f}  "
                f"(struct_only={row['struct_impairment_score']:.3f})"
            )
            lines.append(
                f"  Combined    : llm={row['llm_combined_score']:.3f}  "
                f"struct_only={row['struct_combined_score']:.3f}"
            )
            lines.append(f"  Confidence  : {row['confidence']}")
            if row["story_adjustments"]:
                lines.append(f"  Story adj.  : {row['story_adjustments']}")
            lines.append(f"  Summary     : {row['summary']}")
            lines.append("-" * 80)

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    total_start = time.perf_counter()

    conn = get_connection()
    data = load_data(conn)
    df              = data["motorical_development"]
    introductory_df = data["introductory"]

    gmfcs_lookup: dict[str, int] = {
        uid: _gmfcs_int_from_str(lvl)
        for uid, lvl in zip(
            introductory_df["id"].to_list(),
            introductory_df["gmfcs_lvl"].to_list(),
        )
    }

    all_results: list[dict] = []

    for introductory_id in INTRODUCTORY_IDS:
        child_start = time.perf_counter()
        print(f"Processing {introductory_id}  (GMFCS={gmfcs_lookup.get(introductory_id, '?')})")

        try:
            child_results = analyze_child(df, introductory_id, gmfcs_lookup)
            if not child_results:
                print(f"  No rows found.")
                continue
            all_results.extend(child_results)
            print(f"  Done. {len(child_results)} row(s) in {time.perf_counter() - child_start:.2f}s.")
        except Exception as e:
            print(f"  Error: {e}")

    results_df = pl.DataFrame(all_results) if all_results else pl.DataFrame()

    if results_df.height > 0:
        results_df.write_csv(OUTPUT_CSV_PATH)

    total_elapsed = time.perf_counter() - total_start
    write_text_report(results_df, OUTPUT_TXT_PATH, total_elapsed)

    print(f"\nFinished in {total_elapsed:.2f}s.")
    print(f"Report : {OUTPUT_TXT_PATH}")
    if results_df.height > 0:
        print(f"CSV    : {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()