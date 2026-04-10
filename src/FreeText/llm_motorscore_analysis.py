# Important note:
# This is not an objective clinical measure, but an LLM-derived summary score.

from typing import List, Literal

import polars as pl
from ollama import chat
from pydantic import BaseModel

import sys
from pathlib import Path

# Resolve imports regardless of where the script is called from
ROOT = Path(__file__).resolve().parents[1]  # src/
sys.path.append(str(ROOT))

from connect_db import get_connection
from dataloader import load_data


# ---------------------------
# SETTINGS
# ---------------------------

MODEL_NAME = "qwen3.5:9b"

INTRODUCTORY_IDS = [
    "c0990a55-916e-47ba-b29a-aee83d9f33c9", 
    "65ab3206-7371-4471-845c-6d238050494f",
    "cd26a009-6e51-4372-b151-b7d2bb8b7183"
]

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_TXT_PATH = OUTPUT_DIR / "llm_motorscore_results.txt"
OUTPUT_CSV_PATH = OUTPUT_DIR / "llm_motorscore_results.csv"


# ---------------------------
# STRUCTURED OUTPUT
# ---------------------------

class MotorAssessment(BaseModel):
    motor_score_1_to_10: int
    confidence: Literal["low", "medium", "high"]
    summary: str
    gross_motor_level: str
    fine_motor_level: str
    impairment_severity: str
    supporting_evidence: List[str]


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

    Args:
        text (str): Formatted input text for one record.
        model_name (str): Ollama model name.

    Returns:
        MotorAssessment: Structured LLM assessment.
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

SCALE GUIDELINES:
1 = very limited motor function, severe impairments
3 = major limitations, very few abilities
5 = moderate motor function with clear limitations
7 = relatively good motor abilities with some impairments
10 = very strong motor function, minimal limitations

You do NOT need to use every number, but stay consistent.

LANGUAGE:
- Input may be multilingual.
- Output MUST be in English.

Return ONLY valid JSON.
"""

    user_prompt = f"""
Assess motor development for this record:

{text}
"""

    response = chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        format=MotorAssessment.model_json_schema(),
    )

    return MotorAssessment.model_validate_json(response.message.content)


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
        text_input = build_md_input(row)
        result = analyze_md_row(text_input)

        child_results.append({
            "introductory_id": row.get("introductory_id"),
            "age": row.get("age"),
            "llm_motor_score": result.motor_score_1_to_10,
            "confidence": result.confidence,
            "summary": result.summary,
            "gross_motor_level": result.gross_motor_level,
            "fine_motor_level": result.fine_motor_level,
            "impairment_severity": result.impairment_severity,
            "supporting_evidence": " | ".join(result.supporting_evidence),
        })

    return child_results


# ---------------------------
# WRITE TEXT REPORT
# ---------------------------

def write_text_report(results_df: pl.DataFrame, output_path: Path) -> None:
    """
    Write a human-readable text report.

    Args:
        results_df (pl.DataFrame): Final results table.
        output_path (Path): Path to output text file.

    Returns:
        None
    """
    lines: list[str] = []
    lines.append("LLM MOTOR SCORE ANALYSIS")
    lines.append("=" * 80)
    lines.append("")

    if results_df.height == 0:
        lines.append("No results were generated.")
    else:
        for row in results_df.sort(["introductory_id", "age"]).iter_rows(named=True):
            lines.append(f"Introductory ID: {row['introductory_id']}")
            lines.append(f"Age: {row['age']}")
            lines.append(f"LLM motor score: {row['llm_motor_score']}")
            lines.append(f"Confidence: {row['confidence']}")
            lines.append(f"Gross motor level: {row['gross_motor_level']}")
            lines.append(f"Fine motor level: {row['fine_motor_level']}")
            lines.append(f"Impairment severity: {row['impairment_severity']}")
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
    conn = get_connection()
    data = load_data(conn)
    df = data["motorical_development"]

    all_results: list[dict] = []

    for introductory_id in INTRODUCTORY_IDS:
        print(f"Processing introductory_id: {introductory_id}")

        try:
            child_results = analyze_child(df, introductory_id)

            if not child_results:
                print(f"  No motorical development rows found for {introductory_id}")
                continue

            all_results.extend(child_results)
            print(f"  Done. Processed {len(child_results)} row(s).")

        except Exception as e:
            print(f"  Error for {introductory_id}: {e}")

    results_df = pl.DataFrame(all_results) if all_results else pl.DataFrame()

    if results_df.height > 0:
        results_df.write_csv(OUTPUT_CSV_PATH)

    write_text_report(results_df, OUTPUT_TXT_PATH)

    print("\nFinished.")
    print(f"Text report saved to: {OUTPUT_TXT_PATH}")
    if results_df.height > 0:
        print(f"CSV file saved to: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()