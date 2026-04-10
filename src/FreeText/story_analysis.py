from ollama import chat
from pydantic import BaseModel
from typing import List, Literal
import sys
from pathlib import Path

# Resolve imports regardless of where the script is called from
ROOT = Path(__file__).resolve().parents[1]  # src/
sys.path.append(str(ROOT))

import polars as pl
from connect_db import get_connection
from dataloader import load_data


# ---------------------------
# SETTINGS
# ---------------------------

MODEL_NAME = "qwen3.5:9b"

INTRODUCTORY_IDS = [
    "c0990a55-916e-47ba-b29a-aee83d9f33c9",
    # "add-more-ids-here",
]

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_TXT_PATH = OUTPUT_DIR / "llm_story_analysis_results.txt"
OUTPUT_CSV_PATH = OUTPUT_DIR / "llm_story_analysis_results.csv"


# ---------------------------
# DEFINE STRUCTURED OUTPUT
# ---------------------------

class PerceivedEffect(BaseModel):
    direction: Literal[
        "positive",
        "negative",
        "mixed",
        "no_clear_effect",
        "unknown",
    ]
    summary: str


class ChildAnalysis(BaseModel):
    training_types_mentioned: List[str]
    outcome_domains_mentioned: List[str]
    perceived_training_effect: PerceivedEffect
    development_over_time: str
    barriers: List[str]
    facilitators: List[str]
    respondent_expresses_uncertainty: bool
    supporting_quotes: List[str]


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
# BUILD COMBINED STORY
# ---------------------------

def build_child_story(data: dict[str, pl.DataFrame], introductory_id: str) -> str:
    """
    Build one combined story for a child using the `story` column
    from all sections and all available ages.

    Args:
        data (dict[str, pl.DataFrame]): Dictionary of loaded survey tables.
        introductory_id (str): Survey identifier for one child.

    Returns:
        str: Combined story text.
    """
    sections: list[str] = []

    # Introductory table uses `id` instead of `introductory_id`
    intro_df = data["introductory"].filter(
        pl.col("id") == introductory_id
    )

    if intro_df.height > 0 and "story" in intro_df.columns:
        intro_story = clean_text(intro_df["story"][0])
        if intro_story:
            sections.append("[INTRODUCTORY]")
            sections.append(intro_story)
            sections.append("")

    def process_age_table(df: pl.DataFrame, table_name: str) -> None:
        """
        Add all stories from one age-based table to the combined story.

        Args:
            df (pl.DataFrame): Age-based survey table.
            table_name (str): Section name used in the combined text.

        Returns:
            None
        """
        if "story" not in df.columns:
            return

        rows = (
            df.filter(pl.col("introductory_id") == introductory_id)
            .sort("age")
        )

        for row in rows.iter_rows(named=True):
            story = clean_text(row.get("story"))
            age = row.get("age")

            if story:
                sections.append(f"[{table_name} | age={age}]")
                sections.append(story)
                sections.append("")

    process_age_table(data["home_training"], "HOME_TRAINING")
    process_age_table(data["intensive_therapies"], "INTENSIVE_THERAPIES")
    process_age_table(data["motorical_development"], "MOTORICAL_DEVELOPMENT")

    return "\n".join(sections).strip()


# ---------------------------
# LLM ANALYSIS
# ---------------------------

def analyze_child_story(text: str, model_name: str = MODEL_NAME) -> ChildAnalysis:
    """
    Run LLM analysis on one combined child story.

    Args:
        text (str): Combined story text for one child.
        model_name (str): Ollama model name.

    Returns:
        ChildAnalysis: Structured analysis of the child's story.
    """
    system_prompt = """
You are extracting structured research data from parent-reported survey responses about training and development in children with cerebral palsy.

IMPORTANT CONTEXT:
The input text is a combination of multiple free-text responses from the same child across different sections and age intervals.
The text may include information from different time periods.
You should analyze the full text as one child-level record, while being careful not to assume details that are not explicitly stated.

This means:
- The text may describe development over time.
- Training and outcomes may vary across periods.
- You should consider the FULL text.

LANGUAGE:
- The input may be in different languages (e.g., Swedish, English, Spanish).
- You must understand the text regardless of language.
- You must ALWAYS return the output in English.

RULES:
- Only extract information explicitly stated.
- Do NOT guess.
- Do NOT make causal claims unless explicitly stated.
- If unclear, use "unknown".
- Be conservative.

Return ONLY valid JSON following the schema.
"""

    user_prompt = f"""
Extract structured information from this combined survey response:

{text}
"""

    response = chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        format=ChildAnalysis.model_json_schema(),
    )

    return ChildAnalysis.model_validate_json(response.message.content)


# ---------------------------
# ANALYZE ONE CHILD
# ---------------------------

def analyze_one_child(data: dict[str, pl.DataFrame], introductory_id: str) -> dict | None:
    """
    Build and analyze the combined story for one child.

    Args:
        data (dict[str, pl.DataFrame]): Loaded survey data.
        introductory_id (str): Survey identifier.

    Returns:
        dict | None: Result dictionary, or None if no story was found.
    """
    combined_text = build_child_story(data, introductory_id)

    if not combined_text:
        return None

    result = analyze_child_story(combined_text)

    return {
        "introductory_id": introductory_id,
        "combined_story": combined_text,
        "training_types_mentioned": " | ".join(result.training_types_mentioned),
        "outcome_domains_mentioned": " | ".join(result.outcome_domains_mentioned),
        "perceived_effect_direction": result.perceived_training_effect.direction,
        "perceived_effect_summary": result.perceived_training_effect.summary,
        "development_over_time": result.development_over_time,
        "barriers": " | ".join(result.barriers),
        "facilitators": " | ".join(result.facilitators),
        "respondent_expresses_uncertainty": result.respondent_expresses_uncertainty,
        "supporting_quotes": " | ".join(result.supporting_quotes),
    }


# ---------------------------
# WRITE TEXT REPORT
# ---------------------------

def write_text_report(results: list[dict], output_path: Path) -> None:
    """
    Write a human-readable text report.

    Args:
        results (list[dict]): List of result dictionaries.
        output_path (Path): Path to output text file.

    Returns:
        None
    """
    lines: list[str] = []
    lines.append("LLM STORY ANALYSIS")
    lines.append("=" * 80)
    lines.append("")

    if not results:
        lines.append("No results were generated.")
    else:
        for row in results:
            lines.append(f"Introductory ID: {row['introductory_id']}")
            lines.append("-" * 80)
            lines.append("COMBINED STORY:")
            lines.append(row["combined_story"])
            lines.append("")
            lines.append("STRUCTURED ANALYSIS:")
            lines.append(f"Training types mentioned: {row['training_types_mentioned']}")
            lines.append(f"Outcome domains mentioned: {row['outcome_domains_mentioned']}")
            lines.append(f"Perceived effect direction: {row['perceived_effect_direction']}")
            lines.append(f"Perceived effect summary: {row['perceived_effect_summary']}")
            lines.append(f"Development over time: {row['development_over_time']}")
            lines.append(f"Barriers: {row['barriers']}")
            lines.append(f"Facilitators: {row['facilitators']}")
            lines.append(
                f"Respondent expresses uncertainty: {row['respondent_expresses_uncertainty']}"
            )
            lines.append(f"Supporting quotes: {row['supporting_quotes']}")
            lines.append("=" * 80)
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------
# MAIN
# ---------------------------

def main() -> None:
    """
    Run story analysis for all ids in INTRODUCTORY_IDS
    and save results to output files.

    Returns:
        None
    """
    conn = get_connection()
    data = load_data(conn)

    all_results: list[dict] = []

    for introductory_id in INTRODUCTORY_IDS:
        print(f"Processing introductory_id: {introductory_id}")

        try:
            result = analyze_one_child(data, introductory_id)

            if result is None:
                print(f"  No story data found for {introductory_id}")
                continue

            all_results.append(result)
            print("  Done.")

        except Exception as e:
            print(f"  Error for {introductory_id}: {e}")

    write_text_report(all_results, OUTPUT_TXT_PATH)

    if all_results:
        csv_rows = []
        for row in all_results:
            csv_rows.append({
                "introductory_id": row["introductory_id"],
                "training_types_mentioned": row["training_types_mentioned"],
                "outcome_domains_mentioned": row["outcome_domains_mentioned"],
                "perceived_effect_direction": row["perceived_effect_direction"],
                "perceived_effect_summary": row["perceived_effect_summary"],
                "development_over_time": row["development_over_time"],
                "barriers": row["barriers"],
                "facilitators": row["facilitators"],
                "respondent_expresses_uncertainty": row["respondent_expresses_uncertainty"],
                "supporting_quotes": row["supporting_quotes"],
            })

        pl.DataFrame(csv_rows).write_csv(OUTPUT_CSV_PATH)

    print("\nFinished.")
    print(f"Text report saved to: {OUTPUT_TXT_PATH}")
    if all_results:
        print(f"CSV file saved to: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()