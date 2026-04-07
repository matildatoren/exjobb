from ollama import chat
from pydantic import BaseModel
from typing import List, Literal
import sys
from pathlib import Path
# ── resolve imports regardless of where the script is called from ──────────
ROOT = Path(__file__).resolve().parents[1]   # src/
sys.path.append(str(ROOT))

import polars as pl
from connect_db import get_connection
from dataloader import load_data


# Define structured Output:
class PercivedEffect(BaseModel):
  direction: Literal["positive", 
      "negative",
      "mixed",
      "no_clear_effect",
      "unknown",] 
  summary: str

class ChildAnalysis(BaseModel):
  training_types_mentioned: List[str]
  outcome_domains_mentioned: List[str]
  percieved_training_effect: PercivedEffect
  development_over_time: str
  barriers: List[str]
  facilitators: List[str]
  respondent_expresses_uncertainty: bool
  supporting_quotes: List[str]


def clean_text(value: object) -> str | None:
  """
  Clean a database value and return a usable string.

  Args: 
    value (object) : Raw database value.
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

  if intro_df.height > 0:
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



# Function to run analysis on a combined story text for 1 child:
def analyze_child_story(text: str, model_name: str = "qwen3.5:9b") -> ChildAnalysis:

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
- If unclear → use "unknown".
- Be conservative.

Return ONLY valid JSON following the schema.
"""
  user_prompt = f"""
Extract structured information from this combined survey response:

{text}
"""

  response = chat(
    model= model_name,
    messages=[{'role': 'system', 'content': system_prompt},
              {"role": "user", "content": user_prompt}
    ],
    format=ChildAnalysis.model_json_schema(),
  )

  return ChildAnalysis.model_validate_json(response.message.content)

def main()-> None:
  conn = get_connection()
  data = load_data(conn)

  introductory_id = input("Enter introductory_id: ").strip()
  combined_text = build_child_story(data, introductory_id)

  if not combined_text:
      print("No story data found for this introductory_id.")
      return

  print("\n--- Combined story sent to model ---\n")
  print(combined_text)

  print("\n--- Running LLM analysis ---\n")
  result = analyze_child_story(combined_text)

  print("\n--- Structured output ---\n")
  print(result.model_dump_json(indent=2))


if __name__ == "__main__":
   main()