from ollama import chat
from pydantic import BaseModel
from typing import List


# Define structured Output:

class PercivedEffect(BaseModel):
  direction: str # positive | negative | mixed | no_clear_effect | unknown
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

# Function to run analysis:
def analyze_child_story(text: str) -> ChildAnalysis:

  system_prompt = """
You are extracting structured research data from parent-reported survey responses about training and development in children with cerebral palsy.

IMPORTANT CONTEXT:
The input text is a COMBINATION of multiple responses from the same child across different time periods (e.g., different ages).

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
    model='qwen3.5:9b',
    messages=[{'role': 'system', 'content': system_prompt},
              {"role": "user", "content": user_prompt}
    ],
    format=ChildAnalysis.model_json_schema(),
  )

  return ChildAnalysis.model_validate_json(response.message.content)

def main():
  print("\nPaste the child's combined story (multiple texts).")
  print("Press ENTER twice when done:\n")

  lines = []
  while True:
      line = input()
      if line == "":
          break
      lines.append(line)

  combined_text = "\n".join(lines)

  print("\n--- Running LLM analysis ---\n")

  result = analyze_child_story(combined_text)

  print("\n--- Structured Output ---\n")
  print(result.model_dump_json(indent=2, ensure_ascii=False))


if __name__ == "__main__":
  main()