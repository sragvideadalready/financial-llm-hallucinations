import json 
from typing import List, Dict

from generate_api_answer import clean_answer, generate_api_answer, write_prompt, query_llm, clean_answer, serialize_context

SAMPLES_PATH= "data/samples.json"
OUTPUT_PATH= "model_answers_deepseek_r1.json"

def load_samples(path: str) -> List[Dict]:
    """Load samples from a JSON file."""

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def save_model_answers(answers: List[Dict], path: str):
    """Save model answers to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(answers, f, indent=2, ensure_ascii=False)


def main():
    samples= load_samples(SAMPLES_PATH)

    model_outputs= generate_api_answer(samples)

    save_model_answers(model_outputs, OUTPUT_PATH)
    print(f"Saved model answers to {OUTPUT_PATH}")

if __name__ == "__main__":
   main()
    