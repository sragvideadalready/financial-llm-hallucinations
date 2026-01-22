import json
import csv

# paths
MODEL_ANSWERS_PATH = "model_answers_qwen2_math_7b.json"
GOLD_ANSWERS_PATH = "data/samples.json"
OUTPUT_CSV = "hallucination_annotation_qwen2_math_7b.csv"

# load jsons
with open(MODEL_ANSWERS_PATH, "r", encoding="utf-8") as f:
    model_data = json.load(f)

with open(GOLD_ANSWERS_PATH, "r", encoding="utf-8") as f:
    gold_data = json.load(f)

gold_by_id = {
    item["id"]: item for item in gold_data
}

rows = []

for item in model_data:
    qid = item["id"]

    if qid not in gold_by_id:
        continue  # skip if gold not found

    gold_item = gold_by_id[qid]

    rows.append({
        "id": qid,
        "question": item.get("question", gold_item.get("question", "")),
        "gold_answer": gold_item.get("answer", ""),
        "model_answer": item.get("model_answer", "")
    })

# write csv
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["id", "question", "gold_answer", "model_answer"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ CSV written to {OUTPUT_CSV} with {len(rows)} rows")
