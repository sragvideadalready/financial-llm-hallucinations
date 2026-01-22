import csv
from verifier import verify_answer
import json
from metrics import compute_claim_accuracy, compute_hallucination_rate, error_type_analysis
def load_samples(path:str):
    """Load samples from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def predict_labels(llm_model:str, samples_path:str, model_answers_path:str, output_path:str):
    samples= load_samples(samples_path)
    model_answers= load_samples(model_answers_path)
    
    model_answer_map= {
        item["id"]: item["model_answer"] for item in model_answers
    }

    with open(output_path, "w", encoding="utf-8") as out_f:
        i=1
        for sample in samples:
            print(f"working on question {i}")
            i+=1
            qid= sample["id"]

            if qid not in model_answer_map:
                continue

            pred_answer= model_answer_map[qid]
            result= verify_answer(
                pred= pred_answer,
                gold= sample["answer"],
                context= sample.get("context", {}),
                answer_type= sample.get("answer_type", "numeric")
            )

            record= {
                "id": qid,
                "model_name": llm_model,
                "answer_type": sample["answer_type"],

                "pred_label": result["verdict"],

                "verification_details" : result["details"]
            }

            out_f.write(json.dumps(record) + "\n")

    print(f"Predicted lables saved for model {llm_model} to {output_path}")

def run_pipeline(llm_model:str, predicted_labels_path:str, gold_labels_csv_path:str, output_path:str):

    gold_lables_map= {}
    
    with open(gold_labels_csv_path, "r", encoding="utf-8") as f:
        reader= csv.DictReader(f)
        for row in reader:
            gold_lables_map[row["id"]]= row["true_label"]

    
    records= []
    true_labels= []
    pred_labels= []

    num_true= []
    num_pred= []

    text_true= []
    text_pred= []

    with open(predicted_labels_path, "r", encoding="utf-8") as f:
        for line in f:
            record= json.loads(line)
            qid= record["id"]
            if qid not in gold_lables_map:
                continue

            true_label= gold_lables_map[qid]
            pred_label= record["pred_label"]

            true_labels.append(true_label)
            pred_labels.append(pred_label)

            if record["answer_type"] == "numeric":
                num_true.append(true_label)
                num_pred.append(pred_label)
            if record["answer_type"] == "textual":
                text_true.append(true_label)
                text_pred.append(pred_label)
            
            records.append(record)
        
        overall_metrics= {
            "accuracy": compute_claim_accuracy(true_labels, pred_labels),
            "hallucination_rate": compute_hallucination_rate(pred_labels), 
            "error_analysis": error_type_analysis(true_labels, pred_labels)
        }

        numeric_metrics = {
            "accuracy": compute_claim_accuracy(num_pred, num_true),
            "hallucination_rate_model": compute_hallucination_rate(num_pred)
        }
        textual_metrics = {
            "accuracy": compute_claim_accuracy(text_pred, text_true),
            "hallucination_rate_model": compute_hallucination_rate(text_pred)
        }

        final_output= {
            "model_name": llm_model,
            "num_samples": len(true_labels),

            "overall_metrics": overall_metrics,
            "numeric_metrics": numeric_metrics,
            "textual_metrics": textual_metrics,
        }

        with open(output_path, "a", encoding="utf-8") as out_f:
            json.dump(final_output, out_f, indent=2)
        
        print(f"Saved evaluation metrics for LLM model {llm_model} to {output_path}")


# predict_labels(llm_model="qwen2.5:7b", samples_path="data/samples.json", model_answers_path="model_answers_qwen2.5_7b.json", output_path="predicted_labels_qwen2.5_7b.jsonl")  

# run_pipeline(llm_model="qwen2.5:7b", predicted_labels_path="predicted_labels_qwen2.5_7b.jsonl", gold_labels_csv_path="hallucination_annotation_qwen2.5_7b.csv", output_path="results/evaluation_metrics.jsonl")

# predict_labels(llm_model= "qwen2-math:7b", samples_path="data/samples.json", model_answers_path="model_answers_qwen2_math_7b.json", output_path="predicted_labels_qwen2_math_7b.jsonl")

run_pipeline(llm_model="qwen2-math:7b", predicted_labels_path="predicted_labels_qwen2_math_7b.jsonl", gold_labels_csv_path="hallucination_annotation_qwen2_math_7b.csv", output_path="results/evaluation_metrics.jsonl")
