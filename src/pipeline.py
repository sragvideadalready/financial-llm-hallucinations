from typing import Dict
from verifier import verify_answer
from metrics import GROUNDED, HALLUCINATED


def run_mitigation_pipeline(sample_data: Dict) -> Dict:
    """
    Runs the hallucination mitigation pipeline on a single sample.

    Args:
        sample_data: {
            "prediction": str,
            "gold_answer": str,
            "context": Dict,
            "answer_type": "numeric" | "text"
        }

    Returns:
        {
            "label": GROUNDED | HALLUCINATED,
            "verifier_output": Dict
        }
    """

    verifier_output = verify_answer(
        pred=sample_data["prediction"],
        gold=sample_data.get("gold_answer"),
        context=sample_data["context"],
        answer_type=sample_data.get("answer_type", "text")
    )

    # --- Mitigation decision ---
    verdict = verifier_output.get("verdict")

    if verdict == "contradicted":
        final_label = HALLUCINATED
    elif verifier_output["is_grounded"]:
        final_label = GROUNDED
    else:
        final_label = HALLUCINATED


    return {
        "label": final_label,
        "verifier_output": verifier_output
    }
