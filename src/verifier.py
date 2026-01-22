"""Verifies if the model's answer have grounding evidence in the provided context"""

import math
from nli import nli_entailment_check, NLIModel, EntailmentResult

nli_model= NLIModel()

def numeric_answer(pred, gold, eps=1e-3):
    """
    Detects if numeric answer is within tolerance of gold answer.

    Scale-invariant numeric match
    Referred from: T^2 RAGBench Number Match
    """

    def clean_number(x):
        if isinstance(x, str):
            x = x.replace("$", "").replace("%", "").replace(",", "").strip()
        return float(x)

    try:
        pred = abs(clean_number(pred))
        gold = abs(clean_number(gold))
    except (ValueError, TypeError):
        return False

    if pred < eps and gold < eps:
        return True
    if pred < eps or gold < eps:
        return False

    if gold == 0:
        return False

    ratio = pred / gold
    scale_adjusted = ratio * (10 ** -round(math.log10(ratio)))

    return abs(1 - scale_adjusted) <= eps


def answer_matches_gold(pred, gold):
    """
    Detects word-to-word match of predicted answer in gold answer.

    Case-insensitive substring match.
    """
    if not pred or not gold:
        return False

    pred = str(pred).lower().strip()
    gold = str(gold).lower().strip()

    return pred in gold or gold in pred



def token_overlap_grounding(pred, gold, threshold=0.5):
    """
    Detects if predicted answer has sufficient token overlap with gold answer.

    Soft token check.
    """
    if not pred or not gold:
        return False

    pred_tokens = set(str(pred).lower().strip().split())
    gold_tokens = set(str(gold).lower().strip().split())

    if not pred_tokens or not gold_tokens:
        return False

    overlap = pred_tokens & gold_tokens
    overlap_ratio = len(overlap) / len(gold_tokens)

    return overlap_ratio >= threshold



def verify_answer(
    pred,
    gold,
    context,
    answer_type="numeric",
    overlap_threshold=0.6
):
    """
    Verifies answer correctness against gold answer.
    Uses NLI only as a grounding / support signal.
    Final verdict is binary: correct / incorrect.
    """
    #NOT_ANSWERABLE CASE

    if isinstance(pred, str) and pred.strip().upper() == "NOT_ANSWERABLE":
        return {
            "is_correct": False,
            "verdict": "HALLUCINATED",
            "details": {
                "verification": "not_answerable_prediction"
            }
        }
    # NUMERIC ANSWERS

    if answer_type == "numeric":
        is_correct = numeric_answer(pred, gold)

        return {
            "is_correct": is_correct,
            "verdict": "GROUNDED" if is_correct else "HALLUCINATED",
            "details": {
                "verification": "numeric_match"
            }
        }

    # TEXTUAL ANSWERS

    #(Gold-based correctness)
    exact_match = answer_matches_gold(pred, gold)
    soft_match = token_overlap_grounding(
        pred,
        gold,
        threshold=overlap_threshold
    )

    is_correct = exact_match or soft_match

    # (NLI-based grounding (auxiliary))
    entailment = nli_entailment_check(
        answer=pred,
        context=context,
        nli_model=nli_model
    )

    is_grounded = entailment.label == "entailed"

    # FINAL VERDICT
    verdict = "GROUNDED" if (is_correct or is_grounded) else "HALLUCINATED"

    return {
        "is_correct": is_correct,
        "verdict": verdict,
        "details": {
            "verification": "gold_match + nli_support",
            "exact_match": exact_match,
            "soft_match": soft_match,
            "nli_label": entailment.label,
            "nli_confidence": entailment.confidence,
            "is_grounded": is_grounded
        }
    }
