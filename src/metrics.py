"""Metrics for evaluating financial LLM hallucinations.
    
    Metrics are defined at individual claim level and operate 
    relative to the input context provided."""
from typing import List, Dict
GROUNDED = "grounded"
HALLUCINATED = "hallucinated"

INTRINSIC= "intrinsic_halu"
EXTRINSIC = "extrinsic_halu"

def compute_claim_accuracy(pred_labels, true_labels)-> float:
    
    '''
    Accuracy = (1/N) * sum( t̂(x,c) == t(x,c) )
    '''
    assert len(pred_labels) == len(true_labels)
    correct= sum(1 for yt, yp in zip(true_labels, pred_labels) if yt == yp)
    return correct / len(true_labels) if true_labels else 0.0

def compute_hallucination_rate(true_labels):
    '''
    hallucination Rate = (1/N) * sum( 1{ t(x,c) == "hallucinated" } )
    '''
    if not true_labels:
        return 0.0
    hallucinated_count= sum(1 for true_label in true_labels if true_label == HALLUCINATED)
    return hallucinated_count / len(true_labels)


def error_type_analysis(true_labels: List[str], pred_labels: List[str]) -> Dict[str, int]:
    """
    Returns counts of FP and FN.
    FP: grounded → predicted hallucinated
    FN: hallucinated → predicted grounded
    """
    assert len(true_labels) == len(pred_labels),

    fp = 0
    fn = 0

    for yt, yp in zip(true_labels, pred_labels):
        if yt == GROUNDED and yp == HALLUCINATED:
            fp += 1
        elif yt == HALLUCINATED and yp == GROUNDED:
            fn += 1

    return {
        "false_positives": fp,
        "false_negatives": fn
    }


def intrinsic_extrinsic_breakdown(
    fine_grained_labels: List[str]
) -> Dict[str, int]:
    """
    Input: original annotation labels (intrinsic/extrinsic).
    """
    return {
        INTRINSIC: sum(1 for l in fine_grained_labels if l == INTRINSIC),
        EXTRINSIC: sum(1 for l in fine_grained_labels if l == EXTRINSIC),
    }
