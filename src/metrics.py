"""Metrics for evaluating financial LLM hallucinations.
    
    Metrics are defined at individual claim level and operate 
    relative to the input context provided."""
from typing import List, Dict
GROUNDED = "GROUNDED"
HALLUCINATED = "HALLUCINATED"

INTRINSIC= "INTRINSIC_HALU"
EXTRINSIC = "EXTRINSIC_HALU"

def compute_claim_accuracy(pred_labels, true_labels)-> float:
    
    '''
    Accuracy = (1/N) * sum( t̂(x,c) == t(x,c) )
    '''
    if len(pred_labels) != len(true_labels):
        raise ValueError("pred_labels and true_labels must have the same length")
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


def error_type_analysis(
    true_labels: List[str],
    pred_labels: List[str]
) -> Dict[str, int]:
    """
    False Positive (FP):
        True label = GROUNDED, Predicted = HALLUCINATED
    False Negative (FN):
        True label = HALLUCINATED, Predicted = GROUNDED
    """

    if len(true_labels) != len(pred_labels):
        raise ValueError("true_labels and pred_labels must have the same length")

    fp = 0
    fn = 0

    for y_true, y_pred in zip(true_labels, pred_labels):
        if y_true == GROUNDED and y_pred == HALLUCINATED:
            fp += 1
        elif y_true == HALLUCINATED and y_pred == GROUNDED:
            fn += 1

    return {
        "false_positives": fp,
        "false_negatives": fn
    }
