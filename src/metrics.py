"""Metrics for evaluating financial LLM hallucinations.
    
    Metrics are defined at individual claim level and operate 
    relative to the input context provided."""

def compute_claim_accuracy(pred_labels, true_labels):
    """Computes binary classification accuracy for
    grounded vs hallucinated claims."""

    raise NotImplementedError("Function not yet implemented.")

def compute_hallucination_rate(labels):
    """Computes the rate of hallucinated claims
    in the provided labels."""

    raise NotImplementedError("Function not yet implemented.")

def detect_hallucination(model_answer, context):
    """
    Determines whether a model-generated answer is
    grounded or hallucinated with respect to the context.

    Returns:
        label: one of {"grounded", "hallucinated"}
    """
    raise NotImplementedError("Function not yet implemented.")

