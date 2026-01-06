"""Verifies if the model's answer have grounding evidence in the provided context"""

import math

def numeric_answer(pred, gold, eps=1e-3):

    '''detects if numeric answer is 
    within tolerance of gold answer
    
    Scale-invariant numeric match
    Reffered from: T^2 RAGBench Number Match'''


    try:
        pred= abs(float(pred))
        gold= abs(float(gold))
    except:
        return False
    
    if pred<eps and gold<eps:
        return True
    
    if gold==0:
        return False
    
    ratio= pred/gold
    scale_adjusted = ratio * (10 ** -round(math.log10(ratio)))

    return abs(1 - scale_adjusted) <= eps
    
def answer_in_context(pred, context):

    '''detects word to word match of answer in context. 

    Case insensitive match.
    '''
    if not pred or not context:
        return False

    answer=str(pred).lower().strip()
    context= context.lower()

    return answer in context


def token_overlap_grounding(answer, context, threshold=0.5):

    '''detects if answer has sufficient token overlap with context
    
    Soft token check'''

    if not answer or not context:
        return False
    
    answer_tokens= set(str(answer).lower().strip().split())
    context_tokens= set(str(context).lower().strip().split())

    if not answer_tokens or not context_tokens:
        return False
    
    overlap= answer_tokens & context_tokens
    overlap_ratio= len(overlap) / len(answer_tokens)
    return overlap_ratio >= threshold


def verify_answer(
    pred,
    gold,
    context,
    answer_type="numeric",
    overlap_threshold=0.6
):
    """
    Verifies correctness and grounding of an answer using
    entailment-based verification (AIS / FEVER style).

    Returns factual verdict:
    - correct_grounded
    - incorrect_grounded
    - incorrect_ungrounded
    """

    # -----------------------
    # NUMERIC ANSWERS
    # -----------------------
    if answer_type == "numeric":
        is_correct = numeric_answer(pred, gold)
        is_grounded = answer_in_context(pred, context)

        if is_correct and is_grounded:
            verdict = "correct_grounded"
        elif is_correct and not is_grounded:
            verdict = "correct_ungrounded"
        elif not is_correct and is_grounded:
            verdict = "incorrect_grounded"
        else:
            verdict = "incorrect_ungrounded"

        return {
            "is_correct": is_correct,
            "is_grounded": is_grounded,
            "verdict": verdict,
            "details": {
                "verification": "numeric_match"
            }
        }

    # -----------------------
    # TEXTUAL ANSWERS
    # -----------------------
    entailment = entailment_check(pred, context, overlap_threshold)

    if entailment is True:
        # Context entails answer
        is_correct = True
        is_grounded = True
        verdict = "correct_grounded"

    elif entailment is False:
        # Context contradicts answer
        is_correct = False
        is_grounded = True
        verdict = "incorrect_grounded"

    else:
        # Unsupported / hallucinated
        is_correct = False
        is_grounded = False
        verdict = "incorrect_ungrounded"

    return {
        "is_correct": is_correct,
        "is_grounded": is_grounded,
        "verdict": verdict,
        "details": {
            "verification": "entailment_based",
            "entailment": entailment
        }
    }

def entailment_check(answer, context, overlap_threshold=0.6):
    '''Rashkin et.al 2023 entailment based verification
    Fabri et al., EMNLP  2022'''

    """
    Weak entailment check inspired by AIS / QAFactEval.
    Uses grounding + semantic overlap as proxy.
    """

    if not answer or not context:
        return None

    answer_tokens = set(str(answer).lower().split())
    context_tokens = set(context.lower().split())

    if not answer_tokens:
        return None

    overlap = answer_tokens & context_tokens
    overlap_ratio = len(overlap) / len(answer_tokens)

    if overlap_ratio >= overlap_threshold:
        return True  # weakly entailed

    return None
