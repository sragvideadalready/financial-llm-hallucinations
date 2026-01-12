from metrics import (
    claim_grounding_accuracy,
    hallucination_rate,
    error_type_analysis,
    intrinsic_extrinsic_breakdown,
    GROUNDED,
    HALLUCINATED
)
from typing import List, Dict



def evaluate_model(outputs:List[str], labels: List[str])-> Dict:
    '''
    Runs evaluation metrics over the model predictions. 

    Args:
        outputs : List of model predicted labels ("grounded"/"hallucinated")
        labels : Ground Truth Labels 
    '''
   
