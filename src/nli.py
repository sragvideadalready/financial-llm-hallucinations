from dataclasses import dataclass
from typing import List, Literal, Dict
import torch.nn.functional as F
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@dataclass
class EntailmentResult:
    label: Literal["entailed", "contradicted", "unsupported"]
    confidence: float
    supporting_chunks: List[int]
@dataclass
class NLIResult:
    label: Literal["entailed", "neutral", "contradicted"]
    score: float
class NLIModel:
    def __init__(self, model_name="roberta-large-mnli"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        ).to(self.device)

        self.model.eval()

        # roberta-large-mnli label mapping
        self.label_map = {
            0: "contradiction",  
            1: "neutral",
            2: "entailed",      
        }
    
    def predict(self, premise:str, hypothesis:str)->NLIResult:
        """Run NLI inference on premise and hypothesis. 
        REturns object with .label and .score
        """

        inputs= self.tokenizer(
            premise, 
            hypothesis, 
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        inputs= {k:v.to(self.device) for k,v in inputs.items()}

        with torch.no_grad():
            outputs= self.model(**inputs)
            logits= outputs.logits
        
        probs= F.softmax(logits, dim=-1)[0]

        label_id= torch.argmax(probs).item()
        label= self.label_map[label_id]
        score= probs[label_id].item()

        return NLIResult(
            label= label,
            score= score
        )
def linearize_table(table: List[List[str]])->List[str]:
    '''Coverts a 2D table into linearized string format
    Returns one sentence per row.'''

    if not table or len(table)<2:
        return []
    
    header= table[0]
    rows= table[1:]

    sentences= []
    for row in rows:
        facts=[]
        for h, v in zip(header, row):
            facts.append(f"{h} is {v}")
        sentence= f"In the table, " + ", ".join(facts) + "."
        sentences.append(sentence)
    return sentences

def build_context_chunks(context:Dict)->List[str]:
    '''Divides the given context in the dataset into 
    list of text chunks to use in the NLI model.'''

    chunks = []

    chunks.extend(context.get("pre_text", []))
    chunks.extend(context.get("post_text", []))

    table = context.get("table", [])
    chunks.extend(linearize_table(table))

    return [c for c in chunks if c and c.strip()]


def nli_entailment_check(
    answer: str,
    context: dict,
    nli_model
):
    """
    Checks if the answer is entailed by the given context using an NLI model.
    Returns entailed / neutral / contradicted label with confidence.
    """
    ENTAILEMENT_THRESHOLD = 0.7
    chunks = build_context_chunks(context)

    best_entailment = (0.0, None)
    best_contradiction = (0.0, None)

    for chunk in chunks:
        result = nli_model.predict(
            premise=chunk,
            hypothesis=answer
        )
        # result.label ∈ {"entailed", "neutral", "contradiction"}
        # result.score ∈ [0, 1]

        if result.label == "entailed" and result.score > best_entailment[0]:
            best_entailment = (result.score, chunk)

        elif result.label == "contradicted" and result.score > best_contradiction[0]:
            best_contradiction = (result.score, chunk)

    # ---- Final decision logic ----
    if best_entailment[0] > best_contradiction[0] and best_entailment[0] > ENTAILEMENT_THRESHOLD:
        label = "entailed"
        confidence = best_entailment[0]
        supporting_chunks = [best_entailment[1]] if best_entailment[1] else []

    elif best_contradiction[0] > 0:
        label = "contradicted"
        confidence = best_contradiction[0]
        supporting_chunks = [best_contradiction[1]] if best_contradiction[1] else []

    else:
        label = "neutral"
        confidence = 0.0
        supporting_chunks = []

    return EntailmentResult(
        label=label,
        confidence=confidence,
        supporting_chunks=supporting_chunks
    )

