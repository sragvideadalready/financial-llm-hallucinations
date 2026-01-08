from dataclasses import dataclass
from typing import List, Literal, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@dataclass
class EntailmentResult:
    label: Literal["entailed", "contradicted", "unsupported"]
    confidence: float
    supporting_chunks: List[int]

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
            0: "contradicted",  
            1: "neutral",
            2: "entailed",      
        }

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


def nli_entailment_check(answer:str, context:Dict, nli_model:NLIModel, )->EntailmentResult:
    '''Checks if the answer is entailed by the given context using NLI model.
    Returns entailed/ neutral/ contradicted label'''

    chunks= build_context_chunks(context)

    best_entailment= (0.0, None)
    best_contradiction= (0.0, None)

    for idx, chunk in enumerate(chunks):
        inputs= nli_model.tokenizer( premise= chunk, hypothesis= answer, truncation= True, padding= True, return_tensors="pt").to(nli_model.device)

        outputs= nli_model.model(**inputs)
        probs= torch.softmax(outputs.logits, dim=-1).squeeze().cpu().detach().numpy()

        label_id = torch.argmax(probs).item()
        label = nli_model.label_map[label_id]
        confidence = probs[label_id].item()

        if label == "entailed" and confidence > best_entailment[0]:
            best_entailment= (confidence, idx)

        if label == "contradicted" and confidence > best_contradiction[0]:
            best_contradiction= (confidence, idx)

        # Aggregate the logic:
        entailment_threshold= 0.7
        contradiction_threshold= 0.75
        if best_entailment[1] is not None and best_entailment[0]> entailment_threshold:
            return EntailmentResult(
                label="entailed",
                confidence= best_entailment[0],
                supporting_chunks= [best_entailment[1]]
            )
        elif best_contradiction[1] is not None and best_contradiction[0]> contradiction_threshold:
            return EntailmentResult(
                label="contradicted",
                confidence= best_contradiction[0],
                supporting_chunks= [best_contradiction[1]]
            )
    return EntailmentResult(
        label="unsupported",
        confidence= max(best_entailment[0], best_contradiction[0]),
        supporting_chunks= []
    )

    


