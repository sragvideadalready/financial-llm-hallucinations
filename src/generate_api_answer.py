import json 
import time 
import os
from typing import List, Dict 
import ollama



def serialize_context(context: Dict)-> str:
    """Serialize the context dictionary into a JSON string."""
    parts=[]

    if context.get("pre_text"):
        parts.append("Pre Text:\\n")
        parts.extend(context["pre_text"])
        
    
    if context.get("table"):
        parts.append("\\nTable:\\n")
        for row in context["table"]:
            parts.append(str(row) + "\\n")
        
    if context.get("post_text"):
        parts.append("\\nPost Text:\\n")
        parts.extend(context["post_text"])

    return "\n".join(parts)

def write_prompt(question: str, context: str, answer_type: str) -> str:
    """
    Constructs a strict prompt based on question type.
    """

    serialized_context = serialize_context(context)

    standard_prompt=f"""You are answering a financial question using ONLY the provided context.

Rules:
- You may use values explicitly stated in the context
- You may perform simple arithmetic using those values
- You may apply basic financial assumptions (e.g., linear continuation or proportional change) ONLY if necessary
- Do NOT use any external knowledge
- Do NOT introduce new facts

Output rules:
- Return ONLY the final answer
- NO explanations, NO reasoning text
"""
    if answer_type == "numeric":
            type_prompt= f"""- The answer must be ONLY a numeric value (integer or float).
            - Output answer without any formatting (e.g., no commas, no currency symbols).
            - If the answer cannot be determined as a numeric value under these rules, return exactly: NOT_ANSWERABLE"""
    elif answer_type =="textual":
            type_prompt= f"""- The answer must be a concise textual response of about 2 lines. 
            - If the question cannot be answered under these rules, return exactly: NOT_ANSWERABLE"""

    prompt= f"""{standard_prompt}

{type_prompt}

If the context is insufficient to answer the question, respond with exactly: NOT_ANSWERABLE

Context:
{serialized_context}

Question:
{question}

"""

    return prompt.strip()


def query_llm(prompt: str) -> str:
    """
    Queries Ollama and returns raw text output.
    """
    response = ollama.chat(
        model="deepseek-r1:latest",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"].strip()

import re

def clean_answer(answer: str) -> str:
    if not answer or not answer.strip():
        return "NOT_ANSWERABLE"

    text = answer.strip()

    # 1️⃣ If equals exists, keep RHS
    if "=" in text:
        text = text.split("=")[-1]

    # 2️⃣ Remove currency symbols and commas
    text = text.replace("$", "")
    text = text.replace(",", "")

    # 3️⃣ Extract ALL integers
    numbers = re.findall(r"-?\d+", text)

    # 4️⃣ If no integer → NOT_ANSWERABLE
    if not numbers:
        return "NOT_ANSWERABLE"

    # 5️⃣ Return ONE integer: the LAST one
    return numbers[-1]


def generate_api_answer(samples : List[Dict])-> List[Dict]:
    ''' Generate answers for a list of samples using the LLM API.'''
    results=[]
    i=1
    for sample in samples:
        context_text= serialize_context(sample["context"])
        prompt= write_prompt(sample["question"], sample["context"], sample["answer_type"])

        model_answer= query_llm(prompt)
        if sample["answer_type"] == "numeric":
            model_answer= clean_answer(model_answer)
            
        results.append({
            "id": sample["id"],
            "question" : sample["question"],
            "model_answer": model_answer, 
            "answer_type": sample.get("answer_type", "unknown"),
            "context": sample["context"]
        })

        time.sleep(3)  # To respect rate limits
        print(i)
        print(model_answer)
        i+=1
    return results

