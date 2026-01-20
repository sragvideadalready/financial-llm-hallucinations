import json 
import time 
import os
from typing import List, Dict 
from openai import OpenAI


from dotenv import load_dotenv

load_dotenv()

# Retrieve the key
api_key = os.getenv("OPENAI_API_KEY")
print(os.getenv("OPENAI_API_KEY"))


client = OpenAI()

response = client.responses.create(
    model="gpt-5.2",
    input="Write a short bedtime story about a unicorn."
)

print(response.output_text)

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

def write_prompt(question:str, context:Dict)-> str:
    """Construct the prompt for the API call."""
    serialized_context=serialize_context(context)
    prompt=f"""
    You are an expert data analyst. Use the following context to answer the question.
    You are answering a question from a financial QA dataset.

    FIRST determine the answer type:
    - If the answer is NUMERIC → output ONLY the number with units if applicable.
    - If the answer is TEXT → output a short factual statement.

    STRICT OUTPUT RULES:

    1. NUMERIC ANSWER:
    - Output ONLY the numeric value.
    - Units are allowed if needed (%, $, million, billion, years, etc.).
    - NO sentences.
    - NO words before or after.
    - NO explanations.
    - Examples:
        93.5%
        $180 million
        4.2 years
    - Don't answer like:
        "The answer is 93.5%"
        "Approximately $180 million"

    2. TEXT ANSWER:
    - Maximum 2 lines (3 lines ONLY if absolutely necessary).
    - Concise, factual, and direct.
    - NO introductory phrases.
    - NO filler words.
    - NO speculation or reasoning.
    - NO references to the question.
    - Example:
        "Net sales increased due to higher sales volume and favorable product mix."
        "According to the passage, net sales increased..."

    3. If the answer is not explicitly stated or cannot be determined:
    - Output exactly:
        Not stated.
    IMPORTANT:
- Do NOT calculate, derive, estimate, or infer values.
- Do NOT use formulas.
- Do NOT substitute related metrics (e.g., Tier 1 ratio instead of CET1).
- The answer MUST be explicitly stated verbatim in the context.
- If the exact metric or value is not explicitly mentioned, output:
  Not stated.

    DO NOT violate these rules under any circumstances.


    Context:
    {serialized_context}

    Question:
    {question}
    """
    return prompt.strip()



def query_llm(prompt):

    response = client.responses.create(
    model="gpt-5.2", 
    input=prompt
    )
    time.sleep(4) 
    return response.text


def generate_api_answer(samples : List[Dict])-> List[Dict]:
    ''' Generate answers for a list of samples using the LLM API.'''
    results=[]
    i=1
    for sample in samples:
        context_text= serialize_context(sample["context"])
        prompt= write_prompt(sample["question"], sample["context"])

        model_answer= query_llm(prompt)

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

print(query_llm("What is 2 + 2?"))