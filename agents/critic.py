import json
from models import SubQuestion
from agents.base import call_llm

SYSTEM = """You are a medical research critic. Your job is to evaluate the quality 
and reliability of answers to medical sub-questions.

Rules:
- Evaluate answers based on accuracy, completeness, and clinical relevance
- Assign a confidence score between 0.0 and 1.0
- Be specific about what is strong or weak in the answer
- Output ONLY valid JSON, no explanation, no markdown, no code blocks
"""

PROMPT = """Original clinical query: {query}

Sub-question that was researched: {question}

Answer provided by the researcher:
{answer}

Evaluate this answer and respond with:
{{
  "confidence_score": <float between 0.0 and 1.0>,
  "critique": "<2-3 sentences on what is strong, weak, or missing in this answer>"
}}

Scoring guide:
0.0 - 0.3: incomplete, inaccurate, or dangerously misleading
0.4 - 0.6: partially correct but missing important elements
0.7 - 0.8: solid answer with minor gaps
0.9 - 1.0: comprehensive, accurate, well-reasoned
"""


async def critique(
    query: str,
    sub_question: SubQuestion,
    answer: str
) -> tuple[float, str]:

    prompt = PROMPT.format(
        query=query,
        question=sub_question.question,
        answer=answer
    )

    raw = await call_llm(prompt=prompt, system=SYSTEM)

    try:
        cleaned = raw.strip()
        
        # FIX 1: Fixed indentation errors for markdown stripping
        if "```" in cleaned:
            parts = cleaned.split("```")
            cleaned = parts[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        # Find the JSON object boundaries explicitly
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object found in response")
        cleaned = cleaned[start:end+1]

        # FIX 2: Actually parse the JSON string into a Python dictionary!
        data = json.loads(cleaned)

        score = float(data["confidence_score"])
        score = max(0.0, min(1.0, score))

        return score, data["critique"]

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise RuntimeError(f"Critic failed to parse LLM output: {e}\nRaw: {raw}")