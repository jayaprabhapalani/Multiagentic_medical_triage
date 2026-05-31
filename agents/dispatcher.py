# agents/dispatcher.py
import json
from models import SubQuestion
from agents.base import call_llm
from config import settings

SYSTEM = """You are a medical research dispatcher. Your job is to decompose a complex 
medical query into focused sub-questions that can be researched independently.

Rules:
- Generate exactly the number of sub-questions requested
- Each sub-question must be self-contained and researchable on its own
- Sub-questions must collectively cover the full scope of the original query
- Output ONLY valid JSON, no explanation, no markdown, no code blocks
"""

PROMPT = """Original medical query: {query}

Generate exactly {n} focused sub-questions to fully research this query.

Output format:
[
  {{"id": 1, "question": "..."}},
  {{"id": 2, "question": "..."}},
  ...
]"""


async def dispatch(query: str) -> list[SubQuestion]:
    prompt = PROMPT.format(query=query, n=settings.max_sub_questions)
    raw = await call_llm(prompt=prompt, system=SYSTEM)

    try:
        cleaned = raw.strip()

        # handle ```json ... ``` or ``` ... ```
        if "```" in cleaned:
            parts = cleaned.split("```")
            # parts[1] is the content between first pair of ```
            cleaned = parts[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        # find the JSON array boundaries explicitly
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start == -1 or end == -1:
            raise ValueError("No JSON array found in response")
        cleaned = cleaned[start:end+1]

        data = json.loads(cleaned)
        return [SubQuestion(id=item["id"], question=item["question"]) for item in data]

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise RuntimeError(f"Dispatcher failed to parse LLM output: {e}\nRaw: {raw}")