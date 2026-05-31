# agents/researcher.py
from models import SubQuestion
from agents.base import call_llm

SYSTEM = """You are a medical research specialist. Your job is to provide accurate, 
evidence-based answers to specific medical sub-questions.

Rules:
- Base answers on established medical knowledge and clinical guidelines
- Be specific and cite reasoning clearly
- Acknowledge uncertainty where it exists
- Keep answers focused and structured
- Do not diagnose or replace clinical judgment
"""

PROMPT = """Original clinical query: {query}

Your specific sub-question to research: {question}

Provide a thorough, evidence-based answer to this sub-question.
Structure your answer with:
- Key findings
- Supporting reasoning
- Any important caveats or limitations
"""


async def research(query: str, sub_question: SubQuestion) -> str:
    prompt = PROMPT.format(
        query=query,
        question=sub_question.question
    )

    answer = await call_llm(prompt=prompt, system=SYSTEM)
    return answer