# agents/synthesizer.py
from models import ResearchResult
from agents.base import call_llm

SYSTEM = """You are a senior medical research synthesizer. Your job is to combine 
multiple researched findings into a coherent, clinically useful report.

Rules:
- Synthesize findings into a unified narrative, do not just list answers
- Weight findings by their confidence scores
- Highlight areas of uncertainty or low confidence explicitly
- Structure the report for clinical readability
- End with a clear summary and any recommended next steps
- Do not diagnose or replace clinical judgment
"""

PROMPT = """Original clinical query: {query}

The following sub-questions were researched and evaluated:

{findings}

Based on these findings, produce a comprehensive clinical research report that:
1. Opens with a direct answer to the original query
2. Synthesizes the evidence across all findings
3. Flags any low confidence findings (score below 0.6)
4. Closes with clinical implications and recommended next steps
"""

FINDING_TEMPLATE = """
---
Sub-question: {question}
Confidence score: {score}
Critique: {critique}
Findings:
{answer}
"""


async def synthesize(query: str, results: list[ResearchResult]) -> str:
    findings_text = ""
    for r in sorted(results, key=lambda x: x.confidence_score, reverse=True):
        findings_text += FINDING_TEMPLATE.format(
            question=r.sub_question.question,
            score=r.confidence_score,
            critique=r.critique,
            answer=r.answer
        )

    prompt = PROMPT.format(query=query, findings=findings_text)

    report = await call_llm(prompt=prompt, system=SYSTEM)
    return report