# agents/base.py
import asyncio
import re
from google import genai
from google.genai import types
from config import settings

client = genai.Client(api_key=settings.gemini_api_key)


def _extract_retry_delay(error_message: str) -> float | None:
    match = re.search(r'retry in (\d+(?:\.\d+)?)s', str(error_message), re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


async def call_llm(prompt: str, system: str = None, max_retries: int = 4) -> str:
    contents = []

    if system:
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part(text=f"[System instructions]: {system}\n\n{prompt}")]
            )
        )
    else:
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part(text=prompt)]
            )
        )

    last_error = None

    for attempt in range(max_retries):
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=settings.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    max_output_tokens=settings.max_tokens,
                )
            )
            return response.text

        except Exception as e:
            last_error = e
            error_str = str(e)

            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                # use the delay gemini tells us, otherwise exponential backoff
                suggested = _extract_retry_delay(error_str)
                wait = suggested if suggested else (2 ** attempt) * 5
                print(f"[llm] rate limited, waiting {wait:.1f}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait)
                continue

            # non-429 error, don't retry
            raise RuntimeError(f"LLM call failed: {error_str}")

    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_error}")