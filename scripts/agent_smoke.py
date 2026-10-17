"""Run representative Arabic questions through the agent locally.

Talks to Gemini and Weaviate. Does **not** touch WhatsApp — no GREEN API call
is made, live or otherwise.

    uv run python scripts/agent_smoke.py
    uv run python scripts/agent_smoke.py "شو أوقات دوامكم؟"
"""

from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv

load_dotenv()

from google.genai import types  # noqa: E402

from app.app_utils.runner import get_runner  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.rag.weaviate_store import close_store  # noqa: E402

QUESTIONS = [
    "مرحبا",
    "وين موقع مخبر الخطيب؟",
    "شو أوقات الدوام عندكم؟",
    "كم سعر تحليل فيتامين د؟",
    "هل عندكم سحب دم منزلي؟",
    "شو سعر تذكرة الطيران لباريس؟",  # out of scope: must not be answered
    "نتيجة تحليلي طالعة مرتفعة، شو يعني هالشي؟",  # must refuse to interpret
    "بدي احكي مع موظف",  # must escalate
]


async def ask(runner, question: str, session_id: str) -> str:
    content = types.Content(role="user", parts=[types.Part.from_text(text=question)])
    chunks: list[str] = []
    async for event in runner.run_async(
        user_id=session_id, session_id=session_id, new_message=content
    ):
        if event.is_final_response():
            for part in getattr(getattr(event, "content", None), "parts", None) or []:
                if getattr(part, "text", None):
                    chunks.append(part.text)
    return "\n".join(c.strip() for c in chunks if c.strip()).strip()


async def main() -> int:
    settings = get_settings()
    if not settings.gemini.api_key.is_set:
        print("GOOGLE_API_KEY is not set — cannot run the agent.")
        return 1

    questions = sys.argv[1:] or QUESTIONS
    runner = get_runner()
    failures = 0
    try:
        for index, question in enumerate(questions):
            # One session per question keeps the answers independent.
            session_id = f"smoke-{index}"
            print(f"\n=== س: {question}")
            try:
                answer = await ask(runner, question, session_id)
            except Exception as exc:
                failures += 1
                print(f"    !! {type(exc).__name__}")
                continue
            if not answer:
                failures += 1
                print("    !! empty response")
                continue
            print(f"    ج: {answer}")
        return 1 if failures else 0
    finally:
        close_store()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
