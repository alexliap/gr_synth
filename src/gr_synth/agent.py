"""Pydantic AI agent wired to an OpenAI-compatible vLLM endpoint."""

from __future__ import annotations

from httpx import Timeout
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .config import Settings
from .prompts import SYSTEM_PROMPT


def build_agent(settings: Settings) -> Agent:
    """Construct a Pydantic AI ``Agent`` that posts to the deployed vLLM endpoint.

    Notes:
      - vLLM accepts ``repetition_penalty`` only via ``extra_body``, not as an OpenAI field.
    """
    client = AsyncOpenAI(
        base_url=settings.vllm_base_url,
        api_key=settings.vllm_api_key,
        timeout=Timeout(
            connect=60,
            read=600,
            write=300,
            pool=60,
        ),
    )

    model = OpenAIChatModel(
        settings.vllm_model_id, provider=OpenAIProvider(openai_client=client)
    )

    return Agent(
        model,
        system_prompt=SYSTEM_PROMPT,
        output_type=str,
        model_settings={
            "temperature": 1,
            "top_p": 0.95,
            "max_tokens": settings.max_tokens,
            "extra_body": {"repetition_penalty": settings.repetition_penalty},
        },
    )
