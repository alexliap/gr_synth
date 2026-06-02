"""Pydantic AI agent wired to one or more OpenAI-compatible vLLM endpoints."""

import random

from httpx import Timeout
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .config import Settings
from .prompts import SYSTEM_PROMPT


class RandomAgent:
    """Duck-typed Agent proxy that dispatches each ``run()`` call to a
    randomly-chosen underlying ``Agent``.

    Designed for two deployments: a uniform sample in ``[0, 1)`` is taken and
    agent 1 is picked iff the sample is ``>= 0.5``, else agent 0. Falls back to
    ``random.randrange(len(agents))`` when the pool has more than two entries.
    """

    def __init__(self, agents: list[Agent]) -> None:
        if not agents:
            raise ValueError("RandomAgent requires at least one Agent")
        self._agents = agents
        self.buckets: list[tuple[int, int]] = self._make_buckets()

    async def run(self, *args, **kwargs):
        number = random.random()
        idx = self._is_in_bucket(number)
        return await self._agents[idx].run(*args, **kwargs)

    def _make_buckets(self):
        n = len(self._agents)

        buckets: list[tuple[int, int]] = []
        for i in range(n):
            low, high = i / n, min((i + 1) / n, 1)
            buckets.append((low, high))

        return buckets

    def _is_in_bucket(self, number: float):
        for i, (low, high) in enumerate(self.buckets):
            if low <= number <= high:
                return i
        # Buckets tile [0, 1], so this is only reached for numbers outside that
        # range; clamp to the last agent rather than returning None.
        return len(self.buckets) - 1


def _build_one_agent(base_url: str, settings: Settings) -> Agent:
    """Construct a single Pydantic AI ``Agent`` posting to ``base_url``.

    Notes:
      - vLLM accepts ``repetition_penalty`` only via ``extra_body``, not as an OpenAI field.
    """
    client = AsyncOpenAI(
        base_url=base_url,
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


def build_agent(settings: Settings) -> RandomAgent:
    """Build one ``Agent`` per port in ``settings.vllm_ports`` and wrap them in
    a random-dispatch proxy.

    Each port produces a full URL of the form ``f"{vllm_base_url}:{port}/v1"``.
    All endpoints share ``vllm_api_key`` and ``vllm_model_id``.
    """
    agents = [
        _build_one_agent(f"{settings.vllm_base_url}:{port}/v1", settings)
        for port in settings.vllm_ports
    ]
    return RandomAgent(agents)
