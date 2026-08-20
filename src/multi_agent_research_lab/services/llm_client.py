"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import json
from dataclasses import dataclass
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


def parse_json_object(content: str) -> dict[str, Any]:
    """Parse the first JSON object from a model response."""

    start = content.find("{")
    end = content.rfind("}")
    if start < 0 or end < start:
        raise json.JSONDecodeError("No JSON object found", content, 0)
    value = json.loads(content[start : end + 1])
    if not isinstance(value, dict):
        raise json.JSONDecodeError("Expected a JSON object", content, start)
    return value


class LLMClient:
    """Small OpenAI adapter with retries, timeout, and usage capture."""

    def __init__(self, settings: Settings | None = None, client: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = client

    @property
    def configured(self) -> bool:
        """Return whether this client can make a provider call."""

        return self._client is not None or bool(self.settings.openai_api_key)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.settings.openai_api_key:
            raise AgentExecutionError("OPENAI_API_KEY is required for an LLM completion")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise AgentExecutionError("Install the 'llm' extra to use OpenAI") from exc
        self._client = OpenAI(
            api_key=self.settings.openai_api_key,
            timeout=self.settings.timeout_seconds,
        )
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        reraise=True,
    )
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return an OpenAI chat completion and normalized token metadata."""

        try:
            response = self._get_client().chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            content = response.choices[0].message.content
            if not content:
                raise AgentExecutionError("LLM provider returned empty content")
            usage = response.usage
            input_tokens = getattr(usage, "prompt_tokens", None)
            output_tokens = getattr(usage, "completion_tokens", None)
            cost = None
            if input_tokens is not None and output_tokens is not None:
                cost = (
                    input_tokens * self.settings.openai_input_cost_per_million
                    + output_tokens * self.settings.openai_output_cost_per_million
                ) / 1_000_000
            return LLMResponse(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
            )
        except AgentExecutionError:
            raise
        except Exception as exc:
            raise AgentExecutionError(f"LLM completion failed: {exc}") from exc
