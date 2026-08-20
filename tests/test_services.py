from types import SimpleNamespace
from urllib.error import URLError

import pytest

from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


def test_llm_client_records_usage_and_estimated_cost() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
    )
    provider = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: response))
    )
    settings = Settings(
        openai_api_key=None,
        openai_input_cost_per_million=1,
        openai_output_cost_per_million=2,
    )

    result = LLMClient(settings, provider).complete("system", "user")

    assert result.input_tokens == 100
    assert result.output_tokens == 50
    assert result.cost_usd == pytest.approx(0.0002)


def test_search_client_falls_back_after_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise URLError("provider unavailable")

    monkeypatch.setattr("multi_agent_research_lab.services.search_client.urlopen", fail)
    client = SearchClient(Settings(tavily_api_key="test-key", timeout_seconds=5))

    results = client.search("agent systems", max_results=1)

    assert len(results) == 1
    assert "Search request failed" in results[0].metadata["fallback_reason"]
