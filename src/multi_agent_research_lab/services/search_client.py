"""Search client abstraction for ResearcherAgent."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Tavily search adapter with a documented offline fallback."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search Tavily, or return explicit offline references when no key is set."""

        if not self.settings.tavily_api_key:
            return self._offline_results(query, max_results)
        try:
            return self._search_tavily(query, max_results)
        except AgentExecutionError as exc:
            results = self._offline_results(query, max_results)
            for result in results:
                result.metadata["fallback_reason"] = str(exc)
            return results

    @retry(
        retry=retry_if_exception_type(AgentExecutionError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=2),
        reraise=True,
    )
    def _search_tavily(self, query: str, max_results: int) -> list[SourceDocument]:
        """Call Tavily with bounded retry before the caller applies fallback."""

        payload = json.dumps(
            {
                "api_key": self.settings.tavily_api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
            }
        ).encode()
        request = Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.settings.timeout_seconds) as response:  # noqa: S310
                data = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise AgentExecutionError(f"Search request failed: {exc}") from exc
        return [
            SourceDocument(
                title=item.get("title") or "Untitled source",
                url=item.get("url"),
                snippet=item.get("content") or "",
                metadata={"provider": "tavily", "score": item.get("score")},
            )
            for item in data.get("results", [])[:max_results]
            if item.get("content")
        ]

    @staticmethod
    def _offline_results(query: str, max_results: int) -> list[SourceDocument]:
        sources = [
            SourceDocument(
                title="Building effective agents",
                url="https://www.anthropic.com/engineering/building-effective-agents",
                snippet=(
                    "Agent systems benefit from simple, composable workflows, explicit routing, "
                    "tool use, evaluation, and guardrails."
                ),
                metadata={"provider": "offline", "query": query},
            ),
            SourceDocument(
                title="OpenAI Agents SDK: orchestration",
                url="https://developers.openai.com/api/docs/guides/agents/orchestration",
                snippet=(
                    "Agent orchestration can be managed centrally or through handoffs, with "
                    "specialized roles and traceable execution."
                ),
                metadata={"provider": "offline", "query": query},
            ),
            SourceDocument(
                title="LangGraph overview",
                url="https://docs.langchain.com/oss/python/langgraph/overview",
                snippet=(
                    "LangGraph supports stateful agent workflows with persistence, streaming, "
                    "debugging, and human-in-the-loop control."
                ),
                metadata={"provider": "offline", "query": query},
            ),
        ]
        return sources[:max_results]
