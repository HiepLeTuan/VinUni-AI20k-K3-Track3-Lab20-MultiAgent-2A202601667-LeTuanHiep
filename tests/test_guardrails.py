from time import sleep
from typing import Any

import pytest

from multi_agent_research_lab.agents import CriticAgent, ResearcherAgent
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import LLMResponse
from multi_agent_research_lab.services.search_client import SearchClient


class StubLLM:
    configured = True

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        return LLMResponse(
            content='{"verdict":"fail","unsupported_claims":["Claim X"],'
            '"summary":"Claim X lacks evidence."}'
        )


class FailingSearch(SearchClient):
    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        raise AgentExecutionError("search down")


def offline_settings() -> Settings:
    return Settings(
        openai_api_key=None,
        tavily_api_key=None,
        langsmith_api_key=None,
        max_iterations=6,
        timeout_seconds=5,
    )


def test_critic_rejects_unsupported_claims() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain agent systems"),
        sources=[SourceDocument(title="Source", snippet="Supported claim")],
        final_answer="Unsupported Claim X [1]",
    )

    CriticAgent(StubLLM()).run(state)  # type: ignore[arg-type]

    assert state.errors == ["Critic rejected answer: Claim X lacks evidence."]
    assert state.agent_results[-1].metadata["unsupported_claims"] == ["Claim X"]


def test_worker_failure_routes_to_writer_fallback() -> None:
    settings = offline_settings()
    researcher = ResearcherAgent(FailingSearch(settings))

    result = MultiAgentWorkflow(settings=settings, researcher=researcher).run(
        ResearchState(request=ResearchQuery(query="Explain agent systems"))
    )

    assert any("researcher failed" in error for error in result.errors)
    assert result.final_answer
    assert result.route_history[-1] == "done"


def test_workflow_timeout_is_reported() -> None:
    class SlowCompiled:
        def invoke(self, state: ResearchState, config: dict[str, Any]) -> ResearchState:
            sleep(0.1)
            return state

    class SlowGraph:
        def compile(self) -> SlowCompiled:
            return SlowCompiled()

    class SlowWorkflow(MultiAgentWorkflow):
        def build(self) -> SlowGraph:
            return SlowGraph()

    settings = offline_settings().model_copy(update={"timeout_seconds": 0.01})

    with pytest.raises(AgentExecutionError, match="exceeded"):
        SlowWorkflow(settings=settings).run(
            ResearchState(request=ResearchQuery(query="Explain agent systems"))
        )
