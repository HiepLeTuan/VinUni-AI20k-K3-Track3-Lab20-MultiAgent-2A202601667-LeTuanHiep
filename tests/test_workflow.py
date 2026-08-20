from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.search_client import SearchClient


def test_offline_search_is_bounded_and_attributed() -> None:
    client = SearchClient(Settings(tavily_api_key=None))

    results = client.search("multi-agent orchestration", max_results=2)

    assert len(results) == 2
    assert all(source.url for source in results)
    assert all(source.metadata["provider"] == "offline" for source in results)


def test_workflow_produces_cited_answer_and_trace() -> None:
    settings = Settings(max_iterations=6, timeout_seconds=5, tavily_api_key=None)
    initial = ResearchState(request=ResearchQuery(query="Explain multi-agent orchestration"))

    result = MultiAgentWorkflow(settings=settings).run(initial)

    assert result.route_history == ["researcher", "analyst", "writer", "critic", "done"]
    assert result.final_answer
    assert "[1]" in result.final_answer
    assert not result.errors
    assert any(event["name"] == "span" for event in result.trace)
