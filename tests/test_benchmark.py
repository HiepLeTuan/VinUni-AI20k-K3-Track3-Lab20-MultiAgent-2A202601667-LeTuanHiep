from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


def test_benchmark_scores_citation_coverage() -> None:
    settings = Settings(
        max_iterations=6,
        timeout_seconds=5,
        tavily_api_key=None,
        openai_api_key=None,
        langsmith_api_key=None,
    )

    def runner(query: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=query))
        return MultiAgentWorkflow(settings=settings).run(state)

    _, metrics = run_benchmark("multi-agent", "Explain agent guardrails", runner)

    assert metrics.citation_coverage == 1.0
    assert metrics.failure_rate == 0.0
    assert metrics.quality_score is not None and metrics.quality_score > 0
