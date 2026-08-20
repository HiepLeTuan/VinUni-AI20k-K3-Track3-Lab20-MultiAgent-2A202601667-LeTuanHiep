from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routes_through_required_stages() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    supervisor = SupervisorAgent(Settings(max_iterations=6))

    supervisor.run(state)
    assert state.route_history[-1] == "researcher"

    state.research_notes = "Evidence"
    supervisor.run(state)
    assert state.route_history[-1] == "analyst"

    state.analysis_notes = "Analysis"
    supervisor.run(state)
    assert state.route_history[-1] == "writer"


def test_supervisor_stops_at_iteration_limit() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"), iteration=2)
    SupervisorAgent(Settings(max_iterations=2)).run(state)
    assert state.route_history[-1] == "done"
    assert state.errors
