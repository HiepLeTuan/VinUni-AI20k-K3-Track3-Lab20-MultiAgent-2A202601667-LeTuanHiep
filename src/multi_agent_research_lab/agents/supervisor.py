"""Supervisor / router skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def run(self, state: ResearchState) -> ResearchState:
        """Select the next missing stage, with bounded failure fallback."""

        if state.iteration >= self.settings.max_iterations:
            route = "done"
            state.errors.append("Maximum workflow iterations reached")
        elif state.final_answer:
            route = "critic" if "critic" not in state.route_history else "done"
        elif state.errors:
            route = "writer"
        elif not state.research_notes:
            route = "researcher"
        elif not state.analysis_notes:
            route = "analyst"
        else:
            route = "writer"
        state.record_route(route)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.SUPERVISOR,
                content=route,
                metadata={"iteration": state.iteration},
            )
        )
        state.add_trace_event("route", {"next": route, "iteration": state.iteration})
        return state
