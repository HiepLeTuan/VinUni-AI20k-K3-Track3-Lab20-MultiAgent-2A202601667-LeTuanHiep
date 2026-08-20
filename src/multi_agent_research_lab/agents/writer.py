"""Writer agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def run(self, state: ResearchState) -> ResearchState:
        """Synthesize the available evidence and include numbered references."""

        findings = state.analysis_notes or state.research_notes or "No evidence was available."
        references = [
            f"[{index}] {source.title}" + (f" - {source.url}" if source.url else "")
            for index, source in enumerate(state.sources, start=1)
        ]
        state.final_answer = (
            f"# {state.request.query}\n\n"
            f"## Findings\n\n{findings}\n\n"
            f"## Sources\n\n" + ("\n".join(references) or "No external sources available.")
        )
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=state.final_answer,
                metadata={"citation_count": len(references)},
            )
        )
        state.add_trace_event("write", {"citation_count": len(references)})
        return state
