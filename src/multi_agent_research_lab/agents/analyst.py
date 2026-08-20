"""Analyst agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def run(self, state: ResearchState) -> ResearchState:
        """Extract source-backed claims and state evidence limitations."""

        claims = [
            f"- Claim {index}: {source.snippet.strip()} [{index}]"
            for index, source in enumerate(state.sources, start=1)
        ]
        limitations = (
            "Evidence limitation: no usable external sources were available."
            if not state.sources
            else "Evidence limitation: snippets should be verified against the full linked sources."
        )
        state.analysis_notes = "Key findings:\n" + "\n".join(claims) + f"\n\n{limitations}"
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=state.analysis_notes,
                metadata={"claim_count": len(claims)},
            )
        )
        state.add_trace_event("analysis", {"claim_count": len(claims)})
        return state
