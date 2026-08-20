"""Optional critic agent skeleton for bonus work."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Check that the answer exists and references every collected source."""

        answer = state.final_answer or ""
        missing = [
            index
            for index in range(1, len(state.sources) + 1)
            if f"[{index}]" not in answer
        ]
        findings = "Citation check passed." if not missing else f"Missing citations: {missing}"
        if not answer:
            findings = "Final answer is empty."
            state.errors.append(findings)
        elif missing:
            state.errors.append(findings)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=findings,
                metadata={"missing_citations": missing},
            )
        )
        state.add_trace_event("critic", {"missing_citations": missing})
        return state
